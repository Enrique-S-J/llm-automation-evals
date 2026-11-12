"""Runner: load a suite, call the model, collect raw outputs.

This is the only module that touches the network. It returns one RawOutput
per case, in case order, and never scores anything; a reply that is not JSON
is recorded as it came back and left for the scorers to fail.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml
from pydantic import ValidationError

from .models import Case, ConfigError, EndpointEnv, RawOutput, Task
from .scorers import (
    ExactFieldScorer,
    KeywordScorer,
    LlmJudgeScorer,
    SafeFailureScorer,
    build_scorers,
    schema_validator,
    scorers_for,
    specs_for,
)

TASK_FILE = "task.yaml"
CASES_FILE = "cases.yaml"

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    task: Task
    cases: list[Case]


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with the same key twice.

    YAML says keys are unique; PyYAML keeps the last one and carries on. In
    a suite file that turns a duplicated `expect_abstain` or `expected` into
    a silent change of what the case checks.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            # A merge key (<<) is not a key of this mapping; the parent
            # class expands it, and a key it brings in may be overridden.
            if isinstance(key_node, yaml.ScalarNode) and key_node.tag != "tag:yaml.org,2002:merge":
                key = self.construct_object(key_node)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"key {key!r} appears more than once", key_node.start_mark
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"{path.name} not found in '{path.parent}'. A task directory holds "
            f"{TASK_FILE} and {CASES_FILE}."
        ) from None
    except OSError as exc:
        raise ConfigError(f"cannot read {path.name}: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path.name} is not UTF-8 text") from None
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from None


def _validation_message(filename: str, exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(top level)"
        lines.append(f"  {where}: {err['msg']}")
    return f"{filename} is invalid:\n" + "\n".join(lines)


def load_suite(task_dir: Union[str, Path]) -> Suite:
    """Read and cross-check task.yaml and cases.yaml from a task directory.

    Mistakes in the suite itself are caught here, before any request is made:
    a mislabelled expectation would otherwise look exactly like a model that
    keeps getting that case wrong.
    """
    root = Path(task_dir)
    if not root.is_dir():
        raise ConfigError(f"task directory '{root}' does not exist or is not a directory")

    try:
        task = Task.model_validate(_read_yaml(root / TASK_FILE))
    except ValidationError as exc:
        raise ConfigError(_validation_message(TASK_FILE, exc)) from None

    raw_cases = _read_yaml(root / CASES_FILE)
    if isinstance(raw_cases, dict) and set(raw_cases) == {"cases"}:
        raw_cases = raw_cases["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError(f"{CASES_FILE} must contain a non-empty 'cases:' list")
    cases = []
    for index, raw in enumerate(raw_cases):
        try:
            cases.append(Case.model_validate(raw))
        except ValidationError as exc:
            label = raw.get("id", f"#{index + 1}") if isinstance(raw, dict) else f"#{index + 1}"
            raise ConfigError(_validation_message(f"{CASES_FILE} case {label}", exc)) from None

    _check_suite(task, cases)
    return Suite(task=task, cases=cases)


def _enum_of(properties: Any, field: str) -> Optional[list]:
    """The `enum` a schema gives a property, if it gives one.

    A property schema may be a boolean, and `properties` may be absent; both
    are valid JSON Schema and neither has an enum.
    """
    schema = properties.get(field) if isinstance(properties, dict) else None
    enum = schema.get("enum") if isinstance(schema, dict) else None
    return enum if isinstance(enum, list) else None


def _check_suite(task: Task, cases: list[Case]) -> None:
    try:
        schema_validator(task.output_schema)
    except ConfigError as exc:
        raise ConfigError(f"{TASK_FILE}: {exc}") from None

    properties = task.output_schema.get("properties")
    if task.confusion_field is not None:
        enum = _enum_of(properties, task.confusion_field)
        if enum is None or not all(isinstance(v, str) for v in enum):
            raise ConfigError(
                f"{TASK_FILE}: confusion_field '{task.confusion_field}' must be an "
                "output_schema property with a string 'enum'"
            )

    # A schema that lists its properties and allows no others makes a
    # misspelled field name detectable: no reply could ever contain it.
    closed = isinstance(properties, dict) and task.output_schema.get("additionalProperties") is False

    def check_meetable(source: str, field: str, want: Any) -> None:
        if closed and field not in properties:
            raise ConfigError(f"{source} a field '{field}' that the output schema does not have")
        enum = _enum_of(properties, field)
        if enum is not None and want not in enum:
            raise ConfigError(
                f"{source} {field}={want!r}, which the output schema does not allow (one of {enum})"
            )

    for field, want in task.abstain.items():
        check_meetable(f"{TASK_FILE}: abstain sets", field, want)

    task_scorers = build_scorers(task, task.scorers)
    # Per task-level exact_field scorer: the fields it names that at least
    # one case expects.
    used: dict[str, set[str]] = {
        s.name: set() for s in task_scorers if isinstance(s, ExactFieldScorer) and s.fields is not None
    }
    task_list_in_use = False
    placeholders = set(_PLACEHOLDER.findall(task.prompt_template))

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ConfigError(f"{CASES_FILE}: case id '{case.id}' is used more than once")
        seen.add(case.id)
        where = f"{CASES_FILE}: case '{case.id}'"

        missing = sorted(placeholders - set(case.input))
        if missing:
            raise ConfigError(
                f"{where} has no input for template placeholder(s): {', '.join(missing)}"
            )

        # An expectation outside the schema can never be met.
        for field, want in case.expected.items():
            check_meetable(f"{where} expects", field, want)

        # A case that states no expectation would be graded on schema validity
        # alone and pass on any well-formed reply. That is allowed, but only
        # when asked for, so a forgotten `expected` block cannot do it.
        keywords = case.keywords
        has_keywords = keywords is not None and bool(keywords.must_contain or keywords.must_not_contain)
        if case.scorers is None and not (case.expected or case.expect_abstain or has_keywords):
            raise ConfigError(
                f"{where} states nothing to check: no 'expected', "
                "'keywords', or 'expect_abstain'. If a well-formed reply is all it should "
                "require, give the case 'scorers: [json_schema]'."
            )
        listed = build_scorers(task, specs_for(task, case))
        in_force = [scorer for scorer in listed if scorer.applies(case)]
        if not in_force:
            raise ConfigError(
                f"{CASES_FILE}: no scorer applies to case '{case.id}', so it would pass "
                "whatever the model said. Give it 'expected' fields, 'keywords', or "
                "'expect_abstain: true'."
            )

        # Everything a case states has to be checked by a scorer in its
        # list. Each scorer skips the cases it has nothing to check on, so a
        # statement with no scorer behind it is not an error at scoring
        # time: the case just passes without it.
        if case.expect_abstain and not any(isinstance(s, SafeFailureScorer) for s in in_force):
            raise ConfigError(
                f"{where} is flagged expect_abstain, but no safe_failure scorer is in its "
                "scorer list, so nothing would check that the model declined"
            )
        if has_keywords and not any(isinstance(s, KeywordScorer) for s in in_force):
            raise ConfigError(
                f"{where} defines keywords, but no keyword scorer is in its scorer list"
            )
        exact = [s for s in listed if isinstance(s, ExactFieldScorer)]
        compared = {field for s in exact for field in s.compared(case)}
        unchecked = [field for field in case.expected if field not in compared]
        if unchecked:
            raise ConfigError(
                f"{where} expects {', '.join(unchecked)}, but no exact_field scorer in its "
                f"scorer list compares {'it' if len(unchecked) == 1 else 'them'}"
            )

        # The reverse: a field an exact_field scorer names that nothing
        # expects. A misspelled name in `fields` removes the comparison.
        if case.scorers is None:
            task_list_in_use = True
            for scorer in exact:
                if scorer.fields is not None:
                    used[scorer.name].update(scorer.compared(case))
        else:
            for scorer in exact:
                unknown = [f for f in scorer.fields or [] if f not in case.expected]
                if unknown:
                    raise ConfigError(
                        f"{CASES_FILE}: scorer '{scorer.name}' compares {', '.join(unknown)}, "
                        f"which case '{case.id}' does not expect"
                    )

    for scorer in task_scorers:
        if task_list_in_use and isinstance(scorer, ExactFieldScorer) and scorer.fields is not None:
            unknown = [f for f in scorer.fields if f not in used[scorer.name]]
            if unknown:
                raise ConfigError(
                    f"{TASK_FILE}: scorer '{scorer.name}' compares {', '.join(unknown)}, which no "
                    "case expects. If the name is right, give a case that field in 'expected' "
                    "or remove it from the scorer."
                )


def render_prompt(task: Task, case: Case) -> str:
    """Fill the task's prompt template from a case's input.

    Substitution is a single pass, so a ticket body that itself contains
    `{{subject}}` is inserted as text and not expanded again.
    """
    return _PLACEHOLDER.sub(lambda m: case.input[m.group(1)], task.prompt_template)


def build_messages(task: Task, case: Case) -> list[dict[str, str]]:
    messages = []
    if task.system_prompt.strip():
        messages.append({"role": "system", "content": task.system_prompt})
    messages.append({"role": "user", "content": render_prompt(task, case)})
    return messages


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: Optional[str] = None
    judge_model: Optional[str] = None

    @property
    def host(self) -> str:
        """Hostname only: no scheme, port, path, or credentials."""
        return urlsplit(self.base_url).hostname or ""

    @property
    def chat_url(self) -> str:
        # The route is added to the path, not to the end of the string, so a
        # query string on the base URL stays a query string.
        parts = urlsplit(self.base_url)
        return urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/chat/completions"))

    @classmethod
    def from_env(cls, names: EndpointEnv, environ: Optional[Mapping[str, str]] = None) -> "Endpoint":
        env = os.environ if environ is None else environ
        base_url = env.get(names.base_url_env, "").strip()
        model = env.get(names.model_env, "").strip()
        missing = [n for n, v in ((names.base_url_env, base_url), (names.model_env, model)) if not v]
        if missing:
            raise ConfigError(
                f"{' and '.join(missing)} must be set. Point {names.base_url_env} at an "
                "OpenAI-compatible base URL including its version path, for example "
                f"http://localhost:11434/v1 for Ollama, and set {names.model_env} to a "
                "model that endpoint serves."
            )
        not_a_url = ConfigError(
            f"{names.base_url_env} must be an http(s) URL such as "
            "http://localhost:11434/v1; the value set is not one"
        )
        # Everything that would make httpx refuse the URL is checked here.
        # Left to the first request, it surfaces as an exception that is not
        # an httpx.HTTPError and so is not handled as a request failure.
        if any(ch.isspace() or not ch.isprintable() for ch in base_url):
            raise not_a_url
        try:
            parts = urlsplit(base_url)
            parts.port  # raises for a port that is not a number in range
        except ValueError:
            raise not_a_url from None
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.fragment:
            raise not_a_url
        api_key = env.get(names.api_key_env, "").strip() or None
        if api_key and not (api_key.isascii() and api_key.isprintable() and " " not in api_key):
            raise ConfigError(
                f"{names.api_key_env} contains a character that cannot be sent in an HTTP "
                "header: whitespace, a control character, or anything outside ASCII"
            )
        if api_key and (parts.username or parts.password):
            # httpx turns credentials in the URL into a Basic Authorization
            # header that replaces the Bearer one, so the key would silently
            # never be sent.
            raise ConfigError(
                f"{names.base_url_env} contains credentials and {names.api_key_env} is also "
                "set. Only one can be sent; remove one of them."
            )
        judge_model = env.get(names.judge_model_env, "").strip() or None
        endpoint = cls(base_url=base_url, model=model, api_key=api_key, judge_model=judge_model)
        try:
            httpx.URL(endpoint.chat_url)
        except httpx.InvalidURL:
            raise not_a_url from None
        return endpoint


# ---------------------------------------------------------------------------
# Calling the model
# ---------------------------------------------------------------------------

Log = Callable[[str], None]


def _extract_content(response: httpx.Response) -> tuple[Optional[str], Optional[str]]:
    """Pull the reply text out of a chat completion: (text, error)."""
    try:
        body = response.json()
    except ValueError:
        return None, "endpoint returned a body that is not JSON"
    try:
        message = body["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError):
        return None, "endpoint response has no choices[0].message.content"
    if content is None:
        # The model produced nothing. That is an answer, and a wrong one.
        return "", None
    if not isinstance(content, str):
        return None, "endpoint response content is not a string"
    return content, None


class _Caller:
    """Sends chat completion requests under one concurrency limit and policy."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: Endpoint,
        suite: Suite,
        concurrency: int,
        log: Log,
        sleep: Callable[[float], Any],
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._settings = suite.task.request
        self._slots = asyncio.Semaphore(concurrency)
        self._log = log
        self._sleep = sleep
        self._headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}

    def _redact(self, text: str) -> str:
        key = self._endpoint.api_key
        return text.replace(key, "[redacted]") if key else text

    async def chat(
        self, label: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[Optional[str], Optional[str]]:
        """One completion, retried once on a 5xx or a timeout: (text, error).

        Error strings are built here from the status code or the exception
        class. httpx's own messages include the full request URL, which can
        carry credentials, and these strings are written into the report.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._settings.temperature,
        }
        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens
        payload.update(self._settings.extra_body)

        async with self._slots:
            failure = ""
            for attempt in (1, 2):
                try:
                    response = await self._client.post(
                        self._endpoint.chat_url, json=payload, headers=self._headers
                    )
                except httpx.TimeoutException:
                    failure = f"timed out after {self._settings.timeout_seconds:g}s"
                except httpx.HTTPError as exc:
                    # Connection refused, DNS failure, protocol error. Not
                    # retried: an endpoint that is not there will not be
                    # there a second later either.
                    failure = f"request failed ({type(exc).__name__})"
                    self._log(f"{label}: {failure}")
                    return None, failure
                else:
                    if response.status_code == 200:
                        return _extract_content(response)
                    failure = f"HTTP {response.status_code}"
                    if response.status_code < 500:
                        # Redact before cutting: a key that crosses the cut
                        # would no longer match and its first part would show.
                        snippet = self._redact(" ".join(response.text.split()))[:200]
                        self._log(f"{label}: {failure} {snippet}".rstrip())
                        return None, failure
                if attempt == 1:
                    self._log(f"{label}: {failure}, retrying once")
                    await self._sleep(self._settings.retry_delay_seconds)
            return None, f"{failure} (after 1 retry)"


async def collect_outputs(
    suite: Suite,
    endpoint: Endpoint,
    *,
    judge: bool = False,
    concurrency: Optional[int] = None,
    timeout: Optional[float] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    log: Log = lambda message: None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> list[RawOutput]:
    """Call the model once per case and return the raw replies in case order.

    `judge` adds the LLM judge to every case it applies to; without it the
    judge is called only for cases that list `llm_judge` themselves.
    `transport` lets tests substitute the HTTP layer.
    """
    task = suite.task
    if timeout is not None:
        task = task.model_copy(update={"request": task.request.model_copy(update={"timeout_seconds": timeout})})
        suite = Suite(task=task, cases=suite.cases)
    limit = concurrency if concurrency is not None else task.request.concurrency
    if limit < 1:
        raise ConfigError("concurrency must be at least 1")

    # Resolve every case's judge up front so a missing rubric stops the run
    # before any request is paid for.
    judges: dict[str, Optional[LlmJudgeScorer]] = {}
    for case in suite.cases:
        found = [s for s in scorers_for(task, case, judge) if isinstance(s, LlmJudgeScorer)]
        judges[case.id] = found[0] if found else None

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(task.request.timeout_seconds), transport=transport
    ) as client:
        caller = _Caller(client, endpoint, suite, limit, log, sleep)

        async def one(case: Case) -> RawOutput:
            text, error = await caller.chat(case.id, endpoint.model, build_messages(task, case))
            output = RawOutput(case_id=case.id, text=text, error=error)
            scorer = judges[case.id]
            candidate = scorer.candidate(output) if scorer is not None else None
            if scorer is not None and candidate is not None:
                output.judge_text, output.judge_error = await caller.chat(
                    f"{case.id} (judge)",
                    endpoint.judge_model or endpoint.model,
                    scorer.build_messages(render_prompt(task, case), candidate),
                )
            return output

        # gather returns results in argument order, whatever order the
        # requests finish in, so the output list always follows cases.yaml.
        return list(await asyncio.gather(*(one(case) for case in suite.cases)))


def run_suite(suite: Suite, endpoint: Endpoint, **kwargs: Any) -> list[RawOutput]:
    """Synchronous wrapper around collect_outputs."""
    return asyncio.run(collect_outputs(suite, endpoint, **kwargs))
