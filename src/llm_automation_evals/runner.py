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
from urllib.parse import urlsplit

import httpx
import yaml
from jsonschema import Draft202012Validator, SchemaError
from pydantic import ValidationError

from .models import Case, ConfigError, EndpointEnv, RawOutput, Task
from .scorers import LlmJudgeScorer, build_scorers, scorers_for

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
    try:
        return yaml.safe_load(text)
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


def _check_suite(task: Task, cases: list[Case]) -> None:
    try:
        Draft202012Validator.check_schema(task.output_schema)
    except SchemaError as exc:
        raise ConfigError(f"{TASK_FILE}: output_schema is not a valid JSON Schema: {exc.message}") from None

    properties = task.output_schema.get("properties", {})
    if task.confusion_field is not None:
        enum = properties.get(task.confusion_field, {}).get("enum")
        if not isinstance(enum, list) or not all(isinstance(v, str) for v in enum):
            raise ConfigError(
                f"{TASK_FILE}: confusion_field '{task.confusion_field}' must be an "
                "output_schema property with a string 'enum'"
            )

    build_scorers(task, task.scorers)
    placeholders = set(_PLACEHOLDER.findall(task.prompt_template))

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ConfigError(f"{CASES_FILE}: case id '{case.id}' is used more than once")
        seen.add(case.id)

        missing = sorted(placeholders - set(case.input))
        if missing:
            raise ConfigError(
                f"{CASES_FILE}: case '{case.id}' has no input for template "
                f"placeholder(s): {', '.join(missing)}"
            )

        # An expectation outside the schema's enum can never be met.
        for field, want in case.expected.items():
            enum = properties.get(field, {}).get("enum")
            if isinstance(enum, list) and want not in enum:
                raise ConfigError(
                    f"{CASES_FILE}: case '{case.id}' expects {field}={want!r}, "
                    f"which the output schema does not allow (one of {enum})"
                )

        if case.scorers is not None:
            build_scorers(task, case.scorers)
        # A case that states no expectation would be graded on schema validity
        # alone and pass on any well-formed reply. That is allowed, but only
        # when asked for, so a forgotten `expected` block cannot do it.
        keywords = case.keywords
        states_something = (
            bool(case.expected)
            or case.expect_abstain
            or (keywords is not None and bool(keywords.must_contain or keywords.must_not_contain))
        )
        if case.scorers is None and not states_something:
            raise ConfigError(
                f"{CASES_FILE}: case '{case.id}' states nothing to check: no 'expected', "
                "'keywords', or 'expect_abstain'. If a well-formed reply is all it should "
                "require, give the case 'scorers: [json_schema]'."
            )
        if not scorers_for(task, case):
            raise ConfigError(
                f"{CASES_FILE}: no scorer applies to case '{case.id}', so it would pass "
                "whatever the model said. Give it 'expected' fields, 'keywords', or "
                "'expect_abstain: true'."
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
        return self.base_url.rstrip("/") + "/chat/completions"

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
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError(
                f"{names.base_url_env} must be an http(s) URL such as "
                "http://localhost:11434/v1; the value set is not one"
            )
        api_key = env.get(names.api_key_env, "").strip() or None
        if api_key and (parts.username or parts.password):
            # httpx turns credentials in the URL into a Basic Authorization
            # header that replaces the Bearer one, so the key would silently
            # never be sent.
            raise ConfigError(
                f"{names.base_url_env} contains credentials and {names.api_key_env} is also "
                "set. Only one can be sent; remove one of them."
            )
        judge_model = env.get(names.judge_model_env, "").strip() or None
        return cls(base_url=base_url, model=model, api_key=api_key, judge_model=judge_model)


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
                        snippet = self._redact(" ".join(response.text.split())[:200])
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
