"""Runner behaviour against a mocked HTTP layer. No network.

httpx.MockTransport replaces the transport, so everything above it is the
real code path: request construction, the retry policy, the concurrency
limit, and response parsing.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
import yaml

from helpers import ABSTAIN, TASK, abstain_case, completion, make_case, make_suite, reply
from llm_automation_evals.models import ConfigError, EndpointEnv, RawOutput
from llm_automation_evals.runner import (
    Endpoint,
    collect_outputs,
    load_suite,
    render_prompt,
    run_suite,
)
from llm_automation_evals.scorers import score_case

ENDPOINT = Endpoint(base_url="http://llm.test/v1", model="test-model")
JUDGE_MARKER = "You are grading"


def run(suite, handler, endpoint=ENDPOINT, **kwargs):
    return run_suite(suite, endpoint, transport=httpx.MockTransport(handler), **kwargs)


def is_judge_request(request: httpx.Request) -> bool:
    messages = json.loads(request.content)["messages"]
    return messages[0]["content"].startswith(JUDGE_MARKER)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_is_an_openai_style_chat_completion_at_temperature_zero():
    seen = []

    def handler(request):
        seen.append(request)
        return completion(reply())

    suite = make_suite(make_case(input={"subject": "No dial tone", "body": "Extension 104 is dead."}))
    run(suite, handler)

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 512
    assert body["messages"] == [
        {"role": "system", "content": "Classify the ticket. Reply with JSON only."},
        {"role": "user", "content": "Subject: No dial tone\n\nExtension 104 is dead.\n"},
    ]


def test_api_key_is_sent_as_a_bearer_token_only_when_set():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return completion(reply())

    run(make_suite(), handler)
    run(make_suite(), handler, endpoint=Endpoint("http://llm.test/v1", "m", api_key="sk-test-123"))
    assert seen == [None, "Bearer sk-test-123"]


def test_extra_body_is_merged_and_max_tokens_can_be_dropped():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return completion(reply())

    request = {**TASK["request"], "max_tokens": None, "extra_body": {"seed": 7}}
    run(make_suite(request=request), handler)
    assert seen[0]["seed"] == 7
    assert "max_tokens" not in seen[0]


def test_temperature_and_timeout_come_from_the_task_and_reach_the_request():
    seen = []

    def handler(request):
        seen.append((json.loads(request.content)["temperature"], request.extensions["timeout"]))
        return completion(reply())

    request = {**TASK["request"], "temperature": 0.7, "timeout_seconds": 12}
    run(make_suite(request=request), handler)
    run(make_suite(request=request), handler, timeout=2.5)  # the --timeout override
    assert seen == [
        (0.7, {"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0}),
        (0.7, {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}),
    ]


def test_template_substitution_is_a_single_pass():
    # A ticket that contains a placeholder must be inserted as text. Expanding
    # it again would let ticket content pull other inputs into the prompt.
    case = make_case(input={"subject": "{{body}}", "body": "SECOND"})
    assert render_prompt(make_suite().task, case) == "Subject: {{body}}\n\nSECOND\n"


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_retries_once_on_503_then_succeeds():
    calls = []
    logged = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="overloaded") if len(calls) == 1 else completion(reply())

    (output,) = run(make_suite(), handler, log=logged.append)
    assert len(calls) == 2
    assert output.text == reply()
    assert output.error is None
    assert logged == ["case-1: HTTP 503, retrying once"]


def test_gives_up_after_one_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="overloaded")

    (output,) = run(make_suite(), handler)
    assert len(calls) == 2  # the original request and exactly one retry
    assert output.text is None
    assert output.error == "HTTP 503 (after 1 retry)"


def test_retries_once_on_timeout():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return completion(reply())

    (output,) = run(make_suite(), handler)
    assert len(calls) == 2
    assert output.text == reply()


def test_a_timeout_on_both_attempts_is_recorded_with_the_configured_limit():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    (output,) = run(make_suite(), handler, timeout=2.5)
    assert output.error == "timed out after 2.5s (after 1 retry)"


@pytest.mark.parametrize("status", [400, 401, 404, 429])
def test_does_not_retry_a_client_error(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "nope"}})

    (output,) = run(make_suite(), handler)
    assert len(calls) == 1
    assert output.error == f"HTTP {status}"


def test_does_not_retry_a_connection_failure():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("All connection attempts failed", request=request)

    (output,) = run(make_suite(), handler)
    assert len(calls) == 1
    assert output.error == "request failed (ConnectError)"


def test_waits_the_configured_delay_before_the_retry():
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    def handler(request):
        return httpx.Response(500)

    suite = make_suite(request={**TASK["request"], "retry_delay_seconds": 1.5})
    run(suite, handler, sleep=fake_sleep)
    assert waits == [1.5]


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


def test_records_raw_output_that_is_not_json_instead_of_crashing():
    rambling = "Sure! This looks like a voice ticket.\n\n{category: voice"

    (output,) = run(make_suite(), lambda request: completion(rambling))
    assert output.text == rambling
    assert output.error is None


def test_a_null_reply_is_an_empty_output_not_a_request_error():
    (output,) = run(make_suite(), lambda request: completion(None))
    assert output.text == ""
    assert output.error is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"object": "error"}),
        httpx.Response(200, json={"choices": [{"message": {"content": ["parts"]}}]}),
    ],
)
def test_a_response_that_is_not_a_chat_completion_is_a_request_error(response):
    (output,) = run(make_suite(), lambda request: response)
    assert output.text is None
    assert output.error.startswith("endpoint ")


def test_error_strings_never_contain_the_url_or_credentials():
    """httpx puts the full URL, credentials included, in its exception text.

    The runner's error strings are written into the report, so they are built
    from the status code or the exception class instead.
    """
    endpoint = Endpoint("http://alice:hunter2@llm.test/secret-path/v1", "m")
    logged = []

    def by_subject(request):
        subject = json.loads(request.content)["messages"][1]["content"]
        if "boom" in subject:
            raise httpx.ConnectError("failed for http://alice:hunter2@llm.test/", request=request)
        if "slow" in subject:
            raise httpx.ReadTimeout("timeout for http://alice:hunter2@llm.test/", request=request)
        return httpx.Response(502, text="bad gateway")

    suite = make_suite(
        make_case("a", input={"subject": "boom", "body": ""}),
        make_case("b", input={"subject": "slow", "body": ""}),
        make_case("c", input={"subject": "five-oh-two", "body": ""}),
    )
    outputs = run(suite, by_subject, endpoint=endpoint, log=logged.append)
    errors = [o.error for o in outputs]
    assert errors == [
        "request failed (ConnectError)",
        "timed out after 5s (after 1 retry)",
        "HTTP 502 (after 1 retry)",
    ]
    for text in errors + logged:
        assert "hunter2" not in text and "llm.test" not in text and "secret-path" not in text


def test_the_api_key_is_redacted_from_logged_error_bodies():
    logged = []
    endpoint = Endpoint("http://llm.test/v1", "m", api_key="sk-test-123")

    def handler(request):
        return httpx.Response(401, text="Incorrect API key provided: sk-test-123")

    run(make_suite(), handler, endpoint=endpoint, log=logged.append)
    assert logged == ["case-1: HTTP 401 Incorrect API key provided: [redacted]"]


def test_a_key_that_straddles_the_log_cut_is_redacted_whole():
    """The logged body is cut at 200 characters, and redaction is a string replace.

    Cutting first left the leading part of a key that crossed the cut on
    stderr, where the replace no longer matched it.
    """
    logged = []
    key = "sk-test-SECRETKEY-0123456789"
    endpoint = Endpoint("http://llm.test/v1", "m", api_key=key)
    filler = "x" * 185

    def handler(request):
        return httpx.Response(401, text=f"{filler}{key} is not a valid key")

    run(make_suite(), handler, endpoint=endpoint, log=logged.append)
    assert logged == [f"case-1: HTTP 401 {filler}[redacted] is n"]
    assert "sk-test" not in logged[0]


# ---------------------------------------------------------------------------
# Concurrency and ordering
# ---------------------------------------------------------------------------


def test_respects_the_concurrency_limit():
    in_flight = 0
    peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return completion(reply())

    suite = make_suite(*(make_case(f"case-{n}") for n in range(12)))
    outputs = run(suite, handler, concurrency=3)
    assert len(outputs) == 12
    # Exactly the limit: never above it, and the requests really do overlap.
    assert peak == 3


def test_concurrency_defaults_to_the_task_setting():
    in_flight = 0
    peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return completion(reply())

    suite = make_suite(*(make_case(f"case-{n}") for n in range(12)))
    assert suite.task.request.concurrency == 4
    run(suite, handler)
    assert peak == 4


def test_outputs_follow_case_order_whatever_order_replies_arrive_in():
    finished = []

    async def handler(request):
        subject = json.loads(request.content)["messages"][1]["content"]
        number = int(subject.split("case-")[1].split()[0])
        # The first case is the slowest to answer.
        await asyncio.sleep(0.03 - number * 0.01)
        finished.append(number)
        return completion(reply(summary=f"reply {number}"))

    suite = make_suite(*(make_case(f"case-{n}") for n in range(3)))
    outputs = run(suite, handler)
    assert finished == [2, 1, 0]
    assert [o.case_id for o in outputs] == ["case-0", "case-1", "case-2"]
    assert [json.loads(o.text)["summary"] for o in outputs] == ["reply 0", "reply 1", "reply 2"]


# ---------------------------------------------------------------------------
# LLM judge: off unless asked for
# ---------------------------------------------------------------------------


def judge_aware(seen):
    def handler(request):
        seen.append(request)
        if is_judge_request(request):
            return completion('{"score": 4, "reason": "Accurate."}')
        prompt = json.loads(request.content)["messages"][1]["content"]
        return completion(ABSTAIN if "abstain" in prompt else reply())

    return handler


def test_the_judge_is_never_called_by_default():
    seen = []
    suite = make_suite(make_case("a"), make_case("b"), abstain_case())
    outputs = run(suite, judge_aware(seen))
    assert len(seen) == 3  # one request per case and nothing else
    assert not any(is_judge_request(r) for r in seen)
    assert all(o.judge_text is None and o.judge_error is None for o in outputs)


def test_a_case_that_lists_the_judge_gets_one_judge_call():
    seen = []
    suite = make_suite(make_case("a"), make_case("b", scorers=["json_schema", "llm_judge"]))
    a, b = run(suite, judge_aware(seen))
    assert len(seen) == 3
    assert sum(is_judge_request(r) for r in seen) == 1
    assert a.judge_text is None
    assert b.judge_text == '{"score": 4, "reason": "Accurate."}'


def test_the_judge_flag_grades_every_case_except_abstain_cases():
    seen = []
    suite = make_suite(make_case("a"), make_case("b"), abstain_case())
    outputs = run(suite, judge_aware(seen), judge=True)
    assert len(seen) == 5  # three cases, two judge calls
    assert [o.judge_text is not None for o in outputs] == [True, True, False]


def test_the_judge_uses_its_own_model_when_one_is_configured():
    seen = []
    endpoint = Endpoint("http://llm.test/v1", "small-model", judge_model="big-model")
    run(make_suite(), judge_aware(seen), endpoint=endpoint, judge=True)
    assert [json.loads(r.content)["model"] for r in seen] == ["small-model", "big-model"]


def test_the_judge_is_not_called_when_there_is_nothing_to_grade():
    seen = []

    def handler(request):
        seen.append(request)
        return completion("not json at all")

    (output,) = run(make_suite(), handler, judge=True)
    assert len(seen) == 1
    assert output.judge_text is None


def test_the_judge_flag_needs_a_rubric_before_any_request_is_made():
    seen = []
    with pytest.raises(ConfigError, match="rubric"):
        run(make_suite(judge=None), judge_aware(seen), judge=True)
    assert seen == []


# ---------------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------------

NAMES = EndpointEnv()


def test_endpoint_is_read_from_the_environment():
    endpoint = Endpoint.from_env(
        NAMES,
        {"EVAL_BASE_URL": "http://localhost:11434/v1/", "EVAL_MODEL": "llama3.2:3b", "EVAL_API_KEY": "  "},
    )
    assert endpoint.model == "llama3.2:3b"
    assert endpoint.api_key is None  # blank is the same as unset
    assert endpoint.chat_url == "http://localhost:11434/v1/chat/completions"


@pytest.mark.parametrize(
    "url,host",
    [
        ("http://localhost:11434/v1", "localhost"),
        ("https://alice:hunter2@Gateway.Internal.Test:8443/team-a/v1", "gateway.internal.test"),
        ("http://[::1]:8000/v1", "::1"),
    ],
)
def test_endpoint_host_is_the_hostname_and_nothing_else(url, host):
    assert Endpoint(url, "m").host == host


def test_missing_endpoint_variables_are_named_in_the_error():
    with pytest.raises(ConfigError) as excinfo:
        Endpoint.from_env(NAMES, {})
    message = str(excinfo.value)
    assert message.startswith("EVAL_BASE_URL and EVAL_MODEL must be set")
    assert "http://localhost:11434/v1" in message

    with pytest.raises(ConfigError, match="^EVAL_MODEL must be set"):
        Endpoint.from_env(NAMES, {"EVAL_BASE_URL": "http://localhost:11434/v1"})


@pytest.mark.parametrize(
    "url",
    [
        "llm.test:8000/v1",
        "ftp://llm.test/v1",
        "http:///v1",
        # Each of these used to get past this check and end in a traceback.
        "http://llm.test:99999/v1",
        "http://llm.test:abc/v1",
        "http://[::1/v1",
        "http://llm .test/v1",
        "http://llm.test/v1#section",
        "http://[v1.fe]/v1",  # passes urlsplit; only httpx refuses it
    ],
)
def test_a_base_url_that_is_not_http_is_rejected_without_echoing_it(url):
    with pytest.raises(ConfigError) as excinfo:
        Endpoint.from_env(NAMES, {"EVAL_BASE_URL": url, "EVAL_MODEL": "m"})
    assert "must be an http(s) URL" in str(excinfo.value)
    assert url not in str(excinfo.value)


def test_a_query_string_on_the_base_url_stays_a_query_string():
    """Some gateways version their API with a query parameter.

    The route used to be appended to the raw string, which put it inside the
    query: /v1?api-version=1/chat/completions.
    """
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return completion(reply())

    endpoint = Endpoint.from_env(
        NAMES, {"EVAL_BASE_URL": "http://llm.test/v1/?api-version=2024-06-01", "EVAL_MODEL": "m"}
    )
    run(make_suite(), handler, endpoint=endpoint)
    assert seen == ["http://llm.test/v1/chat/completions?api-version=2024-06-01"]
    assert endpoint.host == "llm.test"


@pytest.mark.parametrize("key", ["sk-caf\u00e9", "sk-one two", "sk-line\nX-Injected: 1", "sk-\x00"])
def test_an_api_key_that_cannot_be_a_header_value_is_rejected_without_echoing_it(key):
    env = {"EVAL_BASE_URL": "http://llm.test/v1", "EVAL_MODEL": "m", "EVAL_API_KEY": key}
    with pytest.raises(ConfigError) as excinfo:
        Endpoint.from_env(NAMES, env)
    assert "EVAL_API_KEY" in str(excinfo.value)
    assert key not in str(excinfo.value) and "sk-" not in str(excinfo.value)


def test_url_credentials_plus_an_api_key_is_rejected():
    """httpx sends URL credentials as Basic auth in place of the Bearer header.

    Accepting both would mean the API key is silently never sent.
    """
    env = {"EVAL_BASE_URL": "http://alice:hunter2@llm.test/v1", "EVAL_MODEL": "m", "EVAL_API_KEY": "sk-1"}
    with pytest.raises(ConfigError) as excinfo:
        Endpoint.from_env(NAMES, env)
    assert "Only one can be sent" in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)


def test_the_task_can_rename_the_environment_variables():
    names = EndpointEnv(base_url_env="STAGING_URL", model_env="STAGING_MODEL")
    endpoint = Endpoint.from_env(names, {"STAGING_URL": "http://llm.test/v1", "STAGING_MODEL": "m"})
    assert endpoint.host == "llm.test"


# ---------------------------------------------------------------------------
# Loading a suite from disk
# ---------------------------------------------------------------------------

CASES = {
    "cases": [
        {"id": "voice-1", "input": {"subject": "s", "body": "b"}, "expected": {"category": "voice", "priority": "normal"}},
        {"id": "abstain-1", "input": {"subject": "", "body": ""}, "expect_abstain": True},
        {"id": "net-1", "input": {"subject": "s", "body": "b"}, "expected": {"category": "network", "priority": "high"}},
    ]
}


def write_suite(directory, task=TASK, cases=CASES):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    (directory / "cases.yaml").write_text(yaml.safe_dump(cases), encoding="utf-8")
    return directory


def with_property(**properties):
    """The test task's output schema with some properties replaced, plus a `$defs`."""
    schema = TASK["output_schema"]
    return {
        **schema,
        "properties": {**schema["properties"], **properties},
        "$defs": {"text": {"$anchor": "text", "type": "string", "minLength": 1}},
    }


def with_case(**changes):
    first = {**CASES["cases"][0], **changes}
    return {"cases": [first, *CASES["cases"][1:]]}


def test_loads_a_suite_from_a_directory_whose_path_has_spaces(tmp_path):
    suite = load_suite(write_suite(tmp_path / "my evals" / "ticket classifier"))
    assert suite.task.name == "mini"
    assert [case.id for case in suite.cases] == ["voice-1", "abstain-1", "net-1"]
    assert suite.cases[1].expect_abstain is True
    # A string path works the same as a Path.
    assert load_suite(str(tmp_path / "my evals" / "ticket classifier")).task.name == "mini"


def test_a_misspelled_case_key_is_an_error_not_a_silent_default(tmp_path):
    """The regression this guards: `expect_abstian: true` being ignored.

    The case would load as an ordinary one, the safe-failure scorer would
    skip it, and the suite would stop checking the thing it was written for.
    """
    cases = {"cases": [{"id": "x", "input": {"subject": "", "body": ""}, "expect_abstian": True}]}
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, cases=cases))
    assert "cases.yaml case x is invalid" in str(excinfo.value)
    assert "expect_abstian: Extra inputs are not permitted" in str(excinfo.value)


@pytest.mark.parametrize(
    "cases,message",
    [
        ({"cases": [CASES["cases"][0], CASES["cases"][0]]}, "case id 'voice-1' is used more than once"),
        (with_case(expected={"category": "telephony"}), "expects category='telephony'"),
        (with_case(input={"subject": "s"}), "no input for template placeholder(s): body"),
        (with_case(expected={}), "case 'voice-1' states nothing to check"),
        (with_case(scorers=["keyword"]), "no scorer applies to case 'voice-1'"),
        (with_case(scorers=[{"type": "keyword", "fields": ["x"]}]), "does not take option(s) fields"),
        (with_case(id="has space"), "cases.yaml case has space is invalid"),
        ({"cases": []}, "non-empty 'cases:' list"),
        (
            with_case(keywords={"must_not_contian": ["marmot"]}),
            "keywords.must_not_contian: Extra inputs are not permitted",
        ),
        # Each of the rest used to load, and quietly checked less than it said.
        (
            with_case(expected={"category": "voice", "priorty": "high"}),
            "case 'voice-1' expects a field 'priorty' that the output schema does not have",
        ),
        (
            with_case(expected={"category": "voice", "summary": "Phone is dead."}),
            "case 'voice-1' expects summary, but no exact_field scorer in its scorer list compares it",
        ),
        (
            with_case(expected={}, expect_abstain=True, scorers=["json_schema"]),
            "case 'voice-1' is flagged expect_abstain, but no safe_failure scorer is in its scorer list",
        ),
        (
            with_case(keywords={"must_not_contain": ["marmot"]}, scorers=["json_schema", "exact_field"]),
            "case 'voice-1' defines keywords, but no keyword scorer is in its scorer list",
        ),
        (
            with_case(
                expected={"category": "voice"},
                scorers=["json_schema", {"type": "exact_field", "fields": ["category", "catgory"]}],
            ),
            "scorer 'exact_field' compares catgory, which case 'voice-1' does not expect",
        ),
    ],
)
def test_suite_mistakes_are_caught_at_load(tmp_path, cases, message):
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, cases=cases))
    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"output_schema": {"type": "objekt"}}, "output_schema is not a valid JSON Schema"),
        ({"confusion_field": "summary"}, "confusion_field 'summary' must be"),
        ({"scorers": ["json_schema", "json_schema"]}, "used more than once: json_schema"),
        ({"scorers": ["exact_match"]}, "task.yaml is invalid"),
        ({"temperature": 0}, "temperature: Extra inputs are not permitted"),
        # A misspelled setting must not fall back to the default in silence.
        ({"request": {"temprature": 0.7}}, "request.temprature: Extra inputs are not permitted"),
        ({"judge": {"rubric": "r", "min_scor": 5}}, "judge.min_scor: Extra inputs are not permitted"),
        (
            {
                "scorers": [
                    "json_schema",
                    {"type": "exact_field", "name": "category_match", "fields": ["category", "catgory"]},
                    {"type": "exact_field", "name": "priority_match", "fields": ["priority"]},
                    "safe_failure",
                ]
            },
            "task.yaml: scorer 'category_match' compares catgory, which no case expects",
        ),
        (
            {"scorers": ["json_schema", {"type": "exact_field", "fields": ["category", "priority"]}]},
            "case 'abstain-1' is flagged expect_abstain, but no safe_failure scorer is in its scorer list",
        ),
        (
            {"abstain": {"category": "declined", "needs_human": True}},
            "task.yaml: abstain sets category='declined', which the output schema does not allow",
        ),
        (
            {"abstain": {"category": "other", "need_human": True}},
            "task.yaml: abstain sets a field 'need_human' that the output schema does not have",
        ),
        # A reference is only followed when a reply reaches it, so a
        # misspelled one used to surface after every request had been made.
        (
            {"output_schema": with_property(summary={"$ref": "#/$defs/sumary"})},
            "task.yaml: output_schema refers to '#/$defs/sumary', which does not resolve inside the schema",
        ),
        (
            {"output_schema": with_property(summary={"$ref": "#no-such-anchor"})},
            "task.yaml: output_schema refers to '#no-such-anchor', which does not resolve inside the schema",
        ),
        (
            {"output_schema": with_property(summary={"items": {"$ref": "https://schemas.example/summary.json"}})},
            "task.yaml: output_schema refers to 'https://schemas.example/summary.json', which does not resolve",
        ),
    ],
)
def test_task_mistakes_are_caught_at_load(tmp_path, change, message):
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, task={**TASK, **change}))
    assert message in str(excinfo.value)


def test_a_key_written_twice_is_an_error_not_last_one_wins(tmp_path):
    """YAML parsers keep the last of two identical keys and say nothing.

    `expect_abstain: true` followed further down by `expect_abstain: false`
    loaded as an ordinary case: the same silent downgrade as a misspelled
    key, by another route.
    """
    directory = write_suite(tmp_path)
    (directory / "cases.yaml").write_text(
        "cases:\n"
        "  - id: abstain-1\n"
        "    expect_abstain: true\n"
        "    input: {subject: '', body: ''}\n"
        "    expect_abstain: false\n"
        "    scorers: [json_schema]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_suite(directory)
    assert "cases.yaml is not valid YAML" in str(excinfo.value)
    assert "key 'expect_abstain' appears more than once" in str(excinfo.value)
    assert "line 5" in str(excinfo.value)

    # A merge key that an explicit key then overrides is not a duplicate.
    (directory / "cases.yaml").write_text(
        "cases:\n"
        "  - &base {id: voice-1, input: {subject: s, body: b}, expected: {category: voice, priority: low}}\n"
        "  - {<<: *base, id: voice-2}\n",
        encoding="utf-8",
    )
    assert [case.id for case in load_suite(directory).cases] == ["voice-1", "voice-2"]


@pytest.mark.parametrize(
    "keywords",
    [
        {"must_contain": [""]},
        {"must_contain": [["voicemail", ""]]},
        {"must_contain": [[]]},
        {"must_not_contain": [""]},
    ],
)
def test_an_empty_keyword_term_is_rejected(tmp_path, keywords):
    # "" is a substring of every summary: as a required term it always
    # passes, and as a forbidden one it always fails.
    with pytest.raises(ConfigError, match="cases.yaml case voice-1 is invalid"):
        load_suite(write_suite(tmp_path, cases=with_case(keywords=keywords)))


def test_a_boolean_subschema_is_handled_as_the_valid_schema_it_is(tmp_path):
    """`priority: true` is JSON Schema for "anything goes".

    The load-time checks assumed every property schema was a mapping and
    called .get on the boolean, which ended the command in a traceback.
    """
    properties = {**TASK["output_schema"]["properties"], "priority": True}
    task = {**TASK, "output_schema": {**TASK["output_schema"], "properties": properties}}
    cases = with_case(expected={"category": "voice", "priority": "whenever"})
    suite = load_suite(write_suite(tmp_path / "a", task=task, cases=cases))
    assert suite.cases[0].expected["priority"] == "whenever"

    # It has no enum, so it cannot be the confusion field.
    with pytest.raises(ConfigError, match="confusion_field 'priority' must be"):
        load_suite(write_suite(tmp_path / "b", task={**task, "confusion_field": "priority"}))


def test_a_reference_inside_the_schema_is_followed(tmp_path):
    for ref in ("#/$defs/text", "#text"):
        task = {**TASK, "output_schema": with_property(summary={"$ref": ref})}
        suite = load_suite(write_suite(tmp_path / ref.strip("#/").replace("/", "-"), task=task))
        scores = score_case(suite.task, suite.cases[0], RawOutput(case_id="voice-1", text=reply()))
        assert scores[0].passed
        empty = score_case(suite.task, suite.cases[0], RawOutput(case_id="voice-1", text=reply(summary="")))
        assert not empty[0].passed and "$.summary" in empty[0].detail


def test_scoring_never_fetches_a_schema_over_the_network():
    """A `$ref` to a URL must not turn scoring into an HTTP client.

    jsonschema's default registry still retrieves remote references, with a
    deprecation warning, so the validator used to fetch whatever URL a suite
    file named, at scoring time, from `report` as well as from `run`. Here
    the URL is served locally and must never be asked for.
    """
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'{"type": "string"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = {"$ref": f"http://127.0.0.1:{server.server_address[1]}/summary.json"}
        # Built directly, so the load-time check is not what stops it.
        task = make_suite(output_schema=with_property(summary=remote)).task
        with pytest.raises(ConfigError, match="does not resolve inside the schema"):
            score_case(task, make_case(), RawOutput(case_id="case-1", text=reply()))
    finally:
        server.shutdown()
        server.server_close()
    assert hits == []


def test_a_file_that_is_not_utf8_is_explained(tmp_path):
    directory = write_suite(tmp_path)
    (directory / "cases.yaml").write_bytes("cases: [{id: caf\u00e9}]".encode("latin-1"))
    with pytest.raises(ConfigError, match="cases.yaml is not UTF-8 text"):
        load_suite(directory)


def test_a_case_can_ask_to_be_graded_on_schema_validity_alone(tmp_path):
    suite = load_suite(write_suite(tmp_path, cases=with_case(expected={}, scorers=["json_schema"])))
    assert suite.cases[0].expected == {}


def test_missing_files_and_directories_are_explained(tmp_path):
    with pytest.raises(ConfigError, match="does not exist or is not a directory"):
        load_suite(tmp_path / "nowhere")
    with pytest.raises(ConfigError, match="task.yaml not found"):
        load_suite(tmp_path)
    (tmp_path / "task.yaml").write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="task.yaml is not valid YAML"):
        load_suite(tmp_path)


def test_collect_outputs_can_be_awaited_directly():
    async def main():
        transport = httpx.MockTransport(lambda request: completion(reply()))
        return await collect_outputs(make_suite(), ENDPOINT, transport=transport)

    (output,) = asyncio.run(main())
    assert output.text == reply()
