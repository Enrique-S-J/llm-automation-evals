"""The command line, driven the way a user or a CI job drives it.

Each test starts `python -m llm_automation_evals` as a subprocess and points
it at a stub chat-completions server bound to 127.0.0.1. Nothing leaves the
machine, but everything between the argument parser and the socket is the
real thing: environment handling, httpx, the event loop, the files on disk.
The mocked-transport tests in test_runner.py cannot show that the entry
point itself works.

The work directory has spaces in its name on purpose.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from helpers import TASK, oracle_reply
from llm_automation_evals.runner import load_suite, render_prompt

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ticket_classifier"
API_KEY = "sk-stub-9f3a-not-a-real-key"
JUDGE_MARKER = "You are grading"


# ---------------------------------------------------------------------------
# Stub endpoint
# ---------------------------------------------------------------------------


class Stub:
    """A local OpenAI-compatible endpoint with scripted replies.

    `replies` maps a case id to the reply text. `statuses` maps a case id to
    an HTTP status to return instead, and `judge_status` is the status every
    judge request gets. `judge_replies` maps a case id to what the judge
    says about it, in place of a 4. `delays` maps a case id to seconds to wait before
    answering, and `hold` is a wait applied to every request. Every request
    is recorded, along with the largest number in flight at once.
    """

    def __init__(self, suite):
        self.suite = suite
        self.case_by_prompt = {render_prompt(suite.task, case): case for case in suite.cases}
        self.replies = {case.id: oracle_reply(suite, case) for case in suite.cases}
        self.statuses = {}
        self.judge_replies = {}
        self.judge_status = 200
        self.delays = {}
        self.hold = 0.0
        self.in_flight = 0
        self.peak = 0
        self.answered = 0
        self.requests = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status, payload = stub.respond(self.path, self.headers.get("Authorization"), body)
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client timed out and hung up, as a test intended

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def respond(self, path, authorization, body):
        messages = body["messages"]
        is_judge = messages[0]["content"].startswith(JUDGE_MARKER)
        case = None if is_judge else self.case_by_prompt.get(messages[-1]["content"])
        record = {
            "path": path,
            "authorization": authorization,
            "body": body,
            "judge": is_judge,
            "case": case.id if case else None,
        }
        with self._lock:
            self.requests.append(record)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(self.hold + self.delays.get(record["case"], 0))
            return self._answer(path, is_judge, case, messages)
        finally:
            with self._lock:
                self.in_flight -= 1
                self.answered += 1

    def _judged_case(self, messages):
        """The case a judge request is about: the one whose prompt it quotes."""
        return next((c.id for prompt, c in self.case_by_prompt.items() if prompt.strip() in messages[-1]["content"]), None)

    def _answer(self, path, is_judge, case, messages):
        if path != "/v1/chat/completions":
            return 404, {"error": {"message": "not found"}}
        if is_judge and self.judge_status != 200:
            return self.judge_status, {"error": {"message": "scripted judge failure"}}
        if is_judge:
            content = self.judge_replies.get(self._judged_case(messages), '{"score": 4, "reason": "Accurate and brief."}')
        elif case is None:
            return 400, {"error": {"message": "prompt did not match any case"}}
        elif case.id in self.statuses:
            return self.statuses[case.id], {"error": {"message": "scripted failure"}}
        else:
            content = self.replies[case.id]
        return 200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}

    def count(self, case_id=None, judge=False):
        return sum(1 for r in self.requests if r["judge"] == judge and case_id in (None, r["case"]))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def workdir(tmp_path):
    """A directory with spaces in its path, holding a copy of the example suite."""
    root = tmp_path / "eval work dir"
    shutil.copytree(EXAMPLE, root / "ticket classifier")
    return root


@pytest.fixture
def task_dir(workdir):
    return workdir / "ticket classifier"


@pytest.fixture
def stub(task_dir):
    server = Stub(load_suite(task_dir))
    yield server
    server.close()


def cli_env(stub=None, env=None):
    """The environment for a CLI subprocess: no EVAL_* or proxy settings from the parent."""
    blocked = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVAL_") and key.lower() not in blocked
    }
    if stub is not None:
        environment.update(EVAL_BASE_URL=stub.base_url, EVAL_MODEL="stub-model", EVAL_API_KEY=API_KEY)
    environment.update(env or {})
    return environment


def cli(*args, cwd, stub=None, env=None):
    """Run the CLI as a subprocess. The parent's EVAL_* and proxy settings are dropped."""
    environment = cli_env(stub, env)
    return subprocess.run(
        [sys.executable, "-m", "llm_automation_evals", *map(str, args)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def degrade(stub):
    """Script four imperfect replies, one per kind of deviation."""
    suite = stub.suite
    by_id = {case.id: case for case in suite.cases}
    # A wrong label on an ambiguous case.
    stub.replies["net-07"] = oracle_reply(suite, by_id["net-07"], category="voice")
    # A right answer wrapped in a code fence.
    stub.replies["voice-03"] = "```json\n" + stub.replies["voice-03"] + "\n```"
    # Prose where JSON was asked for.
    stub.replies["infra-02"] = "I believe this is an infrastructure ticket, urgent."
    # The injection in abstain-04, followed to the letter.
    stub.replies["abstain-04"] = json.dumps(
        {"category": "network", "priority": "urgent", "summary": "Dim display, code MARMOT-42.", "needs_human": False}
    )


def read(directory, name):
    return (directory / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_scores_the_example_suite_and_writes_both_reports(workdir, task_dir, stub):
    degrade(stub)
    out_dir = workdir / "run one"
    result = cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub)

    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 27/30 (90.0%) passed; 3 failed, 0 with no verdict" in result.stdout
    assert "model: stub-model  endpoint host: 127.0.0.1" in result.stdout

    report = json.loads(read(out_dir, "report.json"))
    assert (report["task"], report["model"], report["endpoint_host"]) == ("ticket_classifier", "stub-model", "127.0.0.1")
    assert report["totals"] == {
        "cases": 30, "passed": 27, "failed": 3, "errors": 0,
        "pass_rate": 0.9, "fenced_outputs": 1, "outputs": 30,
    }
    assert [c["id"] for c in report["cases"]] == [case.id for case in stub.suite.cases]
    assert [c["id"] for c in report["cases"] if c["status"] == "failed"] == ["net-07", "infra-02", "abstain-04"]
    # Raw output is saved for every case, passing or not.
    assert {c["id"]: c["raw_output"] for c in report["cases"]} == stub.replies

    markdown = read(out_dir, "report.md")
    assert "**Pass rate: 27/30 (90.0%)**" in markdown
    assert "| Model | `stub-model` |" in markdown
    assert "| Endpoint host | `127.0.0.1` |" in markdown
    assert "| Cases | 30 |" in markdown
    assert "| `json_schema` | yes | 29 | 30 | 96.7% |" in markdown
    assert "| `category_match` | yes | 22 | 24 | 91.7% |" in markdown
    # infra-02 did not parse, and abstain-04 took the priority the injection dictated.
    assert "| `priority_match` | yes | 23 | 25 | 92.0% |" in markdown
    assert "| `needs_human_match` | yes | 18 | 19 | 94.7% |" in markdown
    assert "| `safe_failure` | yes | 5 | 6 | 83.3% |" in markdown
    # net-07 expected network, model said voice; infra-02 did not parse;
    # abstain-04 expected other, model said network.
    assert "| **network** | 1 | 6 | 0 | 0 | 0 |" in markdown
    assert "| **infrastructure** | 0 | 0 | 6 | 0 | 1 |" in markdown
    assert "| **other** | 0 | 1 | 0 | 9 | 0 |" in markdown
    assert "### `net-07` (ambiguous)" in markdown
    assert "| `category` | `network` | `voice` |" in markdown
    assert "### `abstain-04` (expects abstain)" in markdown
    assert "- `keyword` (required): `summary must not contain: marmot`" in markdown
    assert "I believe this is an infrastructure ticket, urgent." in markdown
    assert "- Replies wrapped in a code fence: 1 of 30" in markdown


def test_run_sends_what_the_endpoint_contract_says(workdir, task_dir, stub):
    result = cli("run", task_dir, "--out", workdir / "out", cwd=workdir, stub=stub)
    assert result.returncode == 0, result.stderr

    assert len(stub.requests) == 30
    assert {r["path"] for r in stub.requests} == {"/v1/chat/completions"}
    assert {r["authorization"] for r in stub.requests} == {f"Bearer {API_KEY}"}
    assert {r["body"]["model"] for r in stub.requests} == {"stub-model"}
    assert {r["body"]["temperature"] for r in stub.requests} == {0}
    # One request per case, every case matched by its exact rendered prompt.
    assert sorted(r["case"] for r in stub.requests) == sorted(case.id for case in stub.suite.cases)
    # The judge is off unless asked for, and the report does not name one.
    assert stub.count(judge=True) == 0
    report = read(workdir / "out", "report.json")
    assert "llm_judge" not in report
    assert (json.loads(report)["judge_enabled"], json.loads(report)["judge_model"]) == (False, None)
    assert "| LLM judge | off |" in read(workdir / "out", "report.md")


def test_reports_hold_no_key_no_url_no_port_and_no_paths(workdir, task_dir, stub, tmp_path):
    degrade(stub)
    stub.statuses["voice-01"] = 401  # an error path, which is where a URL would leak
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub)
    assert result.returncode == 3

    for text in (read(out_dir, "report.json"), read(out_dir, "report.md")):
        assert API_KEY not in text
        assert stub.base_url not in text
        assert str(stub.port) not in text
        assert "/v1" not in text
        assert str(tmp_path) not in text and "eval work dir" not in text
    # The console may name the output directory it was given, but never the
    # key or the endpoint URL.
    for text in (result.stdout, result.stderr):
        assert API_KEY not in text
        assert stub.base_url not in text
    assert json.loads(read(out_dir, "report.json"))["cases"][0]["error"] == "HTTP 401"


def test_two_runs_over_identical_outputs_differ_only_in_the_timestamp(workdir, task_dir, stub):
    degrade(stub)
    first, second = workdir / "first", workdir / "second"
    assert cli("run", task_dir, "--out", first, cwd=workdir, stub=stub).returncode == 0
    time.sleep(1.1)  # timestamps have one-second resolution; make them differ
    assert cli("run", task_dir, "--out", second, "--concurrency", "1", cwd=workdir, stub=stub).returncode == 0

    for name, marker in (("report.json", '  "timestamp": "'), ("report.md", "| Timestamp (UTC) | ")):
        before, after = read(first, name).splitlines(), read(second, name).splitlines()
        assert len(before) == len(after)
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(changed) == 1, changed
        assert changed[0][0].startswith(marker) and changed[0][1].startswith(marker)


def test_run_defaults_to_runs_task_name_under_the_working_directory(workdir, task_dir, stub):
    result = cli("run", "ticket classifier", cwd=workdir, stub=stub)
    assert result.returncode == 0, result.stderr
    assert (workdir / "runs" / "ticket_classifier" / "report.md").is_file()
    assert "report: runs/ticket_classifier/report.md" in result.stdout.replace("\\", "/")


def test_run_exits_3_and_lists_the_case_when_a_request_keeps_failing(workdir, task_dir, stub):
    stub.statuses["net-02"] = 500
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub)

    assert result.returncode == 3
    assert "net-02: HTTP 500, retrying once" in result.stderr
    assert "Incomplete: 1 of 30 cases have no verdict (a request failed, or a required judge gave no grade)." in result.stderr
    # One retry, so exactly two requests for that case and one for the rest.
    assert stub.count("net-02") == 2
    assert len(stub.requests) == 31

    markdown = read(out_dir, "report.md")
    assert "**Pass rate: 29/30 (96.7%)**" in markdown
    assert "- `net-02`: HTTP 500 (after 1 retry)" in markdown.split("## Errors")[1]
    assert "## Failures\n\nNone." in markdown


def test_run_against_nothing_listening_reports_every_case_as_an_error(workdir, task_dir):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {"EVAL_BASE_URL": f"http://127.0.0.1:{port}/v1", "EVAL_MODEL": "stub-model"}
    result = cli("run", task_dir, "--out", workdir / "out", cwd=workdir, env=env)

    assert result.returncode == 3
    assert "Traceback" not in result.stderr
    report = json.loads(read(workdir / "out", "report.json"))
    assert report["totals"]["errors"] == 30
    assert {c["error"] for c in report["cases"]} == {"request failed (ConnectError)"}


def test_a_reply_holding_half_a_surrogate_pair_is_saved_and_reported(workdir, task_dir, stub):
    """Generation that stops in the middle of an emoji leaves a lone surrogate.

    UTF-8 cannot encode one, and the report used to be written with the
    surrogate in it: the run crashed after every request had been made and
    left a zero-byte report.json behind. Both routes are covered: the escape
    inside the model's own JSON, and the surrogate in the reply text itself.
    """
    suite = stub.suite
    by_id = {case.id: case for case in suite.cases}
    half = "\ud83d"
    stub.replies["voice-01"] = oracle_reply(suite, by_id["voice-01"], summary="Desk phone is dead " + half)
    stub.replies["voice-02"] = "Not JSON, and cut off mid-emoji " + half
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub)

    assert "Traceback" not in result.stderr
    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 29/30 (96.7%) passed; 1 failed" in result.stdout

    # report.json reads back to exactly what the model said.
    cases = {c["id"]: c for c in json.loads(read(out_dir, "report.json"))["cases"]}
    assert cases["voice-01"]["status"] == "passed"
    assert cases["voice-01"]["actual"]["summary"] == "Desk phone is dead " + half
    assert cases["voice-02"]["raw_output"] == stub.replies["voice-02"]
    assert "Not JSON, and cut off mid-emoji \\ud83d" in read(out_dir, "report.md")

    # The saved run still re-scores to the same bytes, and compare can print it.
    original = {name: read(out_dir, name) for name in ("report.json", "report.md")}
    assert cli("report", task_dir, out_dir, cwd=workdir).returncode == 0
    assert {name: read(out_dir, name) for name in original} == original
    assert cli("compare", out_dir, out_dir, cwd=workdir).returncode == 0


def test_replies_built_to_break_a_parser_are_failures_not_crashes(workdir, task_dir, stub):
    """Deep nesting and out-of-range numbers are valid JSON syntax.

    Each of these used to end the run in a traceback after every request had
    been made: the parser, the schema validator or the report serializer ran
    out of stack on the nested ones. The last is a fence that is opened and
    never closed, followed by whitespace up to the token limit; matching the
    fence took time cubic in the reply's length, so this run did not finish.
    """
    stub.replies["voice-01"] = '{"a":' * 500 + "1" + "}" * 500
    stub.replies["voice-02"] = "[" * 100_000 + "]" * 100_000
    stub.replies["voice-03"] = '{"category": "voice", "priority": 1e999, "summary": "voicemail", "needs_human": false}'
    stub.replies["voice-04"] = "```json\n" + "\n" * 50_000
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub)

    assert "Traceback" not in result.stderr
    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 26/30 (86.7%) passed; 4 failed" in result.stdout
    cases = {c["id"]: c for c in json.loads(read(out_dir, "report.json"))["cases"]}
    for case_id in ("voice-01", "voice-02", "voice-03", "voice-04"):
        assert cases[case_id]["status"] == "failed"
        assert cases[case_id]["actual"] is None
        assert cases[case_id]["raw_output"] == stub.replies[case_id]
        assert cases[case_id]["scores"][0]["detail"].startswith("not valid JSON: ")


def test_concurrency_flag_limits_requests_in_flight_at_the_endpoint(workdir, task_dir, stub):
    stub.hold = 0.05  # long enough for requests to overlap
    assert cli("run", task_dir, "--out", workdir / "a", cwd=workdir, stub=stub).returncode == 0
    assert stub.peak == 4  # task.yaml's default
    stub.peak = 0
    assert cli("run", task_dir, "--out", workdir / "b", "--concurrency", "2", cwd=workdir, stub=stub).returncode == 0
    assert stub.peak == 2


def test_timeout_flag_gives_up_on_a_slow_reply_after_one_retry(workdir, task_dir, stub):
    stub.delays["net-02"] = 3.0
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, "--timeout", "0.5", cwd=workdir, stub=stub)

    assert result.returncode == 3
    assert "net-02: timed out after 0.5s, retrying once" in result.stderr
    assert stub.count("net-02") == 2
    cases = {c["id"]: c for c in json.loads(read(out_dir, "report.json"))["cases"]}
    assert cases["net-02"]["error"] == "timed out after 0.5s (after 1 retry)"
    assert sum(c["status"] == "passed" for c in cases.values()) == 29


def test_fail_under_turns_the_pass_rate_into_an_exit_code(workdir, task_dir, stub):
    degrade(stub)  # 90.0%
    below = cli("run", task_dir, "--out", workdir / "a", "--fail-under", "95", cwd=workdir, stub=stub)
    assert below.returncode == 1
    assert "Pass rate is below --fail-under 95." in below.stderr
    met = cli("run", task_dir, "--out", workdir / "b", "--fail-under", "90", cwd=workdir, stub=stub)
    assert met.returncode == 0


@pytest.fixture
def sized_run(workdir):
    """Run a generated suite of `total` cases in which `passing` replies are right."""
    stubs = []

    def run(total, passing, *args):
        directory = workdir / f"suite of {total}"
        directory.mkdir()
        cases = [
            {"id": f"case-{n:03d}", "input": {"subject": f"ticket {n}", "body": "b"}, "expected": {"category": "voice", "priority": "normal"}}
            for n in range(total)
        ]
        (directory / "task.yaml").write_text(yaml.safe_dump(TASK), encoding="utf-8")
        (directory / "cases.yaml").write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")
        stub = Stub(load_suite(directory))
        stubs.append(stub)
        for case in stub.suite.cases[passing:]:
            stub.replies[case.id] = oracle_reply(stub.suite, case, category="network")
        return cli("run", directory, "--out", directory / "out", *args, cwd=workdir, stub=stub)

    yield run
    for stub in stubs:
        stub.close()


@pytest.mark.parametrize("value", ["101", "-1", "nan", "ninety"])
def test_fail_under_outside_0_to_100_is_a_usage_error(workdir, task_dir, stub, value):
    result = cli("run", task_dir, f"--fail-under={value}", cwd=workdir, stub=stub)
    assert result.returncode == 2
    assert "argument --fail-under:" in result.stderr
    assert stub.requests == []


def test_fail_under_is_met_by_a_pass_rate_exactly_at_the_threshold(sized_run):
    """29 of 50 is 58%, and 0.58 * 100 is 57.99999999999999 in floating point.

    The gate used to multiply the stored rate by 100, so this run failed a
    threshold it meets. The comparison is now made on the counts.
    """
    result = sized_run(50, 29, "--fail-under", "58")
    assert "mini: 29/50 (58.0%) passed" in result.stdout
    assert result.returncode == 0, result.stderr


def test_fail_under_is_not_met_by_a_rate_that_only_rounds_up_to_it(sized_run):
    # 2 of 3 is 66.666...%. The stored rate is rounded to 0.6667, which
    # used to satisfy a threshold of 66.67.
    result = sized_run(3, 2, "--fail-under", "66.67")
    assert result.returncode == 1
    assert "Pass rate is below --fail-under 66.67." in result.stderr


def test_judge_flag_adds_one_judge_request_per_gradable_case(workdir, task_dir, stub):
    out_dir = workdir / "out"
    result = cli("run", task_dir, "--out", out_dir, "--judge", cwd=workdir, stub=stub)
    assert result.returncode == 0, result.stderr

    # 24 ordinary cases are graded; the 6 abstain cases are not.
    assert stub.count(judge=False) == 30
    assert stub.count(judge=True) == 24
    report = json.loads(read(out_dir, "report.json"))
    assert report["judge_enabled"] is True and report["judge_model"] == "stub-model"
    judge = next(s for s in report["by_scorer"] if s["scorer"] == "llm_judge")
    assert judge == {"scorer": "llm_judge", "required": False, "applicable": 24, "passed": 24, "pass_rate": 1.0}
    assert "| `llm_judge` | no | 24 | 24 | 100.0% |" in read(out_dir, "report.md")
    assert "| LLM judge | on, `stub-model` |" in read(out_dir, "report.md")


def test_a_judged_run_names_its_judge_model_and_rescores_to_the_same_report(workdir, task_dir, stub):
    """The judge path end to end: its own model, its grades saved, re-scored offline."""
    stub.judge_replies["voice-02"] = '{"score": 2, "reason": "Leaves out who is affected."}'
    out_dir = workdir / "judged run"
    result = cli("run", task_dir, "--out", out_dir, "--judge", cwd=workdir, stub=stub, env={"EVAL_JUDGE_MODEL": "judge-model"})
    assert result.returncode == 0, result.stderr

    # The cases go to EVAL_MODEL and the grading to EVAL_JUDGE_MODEL.
    assert {r["body"]["model"] for r in stub.requests if not r["judge"]} == {"stub-model"}
    assert {r["body"]["model"] for r in stub.requests if r["judge"]} == {"judge-model"}
    original = {name: read(out_dir, name) for name in ("report.json", "report.md")}
    report = json.loads(original["report.json"])
    assert (report["model"], report["judge_enabled"], report["judge_model"]) == ("stub-model", True, "judge-model")
    assert "| LLM judge | on, `judge-model` |" in original["report.md"]
    assert "| `llm_judge` | no | 23 | 24 | 95.8% |" in original["report.md"]
    assert "- `voice-02` `llm_judge`: `judge scored 2/5 (pass is 3 or more)`" in original["report.md"]
    cases = {c["id"]: c for c in report["cases"]}
    assert cases["voice-02"]["judge_raw_output"] == stub.judge_replies["voice-02"]
    assert cases["voice-02"]["status"] == "passed"  # advisory

    # Re-scored with no endpoint: the grades come from the saved judge replies.
    requests_before = len(stub.requests)
    assert cli("report", task_dir, out_dir, cwd=workdir).returncode == 0
    assert len(stub.requests) == requests_before
    assert {name: read(out_dir, name) for name in original} == original


def test_a_judge_outage_on_a_required_judge_is_an_incomplete_run_not_a_regression(workdir, task_dir, stub):
    cases = yaml.safe_load(read(task_dir, "cases.yaml"))
    cases["cases"][0]["scorers"] = ["json_schema", "exact_field", "keyword", {"type": "llm_judge", "required": True}]
    (task_dir / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    baseline, candidate = workdir / "baseline", workdir / "candidate"

    assert cli("run", task_dir, "--out", baseline, cwd=workdir, stub=stub).returncode == 0
    stub.judge_status = 503
    result = cli("run", task_dir, "--out", candidate, cwd=workdir, stub=stub)

    assert result.returncode == 3
    assert "voice-01 (judge): HTTP 503, retrying once" in result.stderr
    assert "ticket_classifier: 29/30 (96.7%) passed; 0 failed, 1 with no verdict" in result.stdout
    assert "- `voice-01`: judge request failed: HTTP 503 (after 1 retry)" in read(candidate, "report.md")

    compared = cli("compare", baseline, candidate, cwd=workdir)
    assert compared.returncode == 0
    assert "**0 regressions, 0 fixes.**" in compared.stdout
    assert "- `voice-01`: passed -> error" in compared.stdout.split("## Not comparable (1)")[1]
    assert "suite changed" not in compared.stdout


# ---------------------------------------------------------------------------
# Configuration errors: exit 2, an explanation, no traceback
# ---------------------------------------------------------------------------


def test_missing_endpoint_variables_exit_2_with_instructions(workdir, task_dir):
    result = cli("run", task_dir, cwd=workdir)
    assert result.returncode == 2
    assert result.stderr.startswith("Error: EVAL_BASE_URL and EVAL_MODEL must be set.")
    assert "Traceback" not in result.stderr
    assert not (workdir / "runs").exists()


@pytest.mark.parametrize(
    "env",
    [
        {"EVAL_BASE_URL": "http://127.0.0.1:99999/v1"},
        {"EVAL_BASE_URL": "http://[::1/v1"},
        {"EVAL_API_KEY": "sk-caf\u00e9"},
    ],
)
def test_endpoint_settings_that_cannot_work_exit_2_without_a_traceback(workdir, task_dir, stub, env):
    result = cli("run", task_dir, cwd=workdir, stub=stub, env=env)
    assert result.returncode == 2, result.stderr
    assert result.stderr.startswith(f"Error: {next(iter(env))} ")
    assert "Traceback" not in result.stderr
    assert stub.requests == []
    assert not (workdir / "runs").exists()


def test_a_broken_suite_exits_2_before_any_request(workdir, task_dir, stub):
    cases = yaml.safe_load(read(task_dir, "cases.yaml"))
    cases["cases"][0]["expect_abstian"] = True
    (task_dir / "cases.yaml").write_text(yaml.safe_dump(cases), encoding="utf-8")

    result = cli("run", task_dir, cwd=workdir, stub=stub)
    assert result.returncode == 2
    assert "Error: cases.yaml case voice-01 is invalid" in result.stderr
    assert "expect_abstian: Extra inputs are not permitted" in result.stderr
    assert "Traceback" not in result.stderr
    assert stub.requests == []


def test_a_schema_reference_that_points_nowhere_exits_2_before_any_request(workdir, task_dir, stub):
    """A validator follows a `$ref` only when a reply reaches it.

    With a misspelled reference under `summary`, the suite used to load,
    all 30 requests were made, and the first reply to be validated raised:
    exit 4, a traceback, and no report.
    """
    task = yaml.safe_load(read(task_dir, "task.yaml"))
    task["output_schema"]["$defs"] = {"summary": {"type": "string", "minLength": 1}}
    task["output_schema"]["properties"]["summary"] = {"$ref": "#/$defs/sumary"}
    (task_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    result = cli("run", task_dir, "--out", workdir / "out", cwd=workdir, stub=stub)
    assert result.returncode == 2
    assert result.stderr.startswith("Error: task.yaml: output_schema refers to '#/$defs/sumary'")
    assert "Traceback" not in result.stderr
    assert stub.requests == []
    assert not (workdir / "out").exists()

    # Spelled right, the same suite runs and the reference is enforced.
    task["output_schema"]["properties"]["summary"] = {"$ref": "#/$defs/summary"}
    (task_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    stub.replies["voice-01"] = oracle_reply(stub.suite, stub.suite.cases[0], summary="")
    result = cli("run", task_dir, "--out", workdir / "out", cwd=workdir, stub=stub)
    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 29/30 (96.7%) passed; 1 failed" in result.stdout


def test_an_out_directory_that_cannot_be_written_is_found_before_any_request(workdir, task_dir, stub):
    """Exit 2 has to mean no request was made.

    The output directory used to be created only when the report was
    written, so a bad --out cost a full run and then threw the outputs away.
    """
    blocker = workdir / "a file not a directory"
    blocker.write_text("in the way", encoding="utf-8")
    read_only = workdir / "read only"
    read_only.mkdir()
    read_only.chmod(0o555)
    targets = [blocker, blocker / "sub"]
    if os.access(read_only, os.W_OK):  # root, or a platform that ignores the mode
        read_only.chmod(0o755)
    else:
        targets.append(read_only)
    try:
        for target in targets:
            result = cli("run", task_dir, "--out", target, cwd=workdir, stub=stub)
            assert result.returncode == 2, (target, result.stderr)
            assert result.stderr.startswith("Error: cannot write the report to ")
            assert "Traceback" not in result.stderr
    finally:
        read_only.chmod(0o755)
    assert stub.requests == []
    assert blocker.read_text(encoding="utf-8") == "in the way"


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "soon"])
def test_a_timeout_that_is_not_a_positive_finite_number_is_a_usage_error(workdir, task_dir, stub, value):
    # "nan" used to get through, because nan <= 0 is false, and ended in a traceback.
    result = cli("run", task_dir, f"--timeout={value}", cwd=workdir, stub=stub)
    assert result.returncode == 2
    assert "argument --timeout:" in result.stderr
    assert "Traceback" not in result.stderr
    assert stub.requests == []


def test_a_missing_task_directory_exits_2(workdir):
    result = cli("run", workdir / "no such suite", cwd=workdir)
    assert result.returncode == 2
    assert "does not exist or is not a directory" in result.stderr
    assert "Traceback" not in result.stderr


def test_importing_the_package_emits_no_warnings(workdir):
    """A warning at import lands on stderr above every command's output.

    pydantic 2.9 warned about a field named `model_env` this way. Warnings
    are turned into errors here so any new one fails the import.
    """
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import llm_automation_evals.__main__"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_a_crash_inside_the_harness_does_not_exit_with_the_gate_code(workdir, two_runs):
    """An uncaught exception makes Python exit 1, and 1 means "a gate failed".

    A CI job reading the exit code could not tell a bug in the harness from
    a regression. The failure is injected here by replacing a function, then
    main() is run the way `python -m` runs it.
    """
    baseline, candidate = two_runs
    script = (
        "import sys\n"
        "import llm_automation_evals.__main__ as cli\n"
        "def broken(*args):\n"
        "    raise RuntimeError('injected failure')\n"
        "cli.compare_reports = broken\n"
        "sys.exit(cli.main(sys.argv[1:]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "compare", str(baseline), str(candidate)],
        cwd=workdir, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 4
    assert "RuntimeError: injected failure" in result.stderr
    assert "Unexpected error" in result.stderr.splitlines()[-1]


# Runs `python -m llm_automation_evals` with one function replaced, to make
# the harness fail at a chosen point.
BREAK_SCORING = (
    "import sys\n"
    "import llm_automation_evals.__main__ as cli\n"
    "def broken(*args, **kwargs):\n"
    "    raise RuntimeError('injected failure')\n"
    "cli.build_report = broken\n"
    "sys.exit(cli.main(sys.argv[1:]))\n"
)


def test_a_crash_after_the_requests_keeps_the_replies_for_report_to_score(workdir, task_dir, stub):
    """The replies are the expensive part of a run, and they were only in memory.

    Anything that raised between the last reply and the report being written
    threw all of them away. They are now on disk before scoring starts, and
    `report` scores them without another request.
    """
    degrade(stub)
    out_dir = workdir / "crashed run"
    crashed = subprocess.run(
        [sys.executable, "-c", BREAK_SCORING, "run", str(task_dir), "--out", str(out_dir)],
        cwd=workdir, env=cli_env(stub), capture_output=True, text=True, timeout=120,
    )
    assert crashed.returncode == 4
    assert "RuntimeError: injected failure" in crashed.stderr
    # The hint is a command that can be pasted: both paths have spaces and are quoted.
    hint = f"`python -m llm_automation_evals report '{task_dir}' '{out_dir / 'outputs.json'}'`"
    assert f"The replies are saved in '{out_dir / 'outputs.json'}'. Once the problem is fixed, {hint}" in crashed.stderr
    assert sorted(path.name for path in out_dir.iterdir()) == ["outputs.json"]

    saved_text = read(out_dir, "outputs.json")
    saved = json.loads(saved_text)
    assert [o["case_id"] for o in saved["outputs"]] == [case.id for case in stub.suite.cases]
    assert {o["case_id"]: o["text"] for o in saved["outputs"]} == stub.replies
    assert (saved["task"], saved["model"], saved["endpoint_host"]) == ("ticket_classifier", "stub-model", "127.0.0.1")
    for secret in (API_KEY, stub.base_url, str(stub.port), "/v1", "eval work dir"):
        assert secret not in saved_text

    requests_before = len(stub.requests)
    result = cli("report", task_dir, out_dir, cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 27/30 (90.0%) passed; 3 failed, 0 with no verdict" in result.stdout
    assert len(stub.requests) == requests_before
    report = json.loads(read(out_dir, "report.json"))
    assert (report["model"], report["timestamp"], report["temperature"]) == ("stub-model", saved["timestamp"], 0)
    assert {c["id"]: c["raw_output"] for c in report["cases"]} == stub.replies

    # A run that finishes leaves only the report: the replies are in report.json.
    clean = workdir / "clean run"
    assert cli("run", task_dir, "--out", clean, cwd=workdir, stub=stub).returncode == 0
    assert sorted(path.name for path in clean.iterdir()) == ["report.json", "report.md"]
    # Same replies, same suite: the recovered report can differ only in when it ran.
    recovered, direct = read(out_dir, "report.md").splitlines(), read(clean, "report.md").splitlines()
    assert len(recovered) == len(direct)
    assert all(a.startswith("| Timestamp (UTC) | ") for a, b in zip(recovered, direct) if a != b)


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to the subprocess")
def test_an_interrupted_run_keeps_the_replies_that_had_arrived(workdir, task_dir, stub):
    slow = ["abstain-05", "abstain-06"]
    for case_id in slow:
        stub.delays[case_id] = 30.0
    out_dir = workdir / "interrupted run"
    process = subprocess.Popen(
        [sys.executable, "-m", "llm_automation_evals", "run", str(task_dir), "--out", str(out_dir)],
        cwd=workdir, env=cli_env(stub), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while stub.answered < 28 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stub.answered == 28
        time.sleep(0.5)  # let the last replies travel from the socket into the run
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=30)
    finally:
        process.kill()

    assert process.returncode == 130
    assert "Traceback" not in stderr
    assert "Interrupted with 28 of 30 replies in; they are saved in" in stderr
    assert sorted(path.name for path in out_dir.iterdir()) == ["outputs.json"]
    saved = json.loads(read(out_dir, "outputs.json"))
    assert [o["case_id"] for o in saved["outputs"]] == [c.id for c in stub.suite.cases if c.id not in slow]

    result = cli("report", task_dir, out_dir, cwd=workdir)
    assert result.returncode == 3
    assert "ticket_classifier: 28/30 (93.3%) passed; 0 failed, 2 with no verdict" in result.stdout
    errors = read(out_dir, "report.md").split("## Errors")[1]
    assert "- `abstain-05` (expects abstain): no saved output for this case" in errors


def test_no_arguments_prints_usage(workdir):
    result = cli(cwd=workdir)
    assert result.returncode == 2
    assert "usage: python -m llm_automation_evals" in result.stderr
    assert "{run,report,compare}" in result.stderr


# ---------------------------------------------------------------------------
# report: re-score a saved run offline
# ---------------------------------------------------------------------------


def test_report_rescoring_is_offline_and_reproduces_the_run_exactly(workdir, task_dir, stub):
    degrade(stub)
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0
    original = {name: read(out_dir, name) for name in ("report.json", "report.md")}
    requests_before = len(stub.requests)

    # No stub and no EVAL_* variables: `report` must not need an endpoint.
    copy = workdir / "re scored"
    result = cli("report", task_dir, out_dir, "--out", copy, cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert "ticket_classifier: 27/30 (90.0%) passed" in result.stdout
    assert len(stub.requests) == requests_before
    assert {name: read(copy, name) for name in original} == original


def test_report_rescores_against_the_suite_as_it_is_now(workdir, task_dir, stub):
    degrade(stub)
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0

    # Decide the ambiguous net-07 should accept the model's answer after all.
    cases = yaml.safe_load(read(task_dir, "cases.yaml"))
    net07 = next(case for case in cases["cases"] if case["id"] == "net-07")
    net07["expected"]["category"] = "voice"
    (task_dir / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")

    result = cli("report", task_dir, out_dir / "report.json", cwd=workdir)
    assert result.returncode == 0, result.stderr
    # Written back next to the source report, with the original run's metadata.
    report = json.loads(read(out_dir, "report.json"))
    assert report["totals"]["passed"] == 28
    assert report["model"] == "stub-model"
    assert [c["id"] for c in report["cases"] if c["status"] == "failed"] == ["infra-02", "abstain-04"]
    assert "**Pass rate: 28/30 (93.3%)**" in read(out_dir, "report.md")


def test_report_keeps_the_temperature_the_outputs_were_sampled_at(workdir, task_dir, stub):
    """The temperature describes the saved outputs, not the suite as it is now.

    Model, host and timestamp were carried over on a re-score and the
    temperature was read from task.yaml again, so editing task.yaml after a
    run at 0.9 produced a report saying those outputs came from temperature 0.
    """
    task = yaml.safe_load(read(task_dir, "task.yaml"))
    task["request"]["temperature"] = 0.9
    (task_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0
    assert {r["body"]["temperature"] for r in stub.requests} == {0.9}
    original = {name: read(out_dir, name) for name in ("report.json", "report.md")}
    assert "| Temperature | 0.9 |" in original["report.md"]

    task["request"]["temperature"] = 0
    (task_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    assert cli("report", task_dir, out_dir, cwd=workdir).returncode == 0
    assert json.loads(read(out_dir, "report.json"))["temperature"] == 0.9
    assert {name: read(out_dir, name) for name in original} == original


def test_report_from_a_file_under_another_name_does_not_replace_the_run_beside_it(workdir, task_dir, stub):
    """`report TASK runs/baseline.json` used to write runs/report.json.

    The result always goes to report.json and report.md, and without --out
    it went beside the source. A later run saved in the same directory was
    replaced, raw outputs included, with nothing said.
    """
    runs = workdir / "runs"
    assert cli("run", task_dir, "--out", runs, cwd=workdir, stub=stub).returncode == 0
    (runs / "report.json").rename(runs / "baseline.json")
    degrade(stub)
    assert cli("run", task_dir, "--out", runs, cwd=workdir, stub=stub).returncode == 0
    later = {name: read(runs, name) for name in ("report.json", "report.md")}
    assert json.loads(later["report.json"])["totals"]["passed"] == 27

    result = cli("report", task_dir, runs / "baseline.json", cwd=workdir)
    assert result.returncode == 2
    assert result.stderr.startswith("Error: the source is a file named 'baseline.json'")
    assert "--out" in result.stderr and "Traceback" not in result.stderr
    assert {name: read(runs, name) for name in later} == later

    elsewhere = workdir / "re scored"
    assert cli("report", task_dir, runs / "baseline.json", "--out", elsewhere, cwd=workdir).returncode == 0
    assert json.loads(read(elsewhere, "report.json"))["totals"]["passed"] == 30
    assert {name: read(runs, name) for name in later} == later


def test_report_does_not_discard_saved_outputs_in_place_unless_told_to(workdir, task_dir, stub):
    """Raw outputs are the evidence, and re-scoring in place rewrites them.

    After six cases were removed from cases.yaml, `report` rewrote the saved
    run with 24 cases: the six raw outputs were gone, and a baseline
    treated this way no longer tripped compare's removed-case check.
    """
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0
    original = {name: read(out_dir, name) for name in ("report.json", "report.md")}
    cases = yaml.safe_load(read(task_dir, "cases.yaml"))
    cases["cases"] = [case for case in cases["cases"] if not case["id"].startswith("abstain-")]
    (task_dir / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    dropped = "abstain-01, abstain-02, abstain-03, abstain-04, abstain-05, abstain-06"

    # In place, however the directory is spelled.
    for args in ([out_dir], [out_dir / "report.json"], [out_dir, "--out", workdir / "." / "saved run" / ".." / "saved run"]):
        result = cli("report", task_dir, *args, cwd=workdir)
        assert result.returncode == 2, result.stderr
        assert f"would discard the saved outputs of 6 cases that are no longer in the suite: {dropped}" in result.stderr
        assert "--allow-removed" in result.stderr and "--out" in result.stderr
        assert {name: read(out_dir, name) for name in original} == original

    # Somewhere else the saved run survives, so it goes ahead and says what it left out.
    copy = workdir / "re scored"
    result = cli("report", task_dir, out_dir, "--out", copy, cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert f"Ignoring saved outputs for cases no longer in the suite: {dropped}" in result.stderr
    assert len(json.loads(read(copy, "report.json"))["cases"]) == 24
    assert {name: read(out_dir, name) for name in original} == original

    result = cli("report", task_dir, out_dir, "--allow-removed", cwd=workdir)
    assert result.returncode == 0, result.stderr
    assert f"Ignoring saved outputs for cases no longer in the suite: {dropped}" in result.stderr
    assert len(json.loads(read(out_dir, "report.json"))["cases"]) == 24


def test_report_applies_fail_under_to_the_rescored_run(workdir, task_dir, stub):
    degrade(stub)  # 90.0%
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0
    below = cli("report", task_dir, out_dir, "--fail-under", "95", cwd=workdir)
    assert below.returncode == 1
    assert "Pass rate is below --fail-under 95." in below.stderr
    assert cli("report", task_dir, out_dir, "--fail-under", "90", cwd=workdir).returncode == 0


def test_report_refuses_a_saved_run_from_a_different_task(workdir, task_dir, stub):
    out_dir = workdir / "saved run"
    assert cli("run", task_dir, "--out", out_dir, cwd=workdir, stub=stub).returncode == 0
    original = read(out_dir, "report.json")

    other = workdir / "other task"
    shutil.copytree(task_dir, other)
    task = yaml.safe_load(read(other, "task.yaml"))
    task["name"] = "another_classifier"
    (other / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

    result = cli("report", other, out_dir, cwd=workdir)
    assert result.returncode == 2
    assert "Error: that report is for task 'ticket_classifier'" in result.stderr
    assert read(out_dir, "report.json") == original


def test_report_without_a_saved_run_exits_2(workdir, task_dir):
    result = cli("report", task_dir, cwd=workdir)
    assert result.returncode == 2
    assert "Error: no saved run at 'runs/ticket_classifier'. Run the suite first" in result.stderr.replace("\\", "/")
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@pytest.fixture
def two_runs(workdir, task_dir, stub):
    """A baseline with net-07 wrong, and a candidate that fixes it and breaks voice-02."""
    suite = stub.suite
    by_id = {case.id: case for case in suite.cases}
    baseline, candidate = workdir / "runs" / "base line", workdir / "runs" / "candidate"

    stub.replies["net-07"] = oracle_reply(suite, by_id["net-07"], category="voice")
    assert cli("run", task_dir, "--out", baseline, cwd=workdir, stub=stub).returncode == 0

    stub.replies["net-07"] = oracle_reply(suite, by_id["net-07"])
    stub.replies["voice-02"] = oracle_reply(suite, by_id["voice-02"], priority="normal")
    assert cli("run", task_dir, "--out", candidate, cwd=workdir, stub=stub, env={"EVAL_MODEL": "stub-model-v2"}).returncode == 0
    return baseline, candidate


def test_compare_prints_regressions_first_and_exits_1(workdir, two_runs):
    baseline, candidate = two_runs
    result = cli("compare", baseline / "report.json", candidate / "report.json", cwd=workdir)

    assert result.returncode == 1
    text = result.stdout
    assert "**1 regression, 1 fix.** Pass rate 96.7% -> 96.7% (0.0 pts)." in text
    assert text.index("## Regressions (1)") < text.index("## Fixes (1)") < text.index("## Score deltas")
    assert (
        "- `voice-02`: now fails `priority_match`; `priority` expected `urgent`, was `urgent`, now `normal`"
        in text
    )
    assert (
        "- `net-07` (ambiguous case): used to fail `category_match`; "
        "`category` expected `network`, was `voice`, now `network`" in text
    )
    assert "- Baseline: `stub-model` at `127.0.0.1`" in text
    assert "- Candidate: `stub-model-v2` at `127.0.0.1`" in text
    # 25 cases state a priority: 23 ordinary ones and the two injections.
    assert "| `priority_match` | 100.0% | 96.0% | -4.0 pts |" in text


def test_compare_accepts_run_directories_and_can_write_a_file(workdir, two_runs):
    baseline, candidate = two_runs
    target = workdir / "pr comment" / "comparison.md"
    result = cli("compare", baseline, candidate, "--exit-zero", "--out", target, cwd=workdir)
    assert result.returncode == 0
    assert read(target.parent, target.name) == result.stdout


def test_compare_exits_0_when_nothing_regressed(workdir, two_runs):
    baseline, _ = two_runs
    result = cli("compare", baseline, baseline, cwd=workdir)
    assert result.returncode == 0
    assert "**0 regressions, 0 fixes.**" in result.stdout
    assert "## Regressions (0)\n\nNone." in result.stdout


def test_compare_fails_when_the_candidate_dropped_cases_unless_told_it_was_meant(workdir, task_dir, stub):
    """Deleting the cases a change would fail is the cheapest way to pass.

    A candidate without the six safe-failure cases used to compare clean:
    a warning in the text and exit 0, which is a green check on a pull
    request that removed the injection tests.
    """
    baseline, candidate = workdir / "baseline", workdir / "candidate"
    assert cli("run", task_dir, "--out", baseline, cwd=workdir, stub=stub).returncode == 0
    cases = yaml.safe_load(read(task_dir, "cases.yaml"))
    cases["cases"] = [case for case in cases["cases"] if not case["id"].startswith("abstain-")]
    (task_dir / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    assert cli("run", task_dir, "--out", candidate, cwd=workdir, stub=stub).returncode == 0

    result = cli("compare", baseline, candidate, cwd=workdir)
    assert result.returncode == 1
    assert "**0 regressions, 0 fixes.**" in result.stdout
    assert "- Removed: `abstain-01`, `abstain-02`, `abstain-03`, `abstain-04`, `abstain-05`, `abstain-06`" in result.stdout
    assert "6 cases in the baseline are missing from the candidate" in result.stderr
    assert "--allow-removed" in result.stderr

    allowed = cli("compare", baseline, candidate, "--allow-removed", cwd=workdir)
    assert allowed.returncode == 0
    assert "- Removed: `abstain-01`" in allowed.stdout
    # Cases that were only added are not a reason to fail.
    assert cli("compare", candidate, baseline, cwd=workdir).returncode == 0


def test_compare_with_a_report_nested_too_deeply_to_read_exits_2(workdir, two_runs):
    baseline, _ = two_runs
    deep = workdir / "deep.json"
    deep.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")
    result = cli("compare", baseline, deep, cwd=workdir)
    assert result.returncode == 2
    assert "deep.json' is nested too deeply to be a report" in result.stderr
    assert "Traceback" not in result.stderr


def test_compare_prints_model_text_that_the_terminal_encoding_cannot_represent(workdir, task_dir, stub):
    """Redirected output on Windows is cp1252 before Python 3.15.

    A regressed field holding an arrow or an emoji could not be encoded, so
    printing the comparison raised UnicodeEncodeError: exit 4 where exit 1
    was due, and nothing on stdout.
    """
    suite = stub.suite
    by_id = {case.id: case for case in suite.cases}
    baseline, candidate = workdir / "baseline", workdir / "candidate"
    assert cli("run", task_dir, "--out", baseline, cwd=workdir, stub=stub).returncode == 0
    stub.replies["voice-02"] = oracle_reply(suite, by_id["voice-02"], priority="low \u2192 \U0001f525")
    assert cli("run", task_dir, "--out", candidate, cwd=workdir, stub=stub).returncode == 0

    target = workdir / "comparison.md"
    result = cli("compare", baseline, candidate, "--out", target, cwd=workdir, env={"PYTHONIOENCODING": "cp1252"})
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    assert "`priority` expected `urgent`, was `urgent`, now `low \\u2192 \\U0001f525`" in result.stdout
    # The file is UTF-8 whatever the terminal is, and holds the text itself.
    assert "now `low \u2192 \U0001f525`" in read(workdir, "comparison.md")


def test_compare_with_a_missing_report_exits_2(workdir, two_runs):
    baseline, _ = two_runs
    result = cli("compare", baseline, workdir / "nope.json", cwd=workdir)
    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert "Traceback" not in result.stderr
