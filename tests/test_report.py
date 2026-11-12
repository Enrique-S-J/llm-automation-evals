"""Report building, rendering, and comparison, from fixed outputs.

One fixture run covers every outcome a case can have: pass, fail on a wrong
label, pass with a fenced reply, fail on unparseable output, a request error,
a correct abstain, and an injection that was followed.
"""

import json

import pytest

from helpers import ABSTAIN, LABELS, abstain_case, make_case, make_suite, out, reply
from llm_automation_evals.compare import compare_reports, render_comparison
from llm_automation_evals.models import ConfigError, RawOutput
from llm_automation_evals.report import (
    INVALID,
    build_report,
    fraction,
    load_report,
    outputs_from_report,
    render_json,
    render_markdown,
    write_report,
)

SUITE = make_suite(
    make_case("v1"),
    make_case("v2"),
    make_case("n1", expected={"category": "network", "priority": "high"}),
    make_case("i1", expected={"category": "infrastructure", "priority": "urgent"}, ambiguous=True),
    make_case("o1", expected={"category": "other", "priority": "low"}),
    abstain_case("a1"),
    abstain_case("a2"),
)

OBEYED_INJECTION = reply(category="network", priority="urgent", summary="Routed as instructed.")

OUTPUTS = [
    out(reply(), "v1"),
    out(reply(category="network"), "v2"),
    out("```json\n" + reply(category="network", priority="high") + "\n```", "n1"),
    out("This one is about a server, probably.", "i1"),
    RawOutput(case_id="o1", error="HTTP 503 (after 1 retry)"),
    out(ABSTAIN, "a1"),
    out(OBEYED_INJECTION, "a2"),
]

META = {"model": "test-model", "endpoint_host": "llm.test", "timestamp": "2026-01-02T03:04:05Z"}


def build(outputs=OUTPUTS, suite=SUITE, **meta):
    return build_report(suite, outputs, **{**META, **meta})


def statuses(report):
    return {case.id: case.status for case in report.cases}


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_pass_rate_math():
    report = build()
    assert statuses(report) == {
        "v1": "passed",
        "v2": "failed",
        "n1": "passed",
        "i1": "failed",
        "o1": "error",
        "a1": "passed",
        "a2": "failed",
    }
    totals = report.totals
    assert (totals.cases, totals.passed, totals.failed, totals.errors) == (7, 3, 3, 1)
    # Errors stay in the denominator: 3 of 7, not 3 of 6.
    assert totals.pass_rate == round(3 / 7, 4) == 0.4286
    assert "**Pass rate: 3/7 (42.9%)**" in render_markdown(report)


def test_pass_rate_of_an_empty_run_is_not_a_division_by_zero():
    assert fraction(0, 0) == "n/a (0 cases)"
    assert fraction(27, 30) == "27/30 (90.0%)"


def test_pass_rate_by_scorer_counts_only_the_cases_each_scorer_applies_to():
    summary = {s.scorer: (s.passed, s.applicable, s.pass_rate, s.required) for s in build().by_scorer}
    assert summary == {
        # Six replies; the errored case has nothing to score.
        "json_schema": (5, 6, 0.8333, True),
        "category_match": (2, 4, 0.5, True),
        "priority_match": (3, 4, 0.75, True),
        "safe_failure": (1, 2, 0.5, True),
    }
    # Scorers are listed in the order the task declares them.
    assert list(summary) == ["json_schema", "category_match", "priority_match", "safe_failure"]


def test_fenced_replies_are_counted_but_not_failed():
    report = build()
    assert (report.totals.fenced_outputs, report.totals.outputs) == (1, 6)
    assert statuses(report)["n1"] == "passed"
    assert "- Replies wrapped in a code fence: 1 of 6" in render_markdown(report)


def test_confusion_table_shape_and_cells():
    confusion = build().confusion
    assert confusion.field == "category"
    assert confusion.labels == LABELS
    assert confusion.columns == LABELS + [INVALID]
    # One row per expected label, one column per label plus the catch-all.
    assert len(confusion.matrix) == len(LABELS)
    assert all(len(row) == len(LABELS) + 1 for row in confusion.matrix)
    assert confusion.matrix == [
        [1, 1, 0, 0, 0],  # voice: v1 right, v2 called network
        [0, 1, 0, 0, 0],  # network: n1
        [0, 0, 0, 0, 1],  # infrastructure: i1 did not parse
        [0, 1, 0, 1, 0],  # other: a1 abstained, a2 obeyed the injection; o1 errored
    ]
    # Every case with a reply is counted exactly once.
    assert sum(map(sum, confusion.matrix)) == 6


def test_confusion_table_renders_as_a_markdown_table():
    markdown = render_markdown(build())
    assert "| expected \\ actual | voice | network | infrastructure | other | (invalid) |" in markdown
    assert "| **voice** | 1 | 1 | 0 | 0 | 0 |" in markdown
    assert "| **infrastructure** | 0 | 0 | 0 | 0 | 1 |" in markdown


def test_a_task_without_a_confusion_field_has_no_confusion_table():
    report = build(suite=make_suite(make_case("v1"), confusion_field=None), outputs=[out(reply(), "v1")])
    assert report.confusion is None
    assert "## Confusion" not in render_markdown(report)


# ---------------------------------------------------------------------------
# Per-case content
# ---------------------------------------------------------------------------


def test_failures_show_expected_against_actual_and_the_raw_output():
    markdown = render_markdown(build())
    failures = markdown.split("## Failures")[1].split("## Errors")[0]

    assert "### `v2`\n" in failures
    assert "| `category` | `voice` | `network` |" in failures
    assert "| `priority` | `normal` | `normal` |" in failures
    assert '- `category_match` (required): `category: expected "voice", got "network"`' in failures

    assert "### `i1` (ambiguous)" in failures
    assert "| `category` | `infrastructure` | (not parseable) |" in failures
    assert "This one is about a server, probably." in failures

    assert "### `a2` (expects abstain)" in failures
    assert "| `needs_human` | `true` | `false` |" in failures
    assert "- `safe_failure` (required): `did not abstain; " in failures

    # Passing cases are not listed.
    assert "`v1`" not in failures and "`a1`" not in failures


def test_request_errors_are_listed_apart_from_failures():
    report = build()
    errored = next(case for case in report.cases if case.id == "o1")
    assert errored.scores == [] and errored.raw_output is None
    markdown = render_markdown(report)
    assert "- Errors (a request failed, no verdict): 1" in markdown
    assert "- `o1`: HTTP 503 (after 1 retry)" in markdown.split("## Errors")[1]


def test_raw_output_and_scores_are_saved_for_every_case():
    report = build()
    saved = {case.id: case.raw_output for case in report.cases}
    assert saved == {output.case_id: output.text for output in OUTPUTS}
    i1 = next(case for case in report.cases if case.id == "i1")
    assert i1.actual is None
    assert [score.scorer for score in i1.scores if not score.passed] == [
        "json_schema",
        "category_match",
        "priority_match",
    ]


def test_a_case_with_no_saved_output_is_an_error_not_a_pass():
    report = build(outputs=[o for o in OUTPUTS if o.case_id != "v1"])
    v1 = report.cases[0]
    assert (v1.id, v1.status, v1.error) == ("v1", "error", "no saved output for this case")


def test_advisory_misses_are_reported_without_failing_the_case():
    suite = make_suite(
        make_case("v1"),
        scorers=[
            "json_schema",
            {"type": "exact_field", "name": "priority_match", "fields": ["priority"], "required": False},
        ],
    )
    report = build(suite=suite, outputs=[out(reply(priority="urgent"), "v1")])
    assert statuses(report) == {"v1": "passed"}
    assert report.by_scorer[1].required is False
    markdown = render_markdown(report)
    assert "| `priority_match` | no | 0 | 1 | 0.0% |" in markdown
    assert '- `v1` `priority_match`: `priority: expected "normal", got "urgent"`' in markdown.split(
        "## Advisory misses"
    )[1]


def test_judge_results_appear_only_when_the_judge_ran():
    assert "llm_judge" not in [s.scorer for s in build().by_scorer]
    assert build().judge_model is None
    assert "| LLM judge | off |" in render_markdown(build())

    graded = [out(reply(), "v1", judge_text='{"score": 2, "reason": "Vague."}')]
    report = build(suite=make_suite(make_case("v1")), outputs=graded, judge=True, judge_model="big-model")
    judge = report.cases[0].scores[-1]
    assert (judge.scorer, judge.passed, judge.required, judge.score) == ("llm_judge", False, False, 2.0)
    # Advisory: a low grade is reported and the case still passes.
    assert report.cases[0].status == "passed"
    assert "| LLM judge | on, `big-model` |" in render_markdown(report)


REQUIRED_JUDGE = ["json_schema", "exact_field", {"type": "llm_judge", "required": True}]
JUDGE_DOWN = "HTTP 503 (after 1 retry)"


def test_a_required_judge_that_could_not_be_reached_is_an_error_not_a_failure():
    """A failed judge request is an outage, like a failed model request.

    The case used to be scored "failed", so `run` exited 0 with one fewer
    pass and compare reported a regression, with the explanation that the
    suite must have changed. Neither the suite nor the model had.
    """
    suite = make_suite(make_case("v1", scorers=REQUIRED_JUDGE), make_case("v2"))
    graded = [out(reply(), "v1", judge_text='{"score": 4, "reason": "Fine."}'), out(reply(), "v2")]
    judge_down = [out(reply(), "v1", judge_error=JUDGE_DOWN), out(reply(), "v2")]

    baseline = build(suite=suite, outputs=graded)
    candidate = build(suite=suite, outputs=judge_down)
    assert statuses(baseline) == {"v1": "passed", "v2": "passed"}
    assert statuses(candidate) == {"v1": "error", "v2": "passed"}
    v1 = candidate.cases[0]
    assert v1.error == f"judge request failed: {JUDGE_DOWN}"
    assert v1.raw_output == reply()  # the model did answer, and the answer is kept
    assert (candidate.totals.passed, candidate.totals.failed, candidate.totals.errors) == (1, 0, 1)
    # The reply exists, so the confusion table still counts it.
    assert candidate.confusion.matrix[0][0] == 2
    assert f"- `v1`: judge request failed: {JUDGE_DOWN}" in render_markdown(candidate).split("## Errors")[1]

    comparison = compare_reports(baseline, candidate)
    assert comparison.regressions == []
    assert comparison.not_comparable == [("v1", "passed", "error")]
    assert "suite changed" not in render_comparison(comparison)

    # Re-scoring the saved report reaches the same verdict.
    assert render_json(build(suite=suite, outputs=outputs_from_report(candidate))) == render_json(candidate)


def test_a_wrong_answer_is_a_failure_whether_or_not_the_judge_could_be_reached():
    suite = make_suite(make_case("v1", scorers=REQUIRED_JUDGE))
    report = build(suite=suite, outputs=[out(reply(category="network"), "v1", judge_error=JUDGE_DOWN)])
    assert statuses(report) == {"v1": "failed"}
    assert report.cases[0].error is None


def test_an_advisory_judge_that_could_not_be_reached_does_not_change_the_verdict():
    suite = make_suite(make_case("v1"))
    report = build(suite=suite, outputs=[out(reply(), "v1", judge_error=JUDGE_DOWN)], judge=True)
    assert statuses(report) == {"v1": "passed"}
    assert f"- `v1` `llm_judge`: `judge request failed: {JUDGE_DOWN}`" in render_markdown(report)


# ---------------------------------------------------------------------------
# Markdown safety
# ---------------------------------------------------------------------------


def test_model_text_cannot_break_out_of_the_report_markdown():
    """Everything the model wrote is confined to code spans and fenced blocks.

    A report gets pasted into a pull request. Unescaped, this reply would end
    the table row, close the raw-output fence, and @-mention a stranger.
    """
    hostile_category = "voice | injected cell |\n@someone `tick"
    hostile = json.dumps(
        {
            "category": hostile_category,
            "priority": "normal",
            "summary": "```\n## Not a heading\n@someone",
            "needs_human": False,
        }
    )
    markdown = render_markdown(build(suite=make_suite(make_case("v1")), outputs=[out(hostile, "v1")]))

    row = next(line for line in markdown.splitlines() if line.startswith("| `category` |"))
    # The pipe is escaped and the newline is gone, so the row keeps 3 cells.
    assert row == "| `category` | `voice` | ``voice \\| injected cell \\| @someone `tick`` |"

    # The raw output sits in a fence longer than any backtick run inside it.
    block = markdown.split("Raw output:\n\n")[1]
    assert block.startswith("````text\n")
    assert block.rstrip().endswith("\n````")
    outside_fences = markdown.split("Raw output:")[0]
    assert "## Not a heading" not in outside_fences
    for line in outside_fences.splitlines():
        if "@someone" in line:
            assert line.count("`") >= 2, line


def test_pipes_are_escaped_in_table_cells_and_nowhere_else():
    """`\\|` inside a code span is a pipe in a table cell and two characters anywhere else.

    Escaping everywhere put a stray backslash in every bullet that held a
    pipe, and the keyword scorer joins alternatives with one, so an ordinary
    missed keyword rendered as `voicemail \\| voice mail`.
    """
    case = make_case("v1", keywords={"must_contain": [["voicemail", "voice mail"]]})
    piped = reply(category="voice | network", summary="Phone issue.")
    baseline = build(suite=make_suite(case), outputs=[out(reply(summary="Voicemail is down."), "v1")])
    candidate = build(suite=make_suite(case), outputs=[out(piped, "v1")])
    markdown = render_markdown(candidate)

    # In the failures table the pipe must be escaped or the row gains a cell.
    assert "| `category` | `voice` | `voice \\| network` |" in markdown
    # In a bullet the same escape would show up as a literal backslash.
    assert "- `keyword` (required): `summary missing: voicemail | voice mail`" in markdown
    assert "\\|" not in "\n".join(line for line in markdown.splitlines() if line.startswith("- "))

    comparison = render_comparison(compare_reports(baseline, candidate))
    assert "`category` expected `voice`, was `voice`, now `voice | network`" in comparison
    assert "\\|" not in comparison


def test_long_raw_output_is_truncated_in_markdown_but_kept_whole_in_json():
    rambling = "word " * 600
    report = build(suite=make_suite(make_case("v1")), outputs=[out(rambling, "v1")])
    markdown = render_markdown(report)
    assert "... (truncated; full output is in report.json)" in markdown
    assert len(markdown) < len(rambling)
    assert json.loads(render_json(report))["cases"][0]["raw_output"] == rambling


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_the_same_outputs_give_byte_identical_reports():
    first, second = build(), build()
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)


def test_two_runs_over_the_same_outputs_differ_only_in_the_timestamp():
    earlier = build(timestamp="2026-01-02T03:04:05Z")
    later = build(timestamp="2026-06-07T08:09:10Z")
    for render in (render_json, render_markdown):
        before, after = render(earlier).splitlines(), render(later).splitlines()
        assert len(before) == len(after)
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(changed) == 1
        assert "2026-01-02T03:04:05Z" in changed[0][0] and "2026-06-07T08:09:10Z" in changed[0][1]


def test_output_order_does_not_change_the_report():
    assert render_json(build(outputs=list(reversed(OUTPUTS)))) == render_json(build())
    assert [case.id for case in build(outputs=list(reversed(OUTPUTS))).cases] == [
        "v1", "v2", "n1", "i1", "o1", "a1", "a2",
    ]


def test_written_files_round_trip_and_carry_no_paths(tmp_path):
    report = build()
    out_dir = tmp_path / "run output" / "nested dir"
    json_path, md_path = write_report(out_dir, report)
    assert (json_path.name, md_path.name) == ("report.json", "report.md")

    assert load_report(json_path) == report
    assert load_report(out_dir) == report  # a run directory works too
    for path in (json_path, md_path):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and "\r" not in text
        assert str(tmp_path) not in text and "run output" not in text


def test_a_failed_write_leaves_the_previous_report_as_it_was(tmp_path, monkeypatch):
    """report.json and report.md are a pair and are replaced, never edited.

    Rendering used to happen while the files were being written, so a failure
    in the second file left a new report.json beside a stale report.md, and a
    failure while encoding left report.json empty.
    """
    import llm_automation_evals.report as report_module

    write_report(tmp_path, build())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert sorted(before) == ["report.json", "report.md"]

    def broken(report):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(report_module, "render_markdown", broken)
    with pytest.raises(RuntimeError):
        write_report(tmp_path, build(outputs=CANDIDATE_OUTPUTS))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_lone_surrogates_are_written_as_escapes_and_read_back_unchanged(tmp_path):
    half = "\ud83d"
    report = build(suite=make_suite(make_case("v1")), outputs=[out(reply(summary="cut off " + half), "v1")])
    json_path, md_path = write_report(tmp_path, report)
    assert "cut off \\ud83d" in json_path.read_text(encoding="utf-8")
    assert load_report(json_path) == report
    assert load_report(json_path).cases[0].actual["summary"] == "cut off " + half
    md_path.read_text(encoding="utf-8")  # decodes, so it was written as valid UTF-8


def test_rescoring_a_saved_report_reproduces_it_exactly():
    report = build()
    again = build_report(
        SUITE,
        outputs_from_report(report),
        model=report.model,
        endpoint_host=report.endpoint_host,
        timestamp=report.timestamp,
    )
    assert render_json(again) == render_json(report)
    assert render_markdown(again) == render_markdown(report)


def test_rescoring_picks_up_a_corrected_label_without_new_outputs():
    report = build()
    assert statuses(report)["v2"] == "failed"
    cases = [
        case.model_copy(update={"expected": {"category": "network", "priority": "normal"}})
        if case.id == "v2"
        else case
        for case in SUITE.cases
    ]
    relabelled = build(suite=make_suite(*cases), outputs=outputs_from_report(report))
    assert statuses(relabelled)["v2"] == "passed"
    assert relabelled.totals.passed == 4


def test_report_metadata_is_what_was_passed_in_and_nothing_more():
    data = json.loads(render_json(build()))
    assert list(data) == [
        "schema_version", "task", "model", "endpoint_host", "timestamp", "temperature",
        "judge_enabled", "judge_model", "totals", "by_scorer", "confusion", "cases",
    ]
    assert (data["schema_version"], data["task"], data["model"]) == (1, "mini", "test-model")
    assert data["endpoint_host"] == "llm.test"
    assert data["temperature"] == 0


def test_unreadable_reports_are_explained(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_report(tmp_path / "missing.json")
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid JSON"):
        load_report(garbage)
    future = tmp_path / "future.json"
    future.write_text(render_json(build()).replace('"schema_version": 1', '"schema_version": 2'), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported report schema_version 2"):
        load_report(future)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

# Against OUTPUTS: v1 now gets the category wrong, v2 now gets it right.
CANDIDATE_OUTPUTS = [
    out(reply(category="other"), "v1"),
    out(reply(), "v2"),
    *OUTPUTS[2:],
]


def test_compare_detects_one_regression_and_one_fix():
    comparison = compare_reports(build(), build(outputs=CANDIDATE_OUTPUTS, model="new-model"))

    assert [flip.case_id for flip in comparison.regressions] == ["v1"]
    assert comparison.regressions[0].failing == ["category_match"]
    assert [flip.case_id for flip in comparison.fixes] == ["v2"]
    assert comparison.fixes[0].failing == ["category_match"]  # what used to fail
    assert comparison.still_failing == ["i1", "a2"]
    # The case that errored on both sides is neither a regression nor a fix.
    assert comparison.not_comparable == [("o1", "error", "error")]
    assert comparison.warnings == []


def test_compare_prints_regressions_first():
    text = render_comparison(compare_reports(build(), build(outputs=CANDIDATE_OUTPUTS, model="new-model")))
    assert "**1 regression, 1 fix.** Pass rate 42.9% -> 42.9% (0.0 pts)." in text
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    assert sections == [
        "## Regressions (1)",
        "## Fixes (1)",
        "## Score deltas",
        "## Still failing (2)",
        "## Not comparable (1)",
    ]
    assert (
        "- `v1`: now fails `category_match`; `category` expected `voice`, was `voice`, now `other`"
        in text.split("## Regressions (1)")[1].split("## Fixes")[0]
    )
    assert (
        "- `v2`: used to fail `category_match`; `category` expected `voice`, was `network`, now `voice`"
        in text.split("## Fixes (1)")[1].split("## Score deltas")[0]
    )
    assert "- Baseline: `test-model` at `llm.test`, 2026-01-02T03:04:05Z: 3/7 (42.9%)" in text
    assert "- Candidate: `new-model` at `llm.test`" in text


def test_compare_reports_score_deltas_in_percentage_points():
    # Candidate fixes v2 and breaks nothing: 3/7 -> 4/7.
    candidate = build(outputs=[OUTPUTS[0], out(reply(), "v2"), *OUTPUTS[2:]])
    comparison = compare_reports(build(), candidate)
    assert comparison.regressions == []
    deltas = {d.name: (d.baseline, d.candidate) for d in comparison.deltas}
    assert deltas["overall"] == (0.4286, 0.5714)
    assert deltas["category_match"] == (0.5, 0.75)
    text = render_comparison(comparison)
    assert "**0 regressions, 1 fix.** Pass rate 42.9% -> 57.1% (+14.3 pts)." in text
    assert "| `category_match` | 50.0% | 75.0% | +25.0 pts |" in text
    assert "| `json_schema` | 83.3% | 83.3% | 0.0 pts |" in text
    assert "## Regressions (0)\n\nNone." in text


def test_compare_says_when_a_flip_comes_from_the_suite_and_not_the_model():
    """Same outputs, a relabelled case: a regression, but not the model's doing."""
    relabelled = [
        case.model_copy(update={"expected": {"category": "voice", "priority": "high"}})
        if case.id == "v1"
        else case
        for case in SUITE.cases
    ]
    comparison = compare_reports(build(), build(suite=make_suite(*relabelled)))
    (flip,) = comparison.regressions
    assert (flip.case_id, flip.failing, flip.output_changed) == ("v1", ["priority_match"], False)
    assert (
        "- `v1`: now fails `priority_match`; the model's output is identical in both reports, "
        "so the suite changed, not the model"
    ) in render_comparison(comparison)

    # When the output did change, the line reports the field values instead.
    changed = compare_reports(build(), build(outputs=CANDIDATE_OUTPUTS))
    assert changed.regressions[0].output_changed is True
    assert "suite changed" not in render_comparison(changed)


def test_compare_blames_the_judge_when_only_the_judge_changed_its_mind():
    """Same model output, a required judge that graded it 4 and then 2.

    That is judge noise. It used to be reported as "the suite changed".
    """
    suite = make_suite(make_case("v1", scorers=REQUIRED_JUDGE))
    baseline = build(suite=suite, outputs=[out(reply(), "v1", judge_text='{"score": 4, "reason": "Fine."}')])
    candidate = build(suite=suite, outputs=[out(reply(), "v1", judge_text='{"score": 2, "reason": "Vague."}')])
    comparison = compare_reports(baseline, candidate)
    (flip,) = comparison.regressions
    assert (flip.failing, flip.output_changed, flip.judge_changed) == (["llm_judge"], False, True)
    text = render_comparison(comparison)
    assert (
        "- `v1`: now fails `llm_judge`; the model's output is identical in both reports and the "
        "judge's reply is not, so the judge changed, not the model"
    ) in text
    assert "suite changed" not in text


def test_compare_does_not_call_an_outage_a_regression():
    # v1 passed in the baseline and got no reply in the candidate.
    candidate = build(outputs=[RawOutput(case_id="v1", error="HTTP 503 (after 1 retry)"), *OUTPUTS[1:]])
    comparison = compare_reports(build(), candidate)
    assert comparison.regressions == []
    assert ("v1", "passed", "error") in comparison.not_comparable


def test_compare_warns_when_the_case_sets_differ():
    smaller = make_suite(*[case for case in SUITE.cases if case.id != "a2"], make_case("new"))
    candidate = build(suite=smaller, outputs=[*OUTPUTS, out(reply(), "new")])
    comparison = compare_reports(build(), candidate)
    assert (comparison.added, comparison.removed) == (["new"], ["a2"])
    text = render_comparison(comparison)
    assert "> Warning: The case sets differ" in text
    assert "- Added: `new`" in text and "- Removed: `a2`" in text
