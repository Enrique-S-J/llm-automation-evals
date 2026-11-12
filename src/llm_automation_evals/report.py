"""Report: score a run's raw outputs and render the result.

`build_report` is a pure function of the suite, the raw outputs, and a few
metadata strings. The same inputs give byte-identical report.json and
report.md, so two runs over the same outputs differ only in their timestamp
and a diff between two reports shows changes in behaviour and nothing else.
Nothing here records latency, token counts, retry counts, or filesystem
paths, because each of those would break that.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from .models import (
    CaseResult,
    Confusion,
    ConfigError,
    RawOutput,
    Report,
    SavedOutputs,
    ScorerSummary,
    Totals,
    visible,
)
from .runner import Suite
from .scorers import NO_VERDICT, case_status, parse_output, score_case

SCHEMA_VERSION = 1
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
# Raw outputs of a run that has not produced its report yet. `run` writes it
# before scoring and removes it once the report is on disk.
OUTPUTS_JSON = "outputs.json"

# Confusion-table column for a reply whose label is missing, unparseable, or
# not one of the schema's values.
INVALID = "(invalid)"

_RAW_LIMIT = 1000


def utc_timestamp(now: Optional[datetime] = None) -> str:
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rate(passed: int, total: int) -> Optional[float]:
    return round(passed / total, 4) if total else None


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_report(
    suite: Suite,
    outputs: list[RawOutput],
    *,
    model: str,
    endpoint_host: str,
    timestamp: str,
    judge: bool = False,
    judge_model: Optional[str] = None,
    temperature: Union[int, float, None] = None,
) -> Report:
    """Score every case and aggregate. Cases stay in cases.yaml order.

    `temperature` is the one the outputs were sampled at. It defaults to
    the task's setting, which is right for outputs that were just produced;
    a re-score of saved outputs passes the value saved with them.
    """
    task = suite.task
    by_id = {output.case_id: output for output in outputs}
    results: list[CaseResult] = []
    fenced = 0
    replies = 0

    for case in suite.cases:
        output = by_id.get(case.id) or RawOutput(
            case_id=case.id, error="no saved output for this case"
        )
        expected = dict(case.expected)
        if case.expect_abstain:
            expected = {**task.abstain, **expected}
        result = CaseResult(
            id=case.id,
            status="error",
            ambiguous=case.ambiguous,
            expect_abstain=case.expect_abstain,
            expected=expected,
            raw_output=output.text,
            error=output.error,
            judge_raw_output=output.judge_text,
            judge_error=output.judge_error,
        )
        if output.text is not None:
            parsed = parse_output(output.text)
            replies += 1
            fenced += parsed.fenced
            result.actual = parsed.obj
            result.scores = score_case(task, case, output, judge)
            result.status = case_status(result.scores)
            if result.status == "error":
                # The model answered; a required judge gave no grade.
                result.error = next(
                    s.detail for s in result.scores if s.required and s.data.get(NO_VERDICT)
                )
        elif result.error is None:
            result.error = "no output was recorded"
        results.append(result)

    passed = sum(r.status == "passed" for r in results)
    errors = sum(r.status == "error" for r in results)
    totals = Totals(
        cases=len(results),
        passed=passed,
        failed=len(results) - passed - errors,
        errors=errors,
        pass_rate=_rate(passed, len(results)),
        fenced_outputs=fenced,
        outputs=replies,
    )
    judge_ran = any(r.judge_raw_output is not None or r.judge_error is not None for r in results)
    return Report(
        schema_version=SCHEMA_VERSION,
        task=task.name,
        model=model,
        endpoint_host=endpoint_host,
        timestamp=timestamp,
        temperature=task.request.temperature if temperature is None else temperature,
        judge_enabled=judge,
        judge_model=judge_model if (judge or judge_ran) else None,
        totals=totals,
        by_scorer=_summarise_scorers(results),
        confusion=_confusion(suite, results),
        cases=results,
    )


def _summarise_scorers(results: list[CaseResult]) -> list[ScorerSummary]:
    # In order of first appearance, which follows the task's scorer list.
    order: list[str] = []
    grouped: dict[str, list] = {}
    for result in results:
        for score in result.scores:
            if score.scorer not in grouped:
                order.append(score.scorer)
                grouped[score.scorer] = []
            grouped[score.scorer].append(score)
    summaries = []
    for name in order:
        scores = grouped[name]
        passed = sum(s.passed for s in scores)
        summaries.append(
            ScorerSummary(
                scorer=name,
                required=any(s.required for s in scores),
                applicable=len(scores),
                passed=passed,
                pass_rate=_rate(passed, len(scores)),
            )
        )
    return summaries


def _confusion(suite: Suite, results: list[CaseResult]) -> Optional[Confusion]:
    field = suite.task.confusion_field
    if field is None:
        return None
    labels = list(suite.task.output_schema["properties"][field]["enum"])
    columns = labels + [INVALID]
    matrix = [[0] * len(columns) for _ in labels]
    for result in results:
        want = result.expected.get(field)
        # Cases with no reply have no "actual" to tabulate, and a case that
        # states no expectation for this field has no row.
        if result.raw_output is None or want not in labels:
            continue
        got = result.actual.get(field) if result.actual is not None else None
        column = labels.index(got) if isinstance(got, str) and got in labels else len(labels)
        matrix[labels.index(want)][column] += 1
    return Confusion(field=field, labels=labels, columns=columns, matrix=matrix)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def encodable(text: str) -> str:
    """Write lone surrogates as `\\udXXX` so the text can be encoded as UTF-8.

    A reply that stops in the middle of an emoji carries half of a surrogate
    pair, and UTF-8 has no encoding for one. Every renderer returns its text
    through here, so a report can always be written. In report.json the
    result is the JSON escape for the same code unit, so the file reads back
    to exactly the value that was saved.
    """
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def render_json(report: Report) -> str:
    return encodable(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")


def percent(passed: int, total: int) -> str:
    """A pass rate for display. Always from the counts, so that every place
    a rate is printed rounds the same number the same way."""
    return f"{100 * passed / total:.1f}%" if total else "n/a"


def fraction(passed: int, total: int) -> str:
    if not total:
        return "n/a (0 cases)"
    return f"{passed}/{total} ({percent(passed, total)})"


def code_span(value: Any, limit: int = 80, *, in_table: bool = True) -> str:
    """Render a value as an inline code span that cannot break its surroundings.

    Everything the model wrote goes through here or through `_block`. A
    report is meant to be pasted into a pull request, where bare model text
    could close a table, open a fence, or @-mention someone.

    Inside a table cell a pipe has to be escaped even within a code span, or
    it ends the cell. Outside a table the same backslash is rendered
    literally, so callers that are not writing a cell pass `in_table=False`.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = visible(" ".join(text.split()))
    if not text:
        return "(empty)"
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    if in_table:
        text = text.replace("|", "\\|")
    runs = re.findall(r"`+", text)
    ticks = "`" * (max(map(len, runs)) + 1 if runs else 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def _block(text: str) -> list[str]:
    """Render text as a fenced block whose fence the text cannot close."""
    if len(text) > _RAW_LIMIT:
        text = text[:_RAW_LIMIT] + f"\n... (truncated; full output is in {REPORT_JSON})"
    runs = re.findall(r"`+", text)
    fence = "`" * max(3, max(map(len, runs)) + 1 if runs else 3)
    return [f"{fence}text", *map(visible, text.splitlines()), fence]


def _flags(result: CaseResult) -> str:
    flags = []
    if result.expect_abstain:
        flags.append("expects abstain")
    if result.ambiguous:
        flags.append("ambiguous")
    return f" ({', '.join(flags)})" if flags else ""


def render_markdown(report: Report) -> str:
    totals = report.totals
    judge = "off"
    if report.judge_model:
        judge = f"on, {code_span(report.judge_model)}" if report.judge_enabled else f"per case, {code_span(report.judge_model)}"
    lines = [
        f"# Eval report: {report.task}",
        "",
        "| | |",
        "|---|---|",
        f"| Model | {code_span(report.model)} |",
        f"| Endpoint host | {code_span(report.endpoint_host)} |",
        f"| Timestamp (UTC) | {report.timestamp} |",
        f"| Cases | {totals.cases} |",
        f"| Temperature | {report.temperature:g} |",
        f"| LLM judge | {judge} |",
        "",
        "## Overall",
        "",
        f"**Pass rate: {fraction(totals.passed, totals.cases)}**",
        "",
        f"- Passed: {totals.passed}",
        f"- Failed: {totals.failed}",
        f"- Errors (no verdict): {totals.errors}",
        f"- Replies wrapped in a code fence: {totals.fenced_outputs} of {totals.outputs}"
        " (stripped before scoring; a prompt-quality signal, not a failure)",
        "",
        "A case passes only if every required scorer that applies to it passes.",
        "",
        "## Pass rate by scorer",
        "",
        "| Scorer | Required | Passed | Applies to | Pass rate |",
        "|---|---|---|---|---|",
    ]
    for summary in report.by_scorer:
        lines.append(
            f"| `{summary.scorer}` | {'yes' if summary.required else 'no'} | "
            f"{summary.passed} | {summary.applicable} | {percent(summary.passed, summary.applicable)} |"
        )

    if report.confusion is not None:
        confusion = report.confusion
        lines += [
            "",
            f"## Confusion: {confusion.field}",
            "",
            f"Rows are the expected {confusion.field}, columns are what the model returned. "
            f"`{INVALID}` collects replies with a missing, unparseable, or unknown value. "
            "Cases with no model output are left out.",
            "",
            "| expected \\ actual | " + " | ".join(confusion.columns) + " |",
            "|---|" + "---|" * len(confusion.columns),
        ]
        for label, row in zip(confusion.labels, confusion.matrix):
            lines.append(f"| **{label}** | " + " | ".join(str(n) for n in row) + " |")

    failures = [r for r in report.cases if r.status == "failed"]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    for result in failures:
        lines += [f"### `{result.id}`{_flags(result)}", ""]
        if result.expected:
            lines += ["| Field | Expected | Actual |", "|---|---|---|"]
            for field, want in result.expected.items():
                if result.actual is None:
                    got = "(not parseable)"
                elif field not in result.actual:
                    got = "(missing)"
                else:
                    got = code_span(result.actual[field])
                lines.append(f"| `{field}` | {code_span(want)} | {got} |")
            lines.append("")
        for score in result.scores:
            if not score.passed:
                kind = "required" if score.required else "advisory"
                lines.append(
                    f"- `{score.scorer}` ({kind}): {code_span(score.detail, limit=300, in_table=False)}"
                )
        lines += ["", "Raw output:", "", *_block(result.raw_output or ""), ""]

    errored = [r for r in report.cases if r.status == "error"]
    if errored:
        lines += [
            "## Errors",
            "",
            "These cases have no verdict. Either the request for the model's reply failed, or "
            "a judge the case requires could not be reached or returned no usable grade.",
            "",
        ]
        lines += [f"- `{r.id}`{_flags(r)}: {r.error}" for r in errored]
        lines.append("")

    advisory = [
        (r, s) for r in report.cases if r.status == "passed" for s in r.scores if not s.passed
    ]
    if advisory:
        lines += ["## Advisory misses", "", "Cases that passed while a non-required scorer did not.", ""]
        lines += [
            f"- `{r.id}` `{s.scorer}`: {code_span(s.detail, limit=300, in_table=False)}"
            for r, s in advisory
        ]
        lines.append("")

    return encodable("\n".join(lines).rstrip("\n") + "\n")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def prepare_out_dir(out_dir: Union[str, Path]) -> None:
    """Create out_dir and prove a file can be written in it; OSError if not.

    Called before a run starts. Finding out that the report has nowhere to
    go after every request has been made would throw the outputs away.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=directory):
        pass


def write_report(out_dir: Union[str, Path], report: Report) -> tuple[Path, Path]:
    """Write report.json and report.md into out_dir, creating it if needed.

    The two files are a pair. Both are rendered, then both are written to
    scratch files beside their targets, and only then is each moved into
    place. A failure before the moves leaves the previous report, if there
    was one, exactly as it was, not half of a new one.
    """
    directory = Path(out_dir)
    json_path = directory / REPORT_JSON
    md_path = directory / REPORT_MD
    _write_together([(json_path, render_json(report)), (md_path, render_markdown(report))])
    return json_path, md_path


def write_outputs(out_dir: Union[str, Path], saved: SavedOutputs) -> Path:
    """Write a run's raw outputs to outputs.json in out_dir."""
    path = Path(out_dir) / OUTPUTS_JSON
    text = json.dumps(saved.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    _write_together([(path, encodable(text))])
    return path


def _write_together(rendered: list[tuple[Path, str]]) -> None:
    """Write every file or none: stage each beside its target, then move."""
    for path, _ in rendered:
        path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in rendered:
            scratch = path.with_name(f".{path.name}.tmp")
            staged.append((scratch, path))
            # newline="" keeps "\n" as written on every platform.
            scratch.write_text(text, encoding="utf-8", newline="")
        for scratch, path in staged:
            os.replace(scratch, path)
    except BaseException:
        for scratch, _ in staged:
            scratch.unlink(missing_ok=True)
        raise


def _read_json(source: Path) -> Any:
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"report '{source}' does not exist") from None
    except OSError as exc:
        raise ConfigError(f"cannot read report '{source}': {exc.strerror}") from None
    except ValueError:
        raise ConfigError(f"report '{source}' is not valid JSON") from None
    except RecursionError:
        raise ConfigError(f"report '{source}' is nested too deeply to be a report") from None


def _problems(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
        for err in exc.errors()[:3]
    )


def _as_report(raw: Any, source: Path) -> Report:
    try:
        return Report.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"'{source}' is not a report this version can read: {_problems(exc)}"
        ) from None


def load_report(path: Union[str, Path]) -> Report:
    """Read a report.json, or the report.json inside a run directory."""
    source = Path(path)
    if source.is_dir():
        source = source / REPORT_JSON
    return _as_report(_read_json(source), source)


def load_saved(path: Union[str, Path]) -> tuple[SavedOutputs, Path]:
    """Read the raw outputs of a saved run, and say which file they came from.

    The source is a report.json, an outputs.json left by a run that did not
    get as far as its report, or a run directory holding either. In a
    directory the report wins: it is the finished run.
    """
    source = Path(path)
    if source.is_dir():
        report, outputs = source / REPORT_JSON, source / OUTPUTS_JSON
        source = outputs if outputs.is_file() and not report.exists() else report
    raw = _read_json(source)
    if isinstance(raw, dict) and "outputs" in raw and "cases" not in raw:
        try:
            return SavedOutputs.model_validate(raw), source
        except ValidationError as exc:
            raise ConfigError(
                f"'{source}' is not a set of saved outputs this version can read: {_problems(exc)}"
            ) from None
    report = _as_report(raw, source)
    saved = SavedOutputs(
        schema_version=report.schema_version,
        task=report.task,
        model=report.model,
        endpoint_host=report.endpoint_host,
        timestamp=report.timestamp,
        temperature=report.temperature,
        judge_enabled=report.judge_enabled,
        judge_model=report.judge_model,
        outputs=outputs_from_report(report),
    )
    return saved, source


def outputs_from_report(report: Report) -> list[RawOutput]:
    """Recover the raw outputs a report was scored from."""
    return [
        RawOutput(
            case_id=case.id,
            text=case.raw_output,
            # A scored case has no request error; carry one only for cases
            # that never got a reply.
            error=case.error if case.raw_output is None else None,
            judge_text=case.judge_raw_output,
            judge_error=case.judge_error,
        )
        for case in report.cases
    ]
