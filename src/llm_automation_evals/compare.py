"""Compare: diff two report.json files, baseline against candidate.

The output leads with regressions, because that is the question a reviewer
has about a prompt or model change: what used to work that no longer does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .models import CaseResult, Report
from .report import code_span, encodable, fraction


@dataclass(frozen=True)
class Flip:
    """A case whose pass/fail status differs between the two reports."""

    case_id: str
    # Required scorers that failed on the side where the case failed.
    failing: list[str]
    expected: dict[str, Any]
    baseline_actual: Optional[dict[str, Any]]
    candidate_actual: Optional[dict[str, Any]]
    ambiguous: bool = False
    # False when the model said exactly the same thing in both reports, which
    # means the flip did not come from the model.
    output_changed: bool = True
    # True when the judge's reply differs. With an unchanged output, that
    # makes the flip the judge's doing; otherwise the suite changed.
    judge_changed: bool = False


@dataclass(frozen=True)
class Delta:
    name: str
    baseline: Optional[float]
    candidate: Optional[float]

    @property
    def change(self) -> Optional[float]:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclass
class Comparison:
    baseline: Report
    candidate: Report
    regressions: list[Flip] = field(default_factory=list)
    fixes: list[Flip] = field(default_factory=list)
    still_failing: list[str] = field(default_factory=list)
    # Cases where a request failed on one side, so there is no verdict to
    # compare: (case id, baseline status, candidate status).
    not_comparable: list[tuple[str, str, str]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    deltas: list[Delta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _failing(case: CaseResult) -> list[str]:
    return [s.scorer for s in case.scores if s.required and not s.passed]


def compare_reports(baseline: Report, candidate: Report) -> Comparison:
    """Classify every case as a regression, a fix, or neither.

    Order follows the candidate report, so the output is deterministic. A
    case that errored on either side is never called a regression or a fix:
    an endpoint outage is not a change in model behaviour.
    """
    result = Comparison(baseline=baseline, candidate=candidate)
    if baseline.task != candidate.task:
        result.warnings.append(
            f"The reports are for different tasks ({baseline.task} and {candidate.task})."
        )

    before = {case.id: case for case in baseline.cases}
    after = {case.id: case for case in candidate.cases}
    result.added = [case.id for case in candidate.cases if case.id not in before]
    result.removed = [case.id for case in baseline.cases if case.id not in after]
    if result.added or result.removed:
        result.warnings.append(
            "The case sets differ, so the pass rates are not over the same cases."
        )

    for case in candidate.cases:
        old = before.get(case.id)
        if old is None:
            continue
        pair = (old.status, case.status)
        if "error" in pair:
            result.not_comparable.append((case.id, old.status, case.status))
            continue
        changed = old.raw_output != case.raw_output
        judge_changed = old.judge_raw_output != case.judge_raw_output
        if pair == ("passed", "failed"):
            result.regressions.append(
                Flip(case.id, _failing(case), case.expected, old.actual, case.actual, case.ambiguous, changed, judge_changed)
            )
        elif pair == ("failed", "passed"):
            result.fixes.append(
                Flip(case.id, _failing(old), case.expected, old.actual, case.actual, case.ambiguous, changed, judge_changed)
            )
        elif pair == ("failed", "failed"):
            result.still_failing.append(case.id)

    result.deltas.append(Delta("overall", baseline.totals.pass_rate, candidate.totals.pass_rate))
    base_scorers = {s.scorer: s.pass_rate for s in baseline.by_scorer}
    cand_scorers = {s.scorer: s.pass_rate for s in candidate.by_scorer}
    for name in list(cand_scorers) + [n for n in base_scorers if n not in cand_scorers]:
        result.deltas.append(Delta(name, base_scorers.get(name), cand_scorers.get(name)))
    return result


def _percent(rate: Optional[float]) -> str:
    return "n/a" if rate is None else f"{100 * rate:.1f}%"


def _points(change: Optional[float]) -> str:
    if change is None:
        return "n/a"
    return f"{100 * change:+.1f} pts" if abs(change) >= 0.0005 else "0.0 pts"


def _count(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _shown(actual: Optional[dict[str, Any]], name: str) -> str:
    if actual is None:
        return "(not parseable)"
    return code_span(actual[name]) if name in actual else "(missing)"


def _describe(report: Report) -> str:
    return (
        f"{code_span(report.model)} at {code_span(report.endpoint_host)}, {report.timestamp}: "
        f"{fraction(report.totals.passed, report.totals.cases)}"
    )


def _flip_line(flip: Flip, failed_side: str) -> str:
    """One bullet: which scorers failed, and what each changed field went from and to."""
    tag = " (ambiguous case)" if flip.ambiguous else ""
    scorers = ", ".join(f"`{name}`" for name in flip.failing) or "(no required scorer recorded)"
    changes = []
    for name, want in flip.expected.items():
        was = flip.baseline_actual.get(name) if flip.baseline_actual is not None else None
        now = flip.candidate_actual.get(name) if flip.candidate_actual is not None else None
        if was != now:
            changes.append(
                f"`{name}` expected {code_span(want)}, "
                f"was {_shown(flip.baseline_actual, name)}, now {_shown(flip.candidate_actual, name)}"
            )
    detail = "; " + "; ".join(changes) if changes else ""
    if not flip.output_changed and flip.judge_changed:
        detail += (
            "; the model's output is identical in both reports and the judge's reply is not, "
            "so the judge changed, not the model"
        )
    elif not flip.output_changed:
        detail += "; the model's output is identical in both reports, so the suite changed, not the model"
    return f"- `{flip.case_id}`{tag}: {failed_side} {scorers}{detail}"


def render_comparison(comparison: Comparison) -> str:
    base, cand = comparison.baseline, comparison.candidate
    overall = comparison.deltas[0]
    lines = [
        f"# Eval comparison: {cand.task}",
        "",
        f"- Baseline: {_describe(base)}",
        f"- Candidate: {_describe(cand)}",
        "",
        f"**{_count(len(comparison.regressions), 'regression', 'regressions')}, "
        f"{_count(len(comparison.fixes), 'fix', 'fixes')}.** "
        f"Pass rate {_percent(overall.baseline)} -> {_percent(overall.candidate)} "
        f"({_points(overall.change)}).",
        "",
    ]
    for warning in comparison.warnings:
        lines.append(f"> Warning: {warning}")
    if comparison.warnings:
        lines.append("")

    lines += [f"## Regressions ({len(comparison.regressions)})", ""]
    if comparison.regressions:
        lines.append("Passed in the baseline, fail in the candidate.")
        lines.append("")
        lines += [_flip_line(flip, "now fails") for flip in comparison.regressions]
    else:
        lines.append("None.")

    lines += ["", f"## Fixes ({len(comparison.fixes)})", ""]
    if comparison.fixes:
        lines.append("Failed in the baseline, pass in the candidate.")
        lines.append("")
        lines += [_flip_line(flip, "used to fail") for flip in comparison.fixes]
    else:
        lines.append("None.")

    lines += [
        "",
        "## Score deltas",
        "",
        "| Scorer | Baseline | Candidate | Change |",
        "|---|---|---|---|",
    ]
    for delta in comparison.deltas:
        name = "**overall**" if delta.name == "overall" else f"`{delta.name}`"
        lines.append(
            f"| {name} | {_percent(delta.baseline)} | {_percent(delta.candidate)} | "
            f"{_points(delta.change)} |"
        )

    if comparison.still_failing:
        lines += ["", f"## Still failing ({len(comparison.still_failing)})", ""]
        lines.append(", ".join(f"`{case_id}`" for case_id in comparison.still_failing))

    if comparison.not_comparable:
        lines += [
            "",
            f"## Not comparable ({len(comparison.not_comparable)})",
            "",
            "A request failed on at least one side, so there is no verdict to compare. "
            "These are neither regressions nor fixes.",
            "",
        ]
        lines += [
            f"- `{case_id}`: {was} -> {now}" for case_id, was, now in comparison.not_comparable
        ]

    if comparison.added or comparison.removed:
        lines += ["", "## Case set changes", ""]
        if comparison.added:
            lines.append("- Added: " + ", ".join(f"`{c}`" for c in comparison.added))
        if comparison.removed:
            lines.append("- Removed: " + ", ".join(f"`{c}`" for c in comparison.removed))

    return encodable("\n".join(lines).rstrip("\n") + "\n")
