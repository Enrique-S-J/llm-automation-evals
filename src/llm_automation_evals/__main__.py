"""Command line: python -m llm_automation_evals {run,report,compare}

  run      call the model for every case, score, write report.md + report.json
  report   re-score a saved run offline (after changing a scorer or a label)
  compare  diff two report.json files, regressions first

Exit codes:
  0  completed
  1  a gate failed: `run`/`report` below --fail-under, or `compare` found regressions
  2  usage or configuration error; nothing was run
  3  the run is incomplete: at least one case got no model output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .compare import compare_reports, render_comparison
from .models import ConfigError, Report
from .report import (
    build_report,
    fraction,
    load_report,
    outputs_from_report,
    utc_timestamp,
    write_report,
)
from .runner import Endpoint, load_suite, run_suite

EXIT_OK = 0
EXIT_GATE = 1
EXIT_CONFIG = 2
EXIT_INCOMPLETE = 3

DEFAULT_OUT_ROOT = "runs"


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _percentage(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("must be a percentage from 0 to 100")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_automation_evals",
        description="Run an eval suite against an OpenAI-compatible endpoint and report on it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="call the model for every case and write a report",
        description="Call the model for every case, score the replies, and write "
        "report.md and report.json. The endpoint comes from EVAL_BASE_URL, EVAL_MODEL "
        "and, optionally, EVAL_API_KEY.",
    )
    run.add_argument("task_dir", help="directory holding task.yaml and cases.yaml")
    run.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT_ROOT}/<task name>)")
    run.add_argument(
        "--judge",
        action="store_true",
        help="also grade with the LLM judge: one extra request per case, noisier than "
        "the other scorers, advisory unless a case makes it required",
    )
    run.add_argument("--concurrency", type=_positive_int, help="parallel requests (default: from task.yaml)")
    run.add_argument("--timeout", type=_positive_float, help="per-request timeout in seconds (default: from task.yaml)")
    run.add_argument("--fail-under", type=_percentage, metavar="PCT", help="exit 1 if the pass rate is below PCT")

    report = commands.add_parser(
        "report",
        help="re-score a saved run without calling the model",
        description="Re-score the raw outputs saved in an earlier report.json against the "
        "task directory as it is now, and write the report again. Makes no requests.",
    )
    report.add_argument("task_dir", help="directory holding task.yaml and cases.yaml")
    report.add_argument(
        "source",
        nargs="?",
        help=f"report.json, or the run directory holding it (default: {DEFAULT_OUT_ROOT}/<task name>)",
    )
    report.add_argument("--out", help="output directory (default: where the source report is)")
    report.add_argument("--fail-under", type=_percentage, metavar="PCT", help="exit 1 if the pass rate is below PCT")

    compare = commands.add_parser(
        "compare",
        help="diff two report.json files, regressions first",
        description="Compare a baseline report with a candidate. Prints markdown and "
        "exits 1 if any case that passed in the baseline fails in the candidate.",
    )
    compare.add_argument("baseline", help="report.json (or run directory) to compare against")
    compare.add_argument("candidate", help="report.json (or run directory) under review")
    compare.add_argument("--out", help="also write the comparison to this file")
    compare.add_argument("--exit-zero", action="store_true", help="exit 0 even when there are regressions")
    return parser


def _finish(report: Report, out_dir: Path, fail_under: Optional[float]) -> int:
    try:
        json_path, md_path = write_report(out_dir, report)
    except OSError as exc:
        raise ConfigError(f"cannot write the report to '{out_dir}': {exc.strerror}") from None
    totals = report.totals
    print(
        f"{report.task}: {fraction(totals.passed, totals.cases)} passed; "
        f"{totals.failed} failed, {totals.errors} without model output"
    )
    print(f"model: {report.model}  endpoint host: {report.endpoint_host}")
    print(f"report: {md_path}")
    print(f"data:   {json_path}")
    if totals.errors:
        _err(
            f"Incomplete: {totals.errors} of {totals.cases} cases got no model output. "
            "The pass rate above counts them as not passed; see the Errors section."
        )
        return EXIT_INCOMPLETE
    if fail_under is not None and (totals.pass_rate or 0.0) * 100 < fail_under:
        _err(f"Pass rate is below --fail-under {fail_under:g}.")
        return EXIT_GATE
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.task_dir)
    endpoint = Endpoint.from_env(suite.task.endpoint)
    out_dir = Path(args.out) if args.out else Path(DEFAULT_OUT_ROOT) / suite.task.name
    timestamp = utc_timestamp()
    outputs = run_suite(
        suite,
        endpoint,
        judge=args.judge,
        concurrency=args.concurrency,
        timeout=args.timeout,
        log=_err,
    )
    report = build_report(
        suite,
        outputs,
        model=endpoint.model,
        endpoint_host=endpoint.host,
        timestamp=timestamp,
        judge=args.judge,
        judge_model=endpoint.judge_model or endpoint.model,
    )
    return _finish(report, out_dir, args.fail_under)


def _cmd_report(args: argparse.Namespace) -> int:
    suite = load_suite(args.task_dir)
    source = Path(args.source) if args.source else Path(DEFAULT_OUT_ROOT) / suite.task.name
    if not args.source and not source.exists():
        raise ConfigError(
            f"no saved run at '{source}'. Run the suite first, or pass the path to a report.json."
        )
    previous = load_report(source)
    if previous.task != suite.task.name:
        raise ConfigError(
            f"that report is for task '{previous.task}', but '{args.task_dir}' is '{suite.task.name}'"
        )
    known = {case.id for case in suite.cases}
    dropped = [case.id for case in previous.cases if case.id not in known]
    if dropped:
        _err(f"Ignoring saved outputs for cases no longer in the suite: {', '.join(dropped)}")
    # Model, host and timestamp describe when the outputs were produced, so
    # they are carried over. Re-scoring unchanged files reproduces the report
    # byte for byte.
    report = build_report(
        suite,
        outputs_from_report(previous),
        model=previous.model,
        endpoint_host=previous.endpoint_host,
        timestamp=previous.timestamp,
        judge=previous.judge_enabled,
        judge_model=previous.judge_model,
    )
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = source if source.is_dir() else source.parent
    return _finish(report, out_dir, args.fail_under)


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_reports(load_report(args.baseline), load_report(args.candidate))
    text = render_comparison(comparison)
    sys.stdout.write(text)
    if args.out:
        target = Path(args.out)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        except OSError as exc:
            raise ConfigError(f"cannot write '{target}': {exc.strerror}") from None
    if comparison.regressions and not args.exit_zero:
        return EXIT_GATE
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"run": _cmd_run, "report": _cmd_report, "compare": _cmd_compare}[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        _err(f"Error: {exc}")
        return EXIT_CONFIG
    except KeyboardInterrupt:
        _err("Interrupted; no report written.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
