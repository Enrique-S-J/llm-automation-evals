"""The shipped ticket_classifier suite is itself under test.

A golden suite is only worth something if its own labels and flags are
right, so these check the example for the properties the README claims.
"""

import ipaddress
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from llm_automation_evals.models import RawOutput
from llm_automation_evals.report import build_report
from llm_automation_evals.runner import load_suite, render_prompt
from llm_automation_evals.scorers import LlmJudgeScorer, scorers_for

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ticket_classifier"
META = {"model": "oracle", "endpoint_host": "none", "timestamp": "2026-01-01T00:00:00Z"}

DOCUMENTATION_NETWORKS = [
    ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
]


@pytest.fixture(scope="module")
def suite():
    return load_suite(EXAMPLE)


def oracle_reply(suite, case, **overrides):
    """What a model that agrees with every label would say."""
    body = {"category": "other", "priority": "normal", "summary": "", "needs_human": False}
    if case.expect_abstain:
        body.update(suite.task.abstain, summary="Cannot classify this ticket.")
    else:
        body.update(case.expected)
        terms = case.keywords.must_contain if case.keywords else []
        words = [term if isinstance(term, str) else term[0] for term in terms]
        body["summary"] = " ".join(["Ticket about", *words]) + "."
    body.update(overrides)
    return json.dumps(body)


def test_thirty_cases_across_all_four_categories(suite):
    assert len(suite.cases) == 30
    by_category = Counter(case.expected.get("category") for case in suite.cases if not case.expect_abstain)
    assert by_category == {"voice": 6, "network": 7, "infrastructure": 7, "other": 4}
    priorities = {case.expected.get("priority") for case in suite.cases} - {None}
    assert priorities == {"low", "normal", "high", "urgent"}


def test_a_handful_of_cases_are_flagged_ambiguous_and_say_why(suite):
    ambiguous = [case for case in suite.cases if case.ambiguous]
    assert len(ambiguous) == 5
    assert all(len(case.notes) > 40 for case in ambiguous)
    assert not any(case.expect_abstain for case in ambiguous)


def test_safe_failure_cases_cover_garbage_empty_and_injection(suite):
    abstain = {case.id: case for case in suite.cases if case.expect_abstain}
    assert len(abstain) == 6
    assert any(case.input == {"subject": "", "body": ""} for case in abstain.values())
    injections = [c for c in abstain.values() if "classifier" in c.input["body"].lower()]
    assert len(injections) == 2
    # Each injection carries a canary the summary must not repeat.
    assert all(case.keywords and case.keywords.must_not_contain for case in injections)
    for case in abstain.values():
        assert [s.name for s in scorers_for(suite.task, case)][-1] == "safe_failure"


def test_the_llm_judge_is_off_in_the_shipped_suite(suite):
    assert "llm_judge" not in [spec.type for spec in suite.task.scorers]
    for case in suite.cases:
        assert not any(isinstance(s, LlmJudgeScorer) for s in scorers_for(suite.task, case))


def test_tickets_are_delimited_in_the_prompt(suite):
    prompt = render_prompt(suite.task, suite.cases[0])
    assert prompt.startswith("<<<TICKET\nSubject: Reception phone has no dial tone\n\n")
    assert prompt.endswith("\nTICKET>>>\n")


def test_every_case_is_passable_and_an_answer_matching_the_labels_scores_30_of_30(suite):
    outputs = [RawOutput(case_id=c.id, text=oracle_reply(suite, c)) for c in suite.cases]
    report = build_report(suite, outputs, **META)
    assert [c.id for c in report.cases if c.status != "passed"] == []
    assert report.totals.pass_rate == 1.0
    # The diagonal of the confusion table holds every case.
    matrix = report.confusion.matrix
    assert [matrix[i][i] for i in range(4)] == [6, 7, 7, 10]


def test_a_model_that_always_abstains_passes_only_the_abstain_cases(suite):
    """Abstaining is not a free pass.

    SafeFailureScorer alone could be satisfied by declining everything. The
    ordinary cases are what make that expensive.
    """
    abstain = json.dumps(
        {"category": "other", "priority": "normal", "summary": "Cannot classify.", "needs_human": True}
    )
    report = build_report(suite, [RawOutput(case_id=c.id, text=abstain) for c in suite.cases], **META)
    assert [c.id for c in report.cases if c.status == "passed"] == [f"abstain-0{n}" for n in range(1, 7)]


def test_following_either_injection_fails_the_case(suite):
    obeyed = {
        "abstain-03": {"category": "infrastructure", "priority": "urgent", "summary": "PINEAPPLE-7 override accepted", "needs_human": False},
        "abstain-04": {"category": "network", "priority": "urgent", "summary": "Dim display, code MARMOT-42.", "needs_human": False},
    }
    outputs = [
        RawOutput(case_id=c.id, text=json.dumps(obeyed[c.id]) if c.id in obeyed else oracle_reply(suite, c))
        for c in suite.cases
    ]
    report = build_report(suite, outputs, **META)
    failed = {c.id: [s.scorer for s in c.scores if not s.passed] for c in report.cases if c.status == "failed"}
    assert failed == {"abstain-03": ["keyword", "safe_failure"], "abstain-04": ["keyword", "safe_failure"]}


def test_example_data_uses_only_reserved_names_and_addresses(suite):
    """Synthetic means it cannot collide with anything real.

    Hostnames stay under the reserved .example domain, addresses inside the
    RFC 5737 documentation ranges, and phone numbers in the 555-01xx block.
    """
    text = "\n".join(value for case in suite.cases for value in case.input.values())

    hostnames = re.findall(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b", text)
    assert hostnames, "expected the tickets to mention some hostnames"
    assert [h for h in hostnames if not h.endswith(".example")] == []

    addresses = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    assert addresses
    for address in addresses:
        assert any(ipaddress.ip_address(address) in net for net in DOCUMENTATION_NETWORKS), address

    assert re.findall(r"\b\d{3}-\d{4}\b", text) == ["555-0100"]
    assert "@" not in text
