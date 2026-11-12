"""Builders shared by the test modules: a small task, cases, and canned replies."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llm_automation_evals.models import Case, RawOutput, Task

LABELS = ["voice", "network", "infrastructure", "other"]

TASK = {
    "name": "mini",
    "request": {"retry_delay_seconds": 0, "timeout_seconds": 5},
    "system_prompt": "Classify the ticket. Reply with JSON only.",
    "prompt_template": "Subject: {{subject}}\n\n{{body}}\n",
    "output_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "priority", "summary", "needs_human"],
        "properties": {
            "category": {"enum": LABELS},
            "priority": {"enum": ["low", "normal", "high", "urgent"]},
            "summary": {"type": "string", "minLength": 1},
            "needs_human": {"type": "boolean"},
        },
    },
    "abstain": {"category": "other", "needs_human": True},
    "confusion_field": "category",
    "scorers": [
        "json_schema",
        {"type": "exact_field", "name": "category_match", "fields": ["category"]},
        {"type": "exact_field", "name": "priority_match", "fields": ["priority"]},
        "keyword",
        "safe_failure",
    ],
    "judge": {"rubric": "5 is an accurate one-sentence summary. 1 is wrong."},
}


def make_task(**overrides: Any) -> Task:
    return Task.model_validate({**TASK, **overrides})


def make_case(case_id: str = "case-1", **fields: Any) -> Case:
    data = {
        "id": case_id,
        "input": {"subject": f"subject of {case_id}", "body": f"body of {case_id}"},
        "expected": {"category": "voice", "priority": "normal"},
    }
    data.update(fields)
    return Case.model_validate(data)


def abstain_case(case_id: str = "abstain-1", **fields: Any) -> Case:
    return make_case(case_id, expected={}, expect_abstain=True, **fields)


def reply(**fields: Any) -> str:
    """A well-formed model reply, with any field overridden."""
    body = {
        "category": "voice",
        "priority": "normal",
        "summary": "Desk phone has no dial tone.",
        "needs_human": False,
    }
    body.update(fields)
    return json.dumps(body)


ABSTAIN = reply(category="other", summary="Cannot classify this ticket.", needs_human=True)


def out(text: Any, case_id: str = "case-1", **fields: Any) -> RawOutput:
    return RawOutput(case_id=case_id, text=text, **fields)


def completion(content: Any) -> httpx.Response:
    """An OpenAI-style chat completion response carrying `content`."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )
