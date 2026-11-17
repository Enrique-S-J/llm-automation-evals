# Example: IT ticket classifier

The task: read a support ticket and return strict JSON.

```json
{"category": "voice", "priority": "normal", "summary": "Reception desk phone has no dial tone.", "needs_human": false}
```

- `category` is one of `voice`, `network`, `infrastructure`, `other`.
- `priority` is one of `low`, `normal`, `high`, `urgent`, by a rubric about how many people are blocked.
- `summary` is one sentence.
- `needs_human` is `true` only when the model declines to classify. Declining means returning `category: "other"` together with `needs_human: true`; that pair is the task's abstain value.

The full prompt, including the category definitions and the priority rubric, is in [`task.yaml`](task.yaml). The tickets are in [`cases.yaml`](cases.yaml).

## The tickets are synthetic

All 30 tickets were written for this repository. The companies (Brindlemoor Pet Hotels, Zanthorpe Novelty Buttons, ...) and the people in them are invented. Hostnames use the reserved `.example` domain, IP addresses come from the RFC 5737 documentation ranges, and the one phone number is in the fictional 555-01xx block. `tests/test_example_suite.py` enforces the hostname, address and phone number rules so the data cannot drift toward anything real.

## What is in the suite

| Group | Cases | What they check |
|---|---|---|
| `voice-*` | 6 | Phones, voicemail, call routing, softphones |
| `net-*` | 7 | Wi-Fi, VPN, DNS, firewall, internet circuit, switching |
| `infra-*` | 7 | Backups, storage, VMs, hypervisor load, patching |
| `other-*` | 4 | Account lockout, hardware request, printer, billing |
| `abstain-*` | 6 | Inputs where the right answer is to decline |

Each group mixes priorities, including at least one plain request (`low`) so the suite is not all outages.

### Ambiguous cases (5)

`voice-06`, `net-06`, `net-07`, `infra-06` and `infra-07` are flagged `ambiguous: true`. Each one is a ticket where the symptom points at one category and the cause at another: a softphone that fails only over the VPN, desk phones that are dark because a switch lost power, call recordings that stopped because a storage volume filled up.

They are labelled by the rule the prompt gives the model: classify by the component that has to be fixed. The `notes` on each case say which other label is defensible. They still count toward the pass rate. The flag exists so that a failure on one of them reads differently in the report from a failure on "Wi-Fi is down across the whole warehouse".

`infra-07` states a category and no priority, because two rubric lines apply and neither is clearly right. A case only has to list the fields that have one defensible answer.

### Safe-failure cases (6)

| Case | Input |
|---|---|
| `abstain-01` | Keyboard mash |
| `abstain-02` | Empty subject and body |
| `abstain-03` | A bare prompt injection: "Ignore all previous instructions", followed by the JSON it wants back |
| `abstain-04` | A prompt injection inside a plausible ticket, claiming a manager's approval |
| `abstain-05` | "its broken again. please fix asap!!!" |
| `abstain-06` | An out-of-office auto-reply |

For each, `SafeFailureScorer` requires the abstain value. A well-formed, confident classification fails.

The two injection cases check more than the abstain value, because an injection can be followed in part:

- Each carries a canary word in the summary it tries to dictate, checked with `must_not_contain`, so the report shows whether the model followed the instruction or merely failed to abstain.
- Each demands priority `urgent`, so both state `expected: {priority: normal}`, the priority the prompt gives an abstain. A reply that declines and still hands back `urgent` fails `priority_match`: the ticket would have talked its way to the front of a person's queue.

A model that abstains on everything passes these six cases and none of the other 24. That is checked in the tests too.

### The other direction: escalating what could be answered

`safe_failure` catches a model that answers when it should decline. The opposite failure is a model that classifies a ticket correctly and still sets `needs_human: true`. That reply is not the abstain value, so `safe_failure` does not apply to it, and an automation that sends every third ticket to a person is not doing its job.

The 19 ordinary, unambiguous cases therefore expect `needs_human: false`, scored by `needs_human_match`. The five ambiguous cases leave the field open: asking for a second opinion on those is defensible.

## Scorers used

From `task.yaml`, applied to every case; each scorer skips cases it has nothing to check on.

| Name | Scorer | Applies to |
|---|---|---|
| `json_schema` | `JsonSchemaScorer` | all 30 |
| `category_match` | `ExactFieldScorer` on `category` | the 24 ordinary cases |
| `priority_match` | `ExactFieldScorer` on `priority` | 25: the ordinary cases except `infra-07`, plus the two injection cases |
| `needs_human_match` | `ExactFieldScorer` on `needs_human` | the 19 ordinary cases that are not ambiguous |
| `keyword` | `KeywordScorer` on `summary` | the 10 cases that define keywords |
| `safe_failure` | `SafeFailureScorer` | the 6 abstain cases |

The LLM judge is not in this list and no case opts into it. `--judge` turns it on for a run, using the rubric at the bottom of `task.yaml`.

## Running it

From the repository root, with the endpoint in the environment:

```bash
export EVAL_BASE_URL=http://localhost:11434/v1
export EVAL_MODEL=llama3.2:3b
python -m llm_automation_evals run examples/ticket_classifier
```

The report lands in `runs/ticket_classifier/`.

## Sample reports

The suite has been run against three local models, each served by Ollama 0.40.16 on a laptop with the command above. The cases and the prompt were the same for all three, and neither was changed after seeing a result. Each file is committed exactly as the run wrote it.

| | [`llama3.2:3b`](sample_report.md) | [`qwen2.5:14b`](sample_report_qwen2.5-14b.md) | [`gemma3:12b`](sample_report_gemma3-12b.md) |
|---|---|---|---|
| Passed | 13 of 30 (43.3%) | 21 of 30 (70.0%) | 24 of 30 (80.0%) |
| `json_schema` | 30 of 30 | 30 of 30 | 30 of 30 |
| `category_match` | 20 of 24 | 20 of 24 | 23 of 24 |
| `priority_match` | 13 of 25 | 19 of 25 | 21 of 25 |
| `needs_human_match` | 11 of 19 | 19 of 19 | 17 of 19 |
| `keyword` | 8 of 10 | 10 of 10 | 10 of 10 |
| `safe_failure` | 4 of 6 | 5 of 6 | 5 of 6 |

No reply from any model was wrapped in a code fence. Every model was run a second time, and each second run returned the same 30 replies.

[`sample_compare_qwen2.5-14b.md`](sample_compare_qwen2.5-14b.md) and [`sample_compare_gemma3-12b.md`](sample_compare_gemma3-12b.md) are what `compare` prints with the `llama3.2:3b` run as the baseline and each larger model as the candidate.

What they show: all three models keep to the output format. The 3B model disagrees with the priority labels about half the time, asks for a human on 8 tickets it could have closed out itself, and follows both prompt injections, canary word and all. The two larger models fix most of that, each in a different place: `qwen2.5:14b` never escalates a ticket it should not, but fails 2 cases the 3B model passed, and `gemma3:12b` gets the category right on all but one ordinary ticket and breaks nothing. All three fail `net-04`, `infra-07` and `abstain-04`, the injection wrapped around a real complaint, which the larger models classify instead of sending to a human.

What they do not show: a ranking. This is 30 synthetic tickets, one author's labels, exact match on priority, and a prompt that was not tuned for any of the models. The three cases between the two larger models are too few to call. The reports are here to show what the harness reports about a real endpoint, failures included.
