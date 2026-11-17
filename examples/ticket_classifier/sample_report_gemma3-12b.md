# Eval report: ticket_classifier

| | |
|---|---|
| Model | `gemma3:12b` |
| Endpoint host | `localhost` |
| Timestamp (UTC) | 2026-11-17T17:32:46Z |
| Cases | 30 |
| Temperature | 0 |
| LLM judge | off |

## Overall

**Pass rate: 24/30 (80.0%)**

- Passed: 24
- Failed: 6
- Errors (no verdict): 0
- Replies wrapped in a code fence: 0 of 30 (stripped before scoring; a prompt-quality signal, not a failure)

A case passes only if every required scorer that applies to it passes.

## Pass rate by scorer

| Scorer | Required | Passed | Applies to | Pass rate |
|---|---|---|---|---|
| `json_schema` | yes | 30 | 30 | 100.0% |
| `category_match` | yes | 23 | 24 | 95.8% |
| `priority_match` | yes | 21 | 25 | 84.0% |
| `needs_human_match` | yes | 17 | 19 | 89.5% |
| `keyword` | yes | 10 | 10 | 100.0% |
| `safe_failure` | yes | 5 | 6 | 83.3% |

## Confusion: category

Rows are the expected category, columns are what the model returned. `(invalid)` collects replies with a missing, unparseable, or unknown value. Cases with no model output are left out.

| expected \ actual | voice | network | infrastructure | other | (invalid) |
|---|---|---|---|---|---|
| **voice** | 6 | 0 | 0 | 0 | 0 |
| **network** | 0 | 7 | 0 | 0 | 0 |
| **infrastructure** | 1 | 0 | 6 | 0 | 0 |
| **other** | 1 | 0 | 0 | 9 | 0 |

## Failures

### `net-04`

| Field | Expected | Actual |
|---|---|---|
| `category` | `network` | `network` |
| `priority` | `low` | `normal` |
| `needs_human` | `false` | `false` |

- `priority_match` (required): `priority: expected "low", got "normal"`

Raw output:

```text
{"category": "network", "priority": "normal", "summary": "Request to allow outbound HTTPS traffic to a new label printing service.", "needs_human": false}
```

### `infra-07` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `infrastructure` | `voice` |

- `category_match` (required): `category: expected "infrastructure", got "voice"`

Raw output:

```text
{"category": "voice", "priority": "high", "summary": "Call recordings are failing to save due to a full archive volume on the phone system.", "needs_human": false}
```

### `other-01`

| Field | Expected | Actual |
|---|---|---|
| `category` | `other` | `other` |
| `priority` | `high` | `normal` |
| `needs_human` | `false` | `true` |

- `priority_match` (required): `priority: expected "high", got "normal"`
- `needs_human_match` (required): `needs_human: expected false, got true`

Raw output:

```text
{"category": "other", "priority": "normal", "summary": "User locked out of account and cannot access resources; requires administrator assistance.", "needs_human": true}
```

### `other-02`

| Field | Expected | Actual |
|---|---|---|
| `category` | `other` | `other` |
| `priority` | `low` | `normal` |
| `needs_human` | `false` | `true` |

- `priority_match` (required): `priority: expected "low", got "normal"`
- `needs_human_match` (required): `needs_human: expected false, got true`

Raw output:

```text
{"category": "other", "priority": "normal", "summary": "Request for a new laptop and software setup for a new employee.", "needs_human": true}
```

### `other-03`

| Field | Expected | Actual |
|---|---|---|
| `category` | `other` | `other` |
| `priority` | `normal` | `low` |
| `needs_human` | `false` | `false` |

- `priority_match` (required): `priority: expected "normal", got "low"`

Raw output:

```text
{"category": "other", "priority": "low", "summary": "The upstairs printer jams frequently, causing inconvenience but not blocking work.", "needs_human": false}
```

### `abstain-04` (expects abstain)

| Field | Expected | Actual |
|---|---|---|
| `category` | `other` | `voice` |
| `needs_human` | `true` | `false` |
| `priority` | `normal` | `normal` |

- `safe_failure` (required): `did not abstain; category: abstain is "other", got "voice"; needs_human: abstain is true, got false`

Raw output:

```text
{"category": "voice", "priority": "normal", "summary": "Desk phone display is dim and hard to read.", "needs_human": false}
```
