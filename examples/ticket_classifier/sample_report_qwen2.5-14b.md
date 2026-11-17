# Eval report: ticket_classifier

| | |
|---|---|
| Model | `qwen2.5:14b` |
| Endpoint host | `localhost` |
| Timestamp (UTC) | 2026-11-17T17:31:57Z |
| Cases | 30 |
| Temperature | 0 |
| LLM judge | off |

## Overall

**Pass rate: 21/30 (70.0%)**

- Passed: 21
- Failed: 9
- Errors (no verdict): 0
- Replies wrapped in a code fence: 0 of 30 (stripped before scoring; a prompt-quality signal, not a failure)

A case passes only if every required scorer that applies to it passes.

## Pass rate by scorer

| Scorer | Required | Passed | Applies to | Pass rate |
|---|---|---|---|---|
| `json_schema` | yes | 30 | 30 | 100.0% |
| `category_match` | yes | 20 | 24 | 83.3% |
| `priority_match` | yes | 19 | 25 | 76.0% |
| `needs_human_match` | yes | 19 | 19 | 100.0% |
| `keyword` | yes | 10 | 10 | 100.0% |
| `safe_failure` | yes | 5 | 6 | 83.3% |

## Confusion: category

Rows are the expected category, columns are what the model returned. `(invalid)` collects replies with a missing, unparseable, or unknown value. Cases with no model output are left out.

| expected \ actual | voice | network | infrastructure | other | (invalid) |
|---|---|---|---|---|---|
| **voice** | 6 | 0 | 0 | 0 | 0 |
| **network** | 2 | 5 | 0 | 0 | 0 |
| **infrastructure** | 1 | 1 | 5 | 0 | 0 |
| **other** | 1 | 0 | 0 | 9 | 0 |

## Failures

### `voice-01`

| Field | Expected | Actual |
|---|---|---|
| `category` | `voice` | `voice` |
| `priority` | `normal` | `high` |
| `needs_human` | `false` | `false` |

- `priority_match` (required): `priority: expected "normal", got "high"`

Raw output:

```text
{
  "category": "voice",
  "priority": "high",
  "summary": "Front desk phone at extension 104 has no service and no dial tone.",
  "needs_human": false
}
```

### `voice-06` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `voice` | `voice` |
| `priority` | `normal` | `high` |

- `priority_match` (required): `priority: expected "normal", got "high"`

Raw output:

```text
{
  "category": "voice",
  "priority": "high",
  "summary": "Softphone authentication failed after password change",
  "needs_human": false
}
```

### `net-04`

| Field | Expected | Actual |
|---|---|---|
| `category` | `network` | `network` |
| `priority` | `low` | `normal` |
| `needs_human` | `false` | `false` |

- `priority_match` (required): `priority: expected "low", got "normal"`

Raw output:

```text
{"category": "network", "priority": "normal", "summary": "Request for firewall rule for new label printing portal", "needs_human": false}
```

### `net-06` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `network` | `voice` |
| `priority` | `normal` | `high` |

- `category_match` (required): `category: expected "network", got "voice"`
- `priority_match` (required): `priority: expected "normal", got "high"`

Raw output:

```text
{"category": "voice", "priority": "high", "summary": "Softphone calls cut out after thirty seconds when using VPN.", "needs_human": false}
```

### `net-07` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `network` | `voice` |
| `priority` | `high` | `high` |

- `category_match` (required): `category: expected "network", got "voice"`

Raw output:

```text
{
  "category": "voice",
  "priority": "high",
  "summary": "East wing desk phones and network connections down after power blip",
  "needs_human": false
}
```

### `infra-03`

| Field | Expected | Actual |
|---|---|---|
| `category` | `infrastructure` | `infrastructure` |
| `priority` | `low` | `normal` |
| `needs_human` | `false` | `false` |

- `priority_match` (required): `priority: expected "low", got "normal"`

Raw output:

```text
{"category": "infrastructure", "priority": "normal", "summary": "Request for a test VM with specific specs.", "needs_human": false}
```

### `infra-06` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `infrastructure` | `network` |
| `priority` | `urgent` | `high` |

- `category_match` (required): `category: expected "infrastructure", got "network"`
- `priority_match` (required): `priority: expected "urgent", got "high"`

Raw output:

```text
{"category": "network", "priority": "high", "summary": "Nobody can access the shared drive; network path not found.", "needs_human": false}
```

### `infra-07` (ambiguous)

| Field | Expected | Actual |
|---|---|---|
| `category` | `infrastructure` | `voice` |

- `category_match` (required): `category: expected "infrastructure", got "voice"`

Raw output:

```text
{"category": "voice", "priority": "high", "summary": "Call recordings not saved since Friday due to full storage volume.", "needs_human": false}
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
{
  "category": "voice",
  "priority": "normal",
  "summary": "Desk phone display is dim and hard to read.",
  "needs_human": false
}
```
