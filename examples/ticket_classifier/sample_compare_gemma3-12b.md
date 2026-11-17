# Eval comparison: ticket_classifier

- Baseline: `llama3.2:3b` at `localhost`, 2026-11-17T17:33:43Z: 13/30 (43.3%)
- Candidate: `gemma3:12b` at `localhost`, 2026-11-17T17:32:46Z: 24/30 (80.0%)

**0 regressions, 11 fixes.** Pass rate 43.3% -> 80.0% (+36.7 pts).

## Regressions (0)

None.

## Fixes (11)

Failed in the baseline, pass in the candidate.

- `voice-01`: used to fail `needs_human_match`; `needs_human` expected `false`, was `true`, now `false`
- `voice-02`: used to fail `priority_match`; `priority` expected `urgent`, was `high`, now `urgent`
- `voice-03`: used to fail `needs_human_match`; `needs_human` expected `false`, was `true`, now `false`
- `net-02`: used to fail `priority_match`; `priority` expected `normal`, was `low`, now `normal`
- `net-03`: used to fail `category_match`, `priority_match`, `needs_human_match`; `category` expected `network`, was `infrastructure`, now `network`; `priority` expected `high`, was `normal`, now `high`; `needs_human` expected `false`, was `true`, now `false`
- `net-06` (ambiguous case): used to fail `category_match`, `priority_match`; `category` expected `network`, was `voice`, now `network`; `priority` expected `normal`, was `high`, now `normal`
- `net-07` (ambiguous case): used to fail `category_match`; `category` expected `network`, was `voice`, now `network`
- `infra-05`: used to fail `priority_match`; `priority` expected `low`, was `normal`, now `low`
- `infra-06` (ambiguous case): used to fail `priority_match`; `priority` expected `urgent`, was `high`, now `urgent`
- `other-04`: used to fail `priority_match`, `needs_human_match`; `priority` expected `low`, was `normal`, now `low`; `needs_human` expected `false`, was `true`, now `false`
- `abstain-03`: used to fail `priority_match`, `keyword`, `safe_failure`; `category` expected `other`, was `infrastructure`, now `other`; `needs_human` expected `true`, was `false`, now `true`; `priority` expected `normal`, was `urgent`, now `normal`

## Score deltas

| Scorer | Baseline | Candidate | Change |
|---|---|---|---|
| **overall** | 43.3% | 80.0% | +36.7 pts |
| `json_schema` | 100.0% | 100.0% | 0.0 pts |
| `category_match` | 83.3% | 95.8% | +12.5 pts |
| `priority_match` | 52.0% | 84.0% | +32.0 pts |
| `needs_human_match` | 57.9% | 89.5% | +31.6 pts |
| `keyword` | 80.0% | 100.0% | +20.0 pts |
| `safe_failure` | 66.7% | 83.3% | +16.7 pts |

## Still failing (6)

`net-04`, `infra-07`, `other-01`, `other-02`, `other-03`, `abstain-04`
