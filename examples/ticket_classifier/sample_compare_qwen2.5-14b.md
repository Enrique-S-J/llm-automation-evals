# Eval comparison: ticket_classifier

- Baseline: `llama3.2:3b` at `localhost`, 2026-11-17T17:33:43Z: 13/30 (43.3%)
- Candidate: `qwen2.5:14b` at `localhost`, 2026-11-17T17:31:57Z: 21/30 (70.0%)

**2 regressions, 10 fixes.** Pass rate 43.3% -> 70.0% (+26.7 pts).

## Regressions (2)

Passed in the baseline, fail in the candidate.

- `voice-06` (ambiguous case): now fails `priority_match`; `priority` expected `normal`, was `normal`, now `high`
- `infra-03`: now fails `priority_match`; `priority` expected `low`, was `low`, now `normal`

## Fixes (10)

Failed in the baseline, pass in the candidate.

- `voice-02`: used to fail `priority_match`; `priority` expected `urgent`, was `high`, now `urgent`
- `voice-03`: used to fail `needs_human_match`; `needs_human` expected `false`, was `true`, now `false`
- `net-02`: used to fail `priority_match`; `priority` expected `normal`, was `low`, now `normal`
- `net-03`: used to fail `category_match`, `priority_match`, `needs_human_match`; `category` expected `network`, was `infrastructure`, now `network`; `priority` expected `high`, was `normal`, now `high`; `needs_human` expected `false`, was `true`, now `false`
- `infra-05`: used to fail `priority_match`; `priority` expected `low`, was `normal`, now `low`
- `other-01`: used to fail `priority_match`, `needs_human_match`; `priority` expected `high`, was `low`, now `high`; `needs_human` expected `false`, was `true`, now `false`
- `other-02`: used to fail `priority_match`, `needs_human_match`; `priority` expected `low`, was `normal`, now `low`; `needs_human` expected `false`, was `true`, now `false`
- `other-03`: used to fail `needs_human_match`; `needs_human` expected `false`, was `true`, now `false`
- `other-04`: used to fail `priority_match`, `needs_human_match`; `priority` expected `low`, was `normal`, now `low`; `needs_human` expected `false`, was `true`, now `false`
- `abstain-03`: used to fail `priority_match`, `keyword`, `safe_failure`; `category` expected `other`, was `infrastructure`, now `other`; `needs_human` expected `true`, was `false`, now `true`; `priority` expected `normal`, was `urgent`, now `normal`

## Score deltas

| Scorer | Baseline | Candidate | Change |
|---|---|---|---|
| **overall** | 43.3% | 70.0% | +26.7 pts |
| `json_schema` | 100.0% | 100.0% | 0.0 pts |
| `category_match` | 83.3% | 83.3% | 0.0 pts |
| `priority_match` | 52.0% | 76.0% | +24.0 pts |
| `needs_human_match` | 57.9% | 100.0% | +42.1 pts |
| `keyword` | 80.0% | 100.0% | +20.0 pts |
| `safe_failure` | 66.7% | 83.3% | +16.7 pts |

## Still failing (7)

`voice-01`, `net-04`, `net-06`, `net-07`, `infra-06`, `infra-07`, `abstain-04`
