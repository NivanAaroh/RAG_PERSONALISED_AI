# Day 2 Part 6 — Grounding Revalidation Report

## Status
Technical revalidation PASS — pending Parent review and acceptance.

## Frozen evaluation inputs
- Dataset: day2-benchmark-v1.json
- Corpus document ID: wp3-test
- Cases: 28 (21 supported, 7 unsupported/ambiguous/near-match)
- Contract revision: 647f0ca8cdd7a89a1937716e8798f901ba5b7c25
- Execution revision: 75251ad96a4c85ffe9d6286cd5a3f37843abca20
- Configurations: Dense, Hybrid / RRF, Cross-Encoder

## Commands executed
- python -m evaluation.runners.run_grounding_robustness --dataset evaluation/datasets/day2-benchmark-v1.json --output evaluation/results/day2-part6-revalidation.json
- python -m pytest

## Unsupported-case checks

| Check | Dense | Hybrid / RRF | Cross-Encoder |
|---|---:|---:|---:|
| Refused | 7/7 | 7/7 | 7/7 |
| LLM not invoked | 7/7 | 7/7 | 7/7 |
| Empty sources | 7/7 | 7/7 | 7/7 |
| Deterministic refusal | 7/7 | 7/7 | 7/7 |

Refusal text: The requested information cannot be established from the available document evidence.

## Supported-case checks

| Check | Dense | Hybrid / RRF | Cross-Encoder |
|---|---:|---:|---:|
| Supported cases | 21 | 21 | 21 |
| Sufficient evidence and LLM invoked | 18/21 | 20/21 | 20/21 |
| Provenance valid | 21/21 | 21/21 | 21/21 |
| Accepted cases passing required-chunk audit | 18/18 | 20/20 | 20/20 |

The direct benchmark audit found zero accepted supported cases missing declared supporting chunk IDs.

Insufficient supported cases remained blocked:
- Dense: d2-004, d2-010, d2-012.
- Hybrid / RRF: d2-004.
- Cross-Encoder: d2-004.

Insufficient cases did not invoke the LLM and returned empty sources.

## Provenance and implementation
- Zero anomalies in the case-level provenance/evidence diagnostic.
- Source chunk IDs were checked against retrieved chunk IDs.
- The gate requires all declared supporting chunk IDs for every evidence requirement; coverage must equal 1.0.
- Insufficient evidence triggers deterministic refusal, no LLM invocation, and empty sources.
- The evaluator uses a fake LLM; this validates gate behavior, not semantic correctness of real LLM-generated answers.

## Tests
- Command: python -m pytest
- Result: 50 passed, 1 non-blocking ChromaDB deprecation warning.
- Duration: 15.83 seconds.

## Artifacts
- evaluation/results/day2-part6-revalidation.json
- evaluation/results/day2-part6-revalidation-report.md

## Verdict
Technical revalidation checks passed against the revised contract and frozen benchmark. Parent review and formal acceptance are still required. Part 8 remains blocked until Parent accepts the revalidation result.

No production logic or benchmark data was changed during this revalidation.
