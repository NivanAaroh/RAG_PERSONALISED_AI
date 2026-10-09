# Day 2 Part 7 — Regression and Reproducibility Report

## Status
Execution checks passed; evidence artifacts require explicit Git tracking because `evaluation/results/` ignores the generated JSON files.

## Frozen evaluation inputs
- Dataset: `day2-benchmark-v1`
- Corpus document ID: `wp3-test`
- Cases: 28 (21 supported, 7 unsupported/ambiguous/near-match)
- Source revision used by all seven result artifacts: `647f0ca8cdd7a89a1937716e8798f901ba5b7c25`
- Branch: `main`

## Retrieval reproducibility
Two runs per configuration produced identical ordered retrieved chunk IDs and per-case metrics across all 28 cases.

| Metric | Dense | Hybrid / RRF | Cross-Encoder |
|---|---:|---:|---:|
| Hit@5 | 0.904762 | 0.952381 | 1.000000 |
| Evidence Recall@5 | 0.880952 | 0.928571 | 0.976190 |
| Precision@5 | 0.200000 | 0.209524 | 0.219048 |
| MRR | 0.771429 | 0.773810 | 0.853175 |

Mean total latency varied between runs, as expected:
- Dense: 46.719 ms / 34.554 ms
- Hybrid / RRF: 36.774 ms / 37.873 ms
- Cross-Encoder: 1799.131 ms / 1827.538 ms

## Grounding and provenance
- Unsupported cases: 7/7 refused for each configuration.
- Unsupported cases: LLM not invoked in 7/7; sources empty in 7/7 for each configuration.
- Positive-case provenance valid: 21/21 per configuration.
- Sufficient positive evidence outcomes: Dense 18/21, Hybrid / RRF 20/21, Cross-Encoder 20/21.
- LLM invocation followed the evidence-sufficiency result; insufficient evidence remained blocked.

## Regression checks
- Python tests: 50 passed, 1 non-blocking ChromaDB deprecation warning.
- Frontend production build: passed.
- Production code diff under `backend` and frontend production paths: none detected.
- Frozen baseline remained `HEAD == origin/main == 647f0ca8cdd7a89a1937716e8798f901ba5b7c25` at audit time.

## Artifact files
- `day2-part7-dense-run1.json`
- `day2-part7-dense-run2.json`
- `day2-part7-hybrid-run1.json`
- `day2-part7-hybrid-run2.json`
- `day2-part7-cross-encoder-run1.json`
- `day2-part7-cross-encoder-run2.json`
- `day2-part7-grounding-run1.json`

## Conclusion
The observed repeatability, grounding, provenance, tests, and frontend build checks passed. Part 7 is not yet frozen in Git: the result artifacts and this report must be explicitly staged and committed, then pushed and verified. No Part 8 work is authorized by this report.
