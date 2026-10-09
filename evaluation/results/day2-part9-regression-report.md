# Day 2 — Part 9 Regression Protection Report

## Verdict

**PASS — Regression invariants passed; Parent acceptance pending.**

This report records the regression audit performed against the accepted Day 2 baseline. It does not authorize Part 10 or claim formal freeze.

## Baseline and scope

- Branch: `day2-part8-performance`
- Expected starting revision: `da3d23728ad267b145fdff925ce377ac2cbe4e6b`
- Recorded regression-run revision: `da3d23728ad267b145fdff925ce377ac2cbe4e6b`
- Benchmark: `evaluation/datasets/day2-benchmark-v1.json`
- Benchmark SHA-256: `427CDE84C0A317AC9AA8739D287F8194AA149B27D0903900C17EC253D194915C`
- Benchmark version: `day2-benchmark-v1`
- Cases: 28 total — 21 supported and 7 unsupported
- Configurations audited: `dense`, `hybrid_rrf`, `cross_encoder`
- Application/retrieval/gating code changes: none authorized or made during this regression audit

The benchmark hash matches the recorded baseline. Part 8 historical artifacts were not modified.

## Grounding and provenance invariants

The regression artifact was parsed successfully and contains 28 cases across all three configurations.

| Invariant | Dense | Hybrid/RRF | Cross-encoder |
|---|---:|---:|---:|
| Supported cases | 21 | 21 | 21 |
| Supported cases with sufficient evidence | 18 | 20 | 20 |
| Supported cases with valid provenance | 21/21 | 21/21 | 21/21 |
| Unsupported cases refused | 7/7 | 7/7 | 7/7 |
| Unsupported cases with LLM not invoked | 7/7 | 7/7 | 7/7 |
| Unsupported cases with empty sources | 7/7 | 7/7 | 7/7 |

All seven unsupported cases in each configuration fail closed: the system refuses, does not invoke the LLM, and returns no sources.

The audit found zero invariant violations across the 84 configuration-case records (28 cases × 3 configurations).

## Supported-case limitations

Three supported cases are not sufficient to invoke generation under dense retrieval:

- `d2-004`: insufficient evidence in all three configurations. The required multi-part evidence was incomplete.
- `d2-010`: insufficient evidence under dense retrieval.
- `d2-012`: insufficient evidence under dense retrieval.

Hybrid/RRF and cross-encoder retrieval met the sufficiency gate for `d2-010` and `d2-012`, but not `d2-004`.

These are observed evidence-retrieval coverage limitations. For the affected cases, generation was correctly withheld when the evidence-sufficiency check failed. The audit did not identify a sufficiency-gate invariant violation. This report does not claim that all supported questions can be answered.

## Automated tests

Command:

`python -m pytest evaluation/tests -q`

Result: **50 passed, 0 failed**, in 13.43 seconds.

One non-blocking ChromaDB dependency deprecation warning was reported for `asyncio.iscoroutinefunction`. No dependency or unrelated code change was made to address it.

## Frontend production build

Command: `npm run build` from `frontend/`

Result: **PASS**. Vite transformed 175 modules and completed the production build in 1.25 seconds.

## Scope and integrity checks

- Starting branch and revision matched the expected Part 8 accepted state.
- Working tree was clean before the regression artifact was considered for submission.
- Benchmark SHA-256 matched the baseline hash.
- No tracked changes or modifications to the Part 8 historical artifacts were detected.
- The regression JSON is ignored by the repository rule for `evaluation/results/`; it must be explicitly staged for this report to be reproducible from the submitted commit.
- The regression JSON's `experiment` field remains `day2_part6_grounding_robustness`. This generated artifact was preserved unchanged and reused as Part 9 regression evidence; its historical metadata was not silently rewritten.
- Performance measurements from earlier parts are not reinterpreted here. Single-run measurements and differing timer scopes remain limitations; this regression audit does not establish semantic answer quality or end-to-end latency.

## Final disposition

**Part 9 validation: PASS, pending Parent review and formal acceptance.**

The tested regression invariants, full evaluation test suite, and frontend production build passed. Retrieval coverage limitations are documented above and are not hidden by the regression verdict.

No retrieval algorithms, ranking parameters, benchmark ground truth, evidence-sufficiency logic, or application implementation were changed for this report.

Part 10 remains unauthorized until Parent formally accepts Part 9.
