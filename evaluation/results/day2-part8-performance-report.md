# Day 2 — Part 8: Performance Measurement Report

## 1. Status and Scope

**Status: MEASUREMENTS COMPLETE — REPORT UNDER REVIEW**

This report records initial retrieval-quality and retrieval-pipeline timing measurements for the frozen Day 2 benchmark. It does not change the retrieval implementation, evidence-sufficiency contract, grounding behavior, or benchmark dataset.

The purpose is to compare Dense, Hybrid/RRF, and Cross-Encoder retrieval using the recorded evaluation outputs. These results do not establish semantic answer-generation quality or end-to-end RAG latency.

Formal acceptance and freeze remain with Parent.

## 2. Environment and Reproducibility

- Benchmark: `day2-benchmark-v1` (`evaluation/datasets/day2-benchmark-v1.json`)
- Active document ID: `wp3-test`
- Cases: 28 total — 21 supported and 7 unsupported, ambiguous, or near-match cases
- Python: 3.14.7
- Platform: Windows 11, build `10.0.26200`
- Recorded Git revision in all three result artifacts: `a494e49d8f35de55dffc739b16e8d34dc6a8950c`
- Dense embedding model: `BAAI/bge-small-en-v1.5`
- Sparse retrieval: BM25, `k1=1.5`, `b=0.75`, tokenizer `lowercase + [a-z0-9]+`
- Fusion: Reciprocal Rank Fusion (RRF), `k=60`, with recorded weights of `1.0`
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`, CPU, higher scores ranked first
- Cross-Encoder model initialization time: approximately 3,793.91 ms
- BM25 index size recorded by the hybrid configurations: 45 documents/chunks

The Dense result identifies its dataset as `day2-benchmark-v1`; the Hybrid/RRF and Cross-Encoder results identify the dataset by filename, `day2-benchmark-v1.json`. These are different metadata representations of the benchmark used in this run.

The results were collected in separate runner executions. They are initial measurements, not repeated-run statistical estimates.

## 3. Retrieval-Quality Results

Metrics below are taken from the recorded JSON aggregate results. Percentages are converted from the stored fractions.

| Metric | Dense | Hybrid/RRF | Cross-Encoder |
|---|---:|---:|---:|
| Hit@5 | 90.48% | 95.24% | 100.00% |
| Evidence Recall@5 | 88.10% | 92.86% | 97.62% |
| Precision@5 | 20.00% | 20.95% | 21.90% |
| MRR | 0.7714 | 0.7738 | 0.8532 |

All configurations evaluated 21 supported cases and 7 unsupported, ambiguous, or near-match cases.

Within this benchmark run, Hybrid/RRF recorded higher Hit@5 and Evidence Recall@5 than Dense. Cross-Encoder recorded the highest values for Hit@5, Evidence Recall@5, Precision@5, and MRR among these three result files.

These are benchmark-specific retrieval observations. They do not establish statistical significance or prove that generated answers are more correct.

## 4. Latency Measurements

### 4.1 Recorded runner timings

| Configuration / stage | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| Dense runner retrieval call | 24.37 | Not recorded here | Not recorded here |
| Hybrid/RRF — Dense candidate retrieval | 32.81 | 32.39 | 40.54 |
| Hybrid/RRF — BM25 retrieval | 2.46 | 2.18 | 4.04 |
| Hybrid/RRF — RRF fusion | 0.095 | 0.093 | 0.136 |
| Hybrid/RRF — total timed pipeline | 35.37 | 35.10 | 44.16 |
| Cross-Encoder — Dense candidate retrieval | 24.07 | 23.99 | 29.62 |
| Cross-Encoder — BM25 retrieval | 1.74 | 1.53 | 2.59 |
| Cross-Encoder — RRF fusion | 0.109 | 0.099 | 0.163 |
| Cross-Encoder — reranking | 1,510.06 | 1,501.06 | 1,589.94 |
| Cross-Encoder — total timed pipeline | 1,535.99 | 1,528.41 | 1,612.94 |

### 4.2 Timing-scope limitations

- The Dense runner times its retrieval call at `top_k=5` across the 28 benchmark cases.
- Hybrid/RRF times Dense candidate retrieval at `top_k=20`, BM25 retrieval, and RRF fusion to a final `top_k=5`.
- Cross-Encoder times Dense candidate retrieval, BM25 retrieval, RRF candidate fusion, and Cross-Encoder reranking to a final `top_k=5`.
- BM25 index construction is measured separately. Cross-Encoder model initialization is also measured separately.
- The Dense mean is therefore not directly comparable with the broader Hybrid/RRF and Cross-Encoder pipeline totals.
- The Hybrid/RRF and Cross-Encoder totals do not include LLM generation or browser streaming. They must not be described as end-to-end RAG latency.
- These are single-run measurements. They do not quantify run-to-run variance, warm-versus-cold behavior, or production concurrency.

The recorded Cross-Encoder run has a substantial reranking cost relative to its other timed stages. Any decision about accepting that trade-off should consider both retrieval-quality results and latency requirements; this report does not itself approve a configuration.

## 5. Validation and Integrity Checks

Recorded checks for this measurement stage:

- All three JSON result artifacts exist and contain 28 cases each.
- Each artifact represents 21 supported and 7 non-supported, ambiguous, or near-match cases.
- Dense case records use `case_id`; Hybrid/RRF and Cross-Encoder case records use `id`.
- The recorded result artifacts share the same Git revision: `a494e49d8f35de55dffc739b16e8d34dc6a8950c`.
- The benchmark file exists at `evaluation/datasets/day2-benchmark-v1.json`.
- The test suite passed: 50 tests passed, with one non-blocking ChromaDB deprecation warning.
- Before report creation, the `day2-part8-performance` branch had a clean working tree and `HEAD` matched `origin/main`.

These checks establish artifact and test-suite status. They do not substitute for repeated performance trials or semantic evaluation of generated answers.

## 6. Limitations and Interpretation

1. **Single-run evidence:** Timing figures come from individual runner executions and should be treated as initial measurements.
2. **Different timer scopes:** Dense, Hybrid/RRF, and Cross-Encoder do not time identical operations. Direct speed rankings across their totals would be misleading.
3. **No end-to-end latency:** LLM time-to-first-token, answer-generation time, full RAG request latency, and browser rendering are not measured here.
4. **No semantic generation evaluation:** Retrieval metrics measure retrieval behavior. They do not independently prove that a generated answer is correct, complete, or well grounded.
5. **Fixed evaluation scope:** Results apply to this benchmark, document, environment, and configuration. Generalization to other corpora or workloads is not established.
6. **No statistical significance claim:** No repeated trials, confidence intervals, or significance tests are reported.
7. **Acceptance remains pending:** The report records measurements; Parent retains authority to accept, reject, or request further controlled measurements.

## 7. Artifacts and Decision

Result artifacts:

- `evaluation/results/day2-part8-dense.json`
- `evaluation/results/day2-part8-hybrid-rrf.json`
- `evaluation/results/day2-part8-cross-encoder.json`

Report:

- `evaluation/results/day2-part8-performance-report.md`

**Part 8 decision:** Initial measurement artifacts are present and their recorded aggregate results have been transcribed for review. The report must pass diff and scope review before it is committed. Part 8 is not formally accepted or frozen until Parent reviews it.

No retrieval runner, grounding logic, or benchmark data change is authorized by this report.
