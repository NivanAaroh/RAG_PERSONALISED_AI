import argparse
import json
from collections import defaultdict
from pathlib import Path


METRICS = (
    "hit_at_5",
    "evidence_recall_at_5",
    "precision_at_5",
    "mrr",
)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_benchmark(path: str) -> list[dict]:
    data = load(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]

    raise ValueError("Benchmark must be a list or contain 'cases'.")


def get_cases(data: dict) -> list[dict]:
    return data["cases"]


def normalize_required_evidence(required_evidence) -> list[str]:
    result = []

    for item in required_evidence:
        if isinstance(item, dict):
            chunk_id = item.get("chunk_id")
            if chunk_id:
                result.append(chunk_id)
        else:
            result.append(item)

    return result


def metric_value(case: dict, metric: str) -> float:
    metrics = case.get("metrics") or {}

    aliases = {
        "hit_at_5": ("hit_at_5", "hit_at_k"),
        "evidence_recall_at_5": (
            "evidence_recall_at_5",
            "evidence_recall_at_k",
        ),
        "precision_at_5": (
            "precision_at_5",
            "precision_at_k",
        ),
        "mrr": ("mrr",),
    }

    for key in aliases[metric]:
        if key in metrics:
            return metrics[key]

    return 0.0


def recovered(case: dict) -> bool:
    return bool(metric_value(case, "hit_at_5"))


def evidence_ranks(
    case: dict,
    required_evidence: list[str],
) -> dict[str, int | None]:
    ranks = {chunk_id: None for chunk_id in required_evidence}

    for chunk in case.get("retrieved_chunks", []):
        chunk_id = chunk["chunk_id"]

        if chunk_id in ranks:
            ranks[chunk_id] = chunk["rank"]

    return ranks


def compare(
    dense: dict,
    bm25: dict,
    hybrid: dict,
    benchmark: list[dict],
) -> dict:
    dense_cases = get_cases(dense)
    bm25_cases = get_cases(bm25)
    hybrid_cases = get_cases(hybrid)

    if not (
        len(dense_cases)
        == len(bm25_cases)
        == len(hybrid_cases)
        == len(benchmark)
    ):
        raise ValueError(
            "Dense, BM25, hybrid, and benchmark case counts must match."
        )

    recovery = defaultdict(list)
    hybrid_losses = []
    per_question = []
    unsupported_cases = []

    supported_ids = []

    for index, benchmark_case in enumerate(benchmark):
        case_id = benchmark_case["id"]

        d = dense_cases[index]
        b = bm25_cases[index]
        h = hybrid_cases[index]

        support_status = benchmark_case["support_status"]
        category = benchmark_case.get("category", "unknown")
        question = benchmark_case["question"]

        required_evidence = normalize_required_evidence(
            benchmark_case["required_evidence"]
        )

        if support_status != "supported":
            status = "unsupported"
        else:
            supported_ids.append(case_id)

            d_hit = recovered(d)
            b_hit = recovered(b)
            h_hit = recovered(h)

            if d_hit and b_hit and h_hit:
                status = "BOTH_BASELINES_AND_HYBRID"

            elif d_hit and not b_hit and h_hit:
                status = "DENSE_ONLY_RECOVERY"

            elif b_hit and not d_hit and h_hit:
                status = "BM25_ONLY_RECOVERY"

            elif not d_hit and not b_hit and h_hit:
                status = "HYBRID_ONLY_RECOVERY"

            elif d_hit and b_hit and not h_hit:
                status = "HYBRID_LOSS_BOTH"
                hybrid_losses.append(case_id)

            elif d_hit and not b_hit and not h_hit:
                status = "HYBRID_LOSS_DENSE"
                hybrid_losses.append(case_id)

            elif b_hit and not d_hit and not h_hit:
                status = "HYBRID_LOSS_BM25"
                hybrid_losses.append(case_id)

            else:
                status = "NEITHER_RECOVERED"

            recovery[status].append(case_id)

        per_question.append(
            {
                "id": case_id,
                "question": question,
                "category": category,
                "support_status": support_status,
                "recovery_status": status,
                "required_evidence": required_evidence,
                "dense_required_evidence_ranks": evidence_ranks(
                    d,
                    required_evidence,
                ),
                "bm25_required_evidence_ranks": evidence_ranks(
                    b,
                    required_evidence,
                ),
                "hybrid_required_evidence_ranks": evidence_ranks(
                    h,
                    required_evidence,
                ),
            }
        )

        if support_status != "supported":
            unsupported_cases.append(
                {
                    "id": case_id,
                    "question": question,
                    "category": category,
                    "dense_candidates": [
                        chunk["chunk_id"]
                        for chunk in d.get("retrieved_chunks", [])
                    ],
                    "bm25_candidates": [
                        chunk["chunk_id"]
                        for chunk in b.get("retrieved_chunks", [])
                    ],
                    "hybrid_candidates": [
                        chunk["chunk_id"]
                        for chunk in h.get("retrieved_chunks", [])
                    ],
                }
            )

    category_metrics = {}

    categories = sorted(
        {
            benchmark_case.get("category", "unknown")
            for benchmark_case in benchmark
            if benchmark_case["support_status"] == "supported"
        }
    )

    for category in categories:
        indexes = [
            index
            for index, benchmark_case in enumerate(benchmark)
            if (
                benchmark_case["support_status"] == "supported"
                and benchmark_case.get("category", "unknown") == category
            )
        ]

        category_metrics[category] = {}

        for name, data in (
            ("dense", dense_cases),
            ("bm25", bm25_cases),
            ("hybrid", hybrid_cases),
        ):
            category_metrics[category][name] = {
                metric: sum(
                    metric_value(data[index], metric)
                    for index in indexes
                )
                / len(indexes)
                for metric in METRICS
            }

    return {
        "experiment": "day2_part4_retrieval_comparison",
        "dataset": "day2-benchmark-v1",
        "systems": {
            "dense": {
                "candidate_k": 20,
                "final_k": 5,
            },
            "bm25": {
                "candidate_k": 20,
                "final_k": 5,
            },
            "hybrid": {
                "dense_candidate_k": 20,
                "bm25_candidate_k": 20,
                "rrf_k": 60,
                "final_k": 5,
                "dense_weight": 1.0,
                "bm25_weight": 1.0,
            },
        },
        "aggregate": {
            "dense": dense["aggregate"],
            "bm25": bm25["aggregate"],
            "hybrid": hybrid["aggregate"],
        },
        "supported_cases": len(supported_ids),
        "recovery": {
            key: value
            for key, value in sorted(recovery.items())
        },
        "hybrid_loss_cases": hybrid_losses,
        "category_metrics": category_metrics,
        "unsupported_cases": unsupported_cases,
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dense", required=True)
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    result = compare(
        load(args.dense),
        load(args.bm25),
        load(args.hybrid),
        load_benchmark(args.benchmark),
    )

    Path(args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Comparison written to: {args.output}")
    print(f"Supported cases: {result['supported_cases']}")

    print("\nRecovery:")
    for key, cases in result["recovery"].items():
        print(f"  {key}: {len(cases)}")

    print(
        f"\nHybrid loss cases: "
        f"{len(result['hybrid_loss_cases'])}"
    )

    print(
        f"Unsupported cases: "
        f"{len(result['unsupported_cases'])}"
    )


if __name__ == "__main__":
    main()