import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense baseline and BM25 retrieval results."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_json(Path(args.dataset))
    dense = load_json(Path(args.dense))
    bm25 = load_json(Path(args.bm25))

    categories = {
        case["id"]: case.get("category")
        for case in dataset["cases"]
    }

    dense_cases = {
        case["case_id"]: case
        for case in dense["cases"]
    }
    bm25_cases = {
        case["case_id"]: case
        for case in bm25["cases"]
    }

    comparisons = []

    for case_id in categories:
        dense_case = dense_cases[case_id]
        bm25_case = bm25_cases[case_id]

        dense_metrics = dense_case.get("metrics")
        bm25_metrics = bm25_case.get("metrics")

        if dense_metrics is None or bm25_metrics is None:
            comparisons.append(
                {
                    "case_id": case_id,
                    "category": categories[case_id],
                    "support_status": dense_case["support_status"],
                    "observation": "UNSUPPORTED_CASE",
                    "dense_top_chunks": [
                        chunk["chunk_id"]
                        for chunk in dense_case["retrieved_chunks"]
                    ],
                    "bm25_top_chunks": [
                        chunk["chunk_id"]
                        for chunk in bm25_case["retrieved_chunks"]
                    ],
                }
            )
            continue

        dense_hit = dense_metrics["hit_at_k"] == 1
        bm25_hit = bm25_metrics["hit_at_k"] == 1
        rank_observation = None

        if bm25_hit and not dense_hit:
            observation = "BM25_RECOVERED_DENSE_MISS"

        elif dense_hit and not bm25_hit:
            observation = "DENSE_RECOVERED_BM25_MISS"

        elif dense_hit and bm25_hit:
            observation = "BOTH_RECOVERED"

            if bm25_metrics["mrr"] > dense_metrics["mrr"]:
                rank_observation = "BM25_HIGHER_RANK"
            elif bm25_metrics["mrr"] < dense_metrics["mrr"]:
                rank_observation = "BM25_LOWER_RANK"
            else:
                rank_observation = "SAME_RANK"

        else:
            observation = "BOTH_MISSED"

        comparison = {
            "case_id": case_id,
            "category": categories[case_id],
            "support_status": dense_case["support_status"],
            "observation": observation,
            "dense": {
                "hit_at_k": dense_metrics["hit_at_k"],
                "evidence_recall_at_k": dense_metrics[
                    "evidence_recall_at_k"
                ],
                "precision_at_k": dense_metrics["precision_at_k"],
                "mrr": dense_metrics["mrr"],
                "top_chunks": [
                    chunk["chunk_id"]
                    for chunk in dense_case["retrieved_chunks"]
                ],
            },
            "bm25": {
                "hit_at_k": bm25_metrics["hit_at_k"],
                "evidence_recall_at_k": bm25_metrics[
                    "evidence_recall_at_k"
                ],
                "precision_at_k": bm25_metrics["precision_at_k"],
                "mrr": bm25_metrics["mrr"],
                "top_chunks": [
                    chunk["chunk_id"]
                    for chunk in bm25_case["retrieved_chunks"]
                ],
            },
        }

        if rank_observation is not None:
            comparison["rank_observation"] = rank_observation

        comparisons.append(comparison)

    supported = [
        item
        for item in comparisons
        if item["support_status"] == "supported"
    ]

    observation_counts = {}
    rank_counts = {}

    for item in supported:
        observation = item["observation"]
        observation_counts[observation] = (
            observation_counts.get(observation, 0) + 1
        )

        rank_observation = item.get("rank_observation")
        if rank_observation:
            rank_counts[rank_observation] = (
                rank_counts.get(rank_observation, 0) + 1
            )

    aggregate = {
        "supported_case_count": len(supported),
        "observation_counts": observation_counts,
        "rank_observation_counts": rank_counts,
        "dense_metrics": dense["aggregate"]["supported_retrieval_metrics"],
        "bm25_metrics": bm25["aggregate"]["supported_retrieval_metrics"],
        "dense_latency_ms": dense["aggregate"].get(
            "mean_latency_ms"
        ),
        "bm25_query_latency_ms": bm25["aggregate"].get(
            "query_latency_ms"
        ),
    }

    result = {
        "experiment": "day2-part3-bm25-comparison",
        "dataset": str(Path(args.dataset)),
        "dense_result": str(Path(args.dense)),
        "bm25_result": str(Path(args.bm25)),
        "aggregate": aggregate,
        "cases": comparisons,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"\nComparison written to: {output_path}")


if __name__ == "__main__":
    main()
