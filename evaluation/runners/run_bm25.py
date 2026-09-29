import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.bm25_retrieval import build_bm25_index
from evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)


def get_git_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standalone BM25 retrieval experiment."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    dataset = load_dataset(dataset_path)

    cases = dataset["cases"]
    dataset_version = dataset.get("dataset_version", "unknown")

    # Index construction is intentionally outside query latency measurement.
    index_start = time.perf_counter()
    index = build_bm25_index(args.document_id)
    index_build_ms = (time.perf_counter() - index_start) * 1000

    case_results = []
    supported_hit_values = []
    supported_recall_values = []
    supported_precision_values = []
    supported_mrr_values = []
    query_latencies = []

    supported_case_count = 0
    unsupported_case_count = 0

    for case in cases:
        query = case["question"]
        support_status = case["support_status"]
        required_evidence = case["required_evidence"]

        query_start = time.perf_counter()
        retrieved = index.search(query, args.top_k)
        latency_ms = (time.perf_counter() - query_start) * 1000

        query_latencies.append(latency_ms)

        retrieved_chunks = [
            {
                "chunk_id": document.chunk_id,
                "document_id": document.document_id,
                "document_name": document.document_name,
                "page": document.page,
                "text": document.text,
                "score": score,
                "rank": rank,
            }
            for rank, (document, score) in enumerate(retrieved, start=1)
        ]

        result = {
            "case_id": case["id"],
            "question": query,
            "support_status": support_status,
            "required_evidence": required_evidence,
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": latency_ms,
        }

        # ✅ Prepare chunk ID sequences for metrics
        required_chunk_ids = [
            evidence["chunk_id"]
            for evidence in required_evidence
        ]

        retrieved_chunk_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        if support_status == "supported":
            supported_case_count += 1

            hit = hit_at_k(
                required_chunk_ids,
                retrieved_chunk_ids,
                args.top_k,
            )
            recall = evidence_recall_at_k(
                required_chunk_ids,
                retrieved_chunk_ids,
                args.top_k,
            )
            precision = precision_at_k(
                required_chunk_ids,
                retrieved_chunk_ids,
                args.top_k,
            )
            mrr = reciprocal_rank(
                required_chunk_ids,
                retrieved_chunk_ids,
            )

            result["metrics"] = {
                "hit_at_k": hit,
                "evidence_recall_at_k": recall,
                "precision_at_k": precision,
                "mrr": mrr,
            }

            supported_hit_values.append(hit)
            supported_recall_values.append(recall)
            supported_precision_values.append(precision)
            supported_mrr_values.append(mrr)

        elif support_status == "unsupported":
            unsupported_case_count += 1
            result["metrics"] = None

        else:
            raise ValueError(
                f"Unsupported support_status for {case['id']}: "
                f"{support_status}"
            )

        case_results.append(result)

    supported_count = supported_case_count

    aggregate = {
        "case_count": len(cases),
        "supported_case_count": supported_case_count,
        "unsupported_case_count": unsupported_case_count,
        "supported_retrieval_metrics": {
            "hit_at_k": (
                sum(supported_hit_values) / supported_count
                if supported_count
                else 0.0
            ),
            "evidence_recall_at_k": (
                sum(supported_recall_values) / supported_count
                if supported_count
                else 0.0
            ),
            "precision_at_k": (
                sum(supported_precision_values) / supported_count
                if supported_count
                else 0.0
            ),
            "mrr": (
                sum(supported_mrr_values) / supported_count
                if supported_count
                else 0.0
            ),
        },
        "query_latency_ms": {
            "mean": (
                statistics.mean(query_latencies)
                if query_latencies
                else 0.0
            ),
            "median": (
                statistics.median(query_latencies)
                if query_latencies
                else 0.0
            ),
            "p95": percentile(query_latencies, 0.95),
        },
        "index_build_ms": index_build_ms,
    }

    result_document = {
        "experiment": "day2-part3-bm25",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": get_git_revision(),
        "dataset": {
            "path": str(dataset_path),
            "version": dataset_version,
        },
        "configuration": {
            "method": "bm25",
            "top_k": args.top_k,
            "document_id": args.document_id,
            "k1": index.k1,
            "b": index.b,
            "tokenizer": "lowercase + [a-z0-9]+ terms",
            "corpus_chunk_count": len(index.documents),
        },
        "aggregate": aggregate,
        "cases": case_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result_document, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"\nResult written to: {output_path}")


if __name__ == "__main__":
    main()
