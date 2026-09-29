import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_adapter import retrieve_for_evaluation
from evaluation.schemas import validate_dataset


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def run(dataset_path: str, top_k: int, output_path: str) -> None:
    with open(dataset_path, encoding="utf-8") as handle:
        dataset = validate_dataset(json.load(handle))

    cases = []
    hit_values = []
    recall_values = []
    precision_values = []
    mrr_values = []
    latencies = []

    for case in dataset.cases:
        started = time.perf_counter()

        retrieved = retrieve_for_evaluation(
            query=case.question,
            document_id=dataset.document_id,
            top_k=top_k,
        )

        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [item.chunk_id for item in retrieved]
        required_ids = [item.chunk_id for item in case.required_evidence]

        result = {
            "case_id": case.case_id,
            "question": case.question,
            "support_status": case.support_status,
            "required_evidence": [
                {
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "page": item.page,
                }
                for item in case.required_evidence
            ],
            "retrieved_chunks": [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "document_name": item.document_name,
                    "page": item.page,
                    "text": item.text,
                    "distance": item.distance,
                }
                for item in retrieved
            ],
            "metrics": {
                "hit_at_k": hit_at_k(required_ids, retrieved_ids, top_k),
                "evidence_recall_at_k": evidence_recall_at_k(
                    required_ids, retrieved_ids, top_k
                ),
                "precision_at_k": precision_at_k(
                    required_ids, retrieved_ids, top_k
                ),
                "mrr": reciprocal_rank(required_ids, retrieved_ids),
            },
            "latency_ms": latency_ms,
        }

        cases.append(result)
        hit_values.append(result["metrics"]["hit_at_k"])
        recall_values.append(result["metrics"]["evidence_recall_at_k"])
        precision_values.append(result["metrics"]["precision_at_k"])
        mrr_values.append(result["metrics"]["mrr"])

    supported_cases = [
        case for case in cases
        if case["support_status"] == "supported"
    ]

    supported_count = len(supported_cases)

    output = {
        "metadata": {
            "dataset_version": dataset.dataset_version,
            "document_id": dataset.document_id,
            "configuration_name": "dense-baseline",
            "top_k": top_k,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "git_revision": git_revision(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sparse": None,
            "fusion": None,
            "reranker": None,
        },
        "aggregate": {
            "case_count": len(cases),
            "supported_case_count": supported_count,
            "unsupported_case_count": len(cases) - supported_count,
            "supported_retrieval_metrics": {
                "hit_at_k": (
                    sum(
                        case["metrics"]["hit_at_k"]
                        for case in supported_cases
                    ) / supported_count
                    if supported_count else 0.0
                ),
                "evidence_recall_at_k": (
                    sum(
                        case["metrics"]["evidence_recall_at_k"]
                        for case in supported_cases
                    ) / supported_count
                    if supported_count else 0.0
                ),
                "precision_at_k": (
                    sum(
                        case["metrics"]["precision_at_k"]
                        for case in supported_cases
                    ) / supported_count
                    if supported_count else 0.0
                ),
                "mrr": (
                    sum(
                        case["metrics"]["mrr"]
                        for case in supported_cases
                    ) / supported_count
                    if supported_count else 0.0
                ),
            },
            "mean_latency_ms": sum(latencies) / len(latencies),
        },
        "cases": cases,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output["aggregate"], indent=2))
    print(f"results: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")

    run(args.dataset, args.top_k, args.output)


if __name__ == "__main__":
    main()
