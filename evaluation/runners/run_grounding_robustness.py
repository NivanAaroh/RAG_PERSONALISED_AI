from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evaluation.bm25_retrieval import build_bm25_index
from evaluation.cross_encoder_reranker import (
    DEFAULT_DEVICE,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from evaluation.grounding import EvidenceRequirement, evaluate_grounding
from evaluation.hybrid_retrieval import (
    bm25_to_retrieved_chunks,
    fuse_retrieval_results,
)
from evaluation.retrieval_adapter import retrieve_for_evaluation


DENSE_TOP_K = 5
BM25_CANDIDATE_K = 20
RRF_CANDIDATE_K = 20
FINAL_TOP_K = DEFAULT_FINAL_TOP_K
RRF_K = 60


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["cases"]


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def as_dicts(items) -> list[dict]:
    output = []
    for item in items:
        if hasattr(item, "__dict__"):
            output.append(dict(item.__dict__))
        else:
            output.append(dict(item))
    return output


def fake_llm(messages) -> str:
    return "Grounded evaluation answer."


def parse_requirements(case: dict) -> tuple[EvidenceRequirement, ...]:
    return tuple(
        EvidenceRequirement(
            evidence_id=item["evidence_id"],
            requirement=item["requirement"],
            supporting_chunk_ids=tuple(item["supporting_chunk_ids"]),
            supporting_pages=tuple(item["supporting_pages"]),
            entity=item.get("entity"),
            attribute=item.get("attribute"),
            relation=item.get("relation"),
            expected_fact=item.get("expected_fact"),
        )
        for item in case.get("evidence_requirements", [])
    )


def run(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    document_id = "wp3-test"

    bm25_index = build_bm25_index(document_id)
    reranker = CrossEncoderReranker(
        model_name=DEFAULT_RERANKER_MODEL,
        device=DEFAULT_DEVICE,
    )

    configurations = {
        "dense": [],
        "hybrid_rrf": [],
        "cross_encoder": [],
    }

    cases = []

    for case in dataset:
        question = case["question"]
        support_status = case["support_status"]
        required_evidence = parse_requirements(case)

        dense_results = retrieve_for_evaluation(
            query=question,
            document_id=document_id,
            top_k=DENSE_TOP_K,
        )

        bm25_results = bm25_to_retrieved_chunks(
            bm25_index,
            question,
            BM25_CANDIDATE_K,
        )

        hybrid_candidates = fuse_retrieval_results(
            dense_results,
            bm25_results,
            rrf_k=RRF_K,
            top_k=RRF_CANDIDATE_K,
        )

        cross_encoder_results = reranker.rerank(
            question,
            hybrid_candidates,
            top_k=FINAL_TOP_K,
        )

        retrieval_sets = {
            "dense": dense_results,
            "hybrid_rrf": hybrid_candidates[:FINAL_TOP_K],
            "cross_encoder": cross_encoder_results,
        }

        config_results = {}

        for config_name, retrieved in retrieval_sets.items():
            retrieved_dicts = as_dicts(retrieved)

            result = evaluate_grounding(
                question,
                retrieved_dicts,
                required_evidence=required_evidence,
                llm=fake_llm,
            )

            expected_supported = support_status == "supported"

            result_record = {
                "sufficient_evidence": result.sufficient_evidence,
                "llm_invoked": result.llm_invoked,
                "grounded": result.grounded,
                "refusal": result.refusal,
                "answer": result.answer,
                "sources": result.sources,
                "provenance_valid": result.provenance_valid,
                "retrieved_chunk_ids": [
                    item["chunk_id"] for item in retrieved_dicts
                ],
                "required_evidence": [
                    item["chunk_id"] for item in case["required_evidence"]
                ],
                "evidence_assessment": {
                    "sufficient": result.assessment.sufficient,
                    "coverage": result.assessment.coverage,
                    "required_evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "requirement": item.requirement,
                            "supporting_chunk_ids": list(
                                item.supporting_chunk_ids
                            ),
                            "supporting_pages": list(item.supporting_pages),
                            "entity": item.entity,
                            "attribute": item.attribute,
                            "relation": item.relation,
                            "expected_fact": item.expected_fact,
                        }
                        for item in result.assessment.required_evidence
                    ],
                    "supported_evidence": list(
                        result.assessment.supported_evidence
                    ),
                    "missing_evidence": list(
                        result.assessment.missing_evidence
                    ),
                    "assessment_reason": result.assessment.assessment_reason,
                },
                "expected_supported": expected_supported,
            }

            config_results[config_name] = result_record
            configurations[config_name].append(result_record)

        cases.append(
            {
                "id": case["id"],
                "question": question,
                "support_status": support_status,
                "category": case["category"],
                "required_evidence": case["required_evidence"],
                "evidence_requirements": case.get(
                    "evidence_requirements", []
                ),
                "configurations": config_results,
            }
        )

    summary = {}

    for config_name, results in configurations.items():
        positive = [
            item for item in results if item["expected_supported"]
        ]
        negative = [
            item for item in results if not item["expected_supported"]
        ]

        summary[config_name] = {
            "positive_cases": len(positive),
            "positive_sufficient": sum(
                item["sufficient_evidence"] for item in positive
            ),
            "positive_llm_invoked": sum(
                item["llm_invoked"] for item in positive
            ),
            "positive_provenance_valid": sum(
                item["provenance_valid"] for item in positive
            ),
            "negative_cases": len(negative),
            "negative_refused": sum(
                item["refusal"] for item in negative
            ),
            "negative_llm_not_invoked": sum(
                not item["llm_invoked"] for item in negative
            ),
            "negative_empty_sources": sum(
                not item["sources"] for item in negative
            ),
        }

    output = {
        "experiment": "day2_part6_grounding_robustness",
        "dataset": dataset_path.name,
        "document_id": document_id,
        "configurations": [
            "dense",
            "hybrid_rrf",
            "cross_encoder",
        ],
        "config": {
            "dense_top_k": DENSE_TOP_K,
            "bm25_candidate_k": BM25_CANDIDATE_K,
            "rrf_candidate_k": RRF_CANDIDATE_K,
            "rrf_k": RRF_K,
            "final_top_k": FINAL_TOP_K,
            "cross_encoder_model": DEFAULT_RERANKER_MODEL,
        },
        "summary": summary,
        "cases": cases,
        "git_revision": git_revision(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Grounding robustness results written to: {output_path}")

    for config_name, values in summary.items():
        print(f"\n[{config_name}]")
        print(
            f"Positive sufficient: "
            f"{values['positive_sufficient']}/{values['positive_cases']}"
        )
        print(
            f"Positive LLM invoked: "
            f"{values['positive_llm_invoked']}/{values['positive_cases']}"
        )
        print(
            f"Positive provenance valid: "
            f"{values['positive_provenance_valid']}/{values['positive_cases']}"
        )
        print(
            f"Negative refused: "
            f"{values['negative_refused']}/{values['negative_cases']}"
        )
        print(
            f"Negative LLM not invoked: "
            f"{values['negative_llm_not_invoked']}/{values['negative_cases']}"
        )
        print(
            f"Negative empty sources: "
            f"{values['negative_empty_sources']}/{values['negative_cases']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
