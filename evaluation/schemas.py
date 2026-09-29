from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequiredEvidence:
    document_id: str
    chunk_id: str
    page: int


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    support_status: str
    required_evidence: tuple[RequiredEvidence, ...]


@dataclass(frozen=True)
class EvaluationDataset:
    dataset_version: str
    document_id: str
    cases: tuple[EvaluationCase, ...]


def validate_dataset(data: Any) -> EvaluationDataset:
    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object.")

    dataset_version = data.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError("dataset_version is required.")

    corpus = data.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("corpus is required.")

    document_id = corpus.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("corpus.document_id is required.")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list.")

    parsed_cases = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object.")

        case_id = case.get("id")
        question = case.get("question")
        support_status = case.get("support_status")
        required = case.get("required_evidence")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case {index}: id is required.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {case_id}: question is required.")

        if support_status not in {"supported", "unsupported"}:
            raise ValueError(
                f"Case {case_id}: support_status must be "
                "'supported' or 'unsupported'."
            )

        if not isinstance(required, list):
            raise ValueError(
                f"Case {case_id}: required_evidence must be a list."
            )

        evidence_items = []

        for evidence_index, item in enumerate(required):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Case {case_id}: required_evidence[{evidence_index}] "
                    "must be an object."
                )

            evidence_document_id = item.get("document_id")
            chunk_id = item.get("chunk_id")
            page = item.get("page")

            if not isinstance(evidence_document_id, str) or not evidence_document_id.strip():
                raise ValueError(
                    f"Case {case_id}: evidence document_id is required."
                )

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Case {case_id}: evidence chunk_id is required."
                )

            if not isinstance(page, int) or page <= 0:
                raise ValueError(
                    f"Case {case_id}: evidence page must be a positive integer."
                )

            evidence_items.append(
                RequiredEvidence(
                    document_id=evidence_document_id,
                    chunk_id=chunk_id,
                    page=page,
                )
            )

        parsed_cases.append(
            EvaluationCase(
                case_id=case_id,
                question=question,
                support_status=support_status,
                required_evidence=tuple(evidence_items),
            )
        )

    return EvaluationDataset(
        dataset_version=dataset_version,
        document_id=document_id,
        cases=tuple(parsed_cases),
    )
