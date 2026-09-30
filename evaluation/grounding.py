from dataclasses import dataclass
from typing import Any, Callable, Iterable


REFUSAL_TEXT = (
    "The requested information cannot be established "
    "from the available document evidence."
)


@dataclass(frozen=True)
class EvidenceRequirement:
    evidence_id: str
    requirement: str
    supporting_chunk_ids: tuple[str, ...]
    supporting_pages: tuple[int, ...]
    entity: str | None = None
    attribute: str | None = None
    relation: str | None = None
    expected_fact: str | None = None


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    coverage: float
    required_evidence: tuple[EvidenceRequirement, ...]
    supported_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    assessment_reason: str


@dataclass(frozen=True)
class GroundingResult:
    sufficient_evidence: bool
    llm_invoked: bool
    grounded: bool
    answer: str | None
    sources: list[dict[str, Any]]
    provenance_valid: bool
    refusal: bool
    assessment: EvidenceAssessment


def validate_provenance(
    evidence: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]],
) -> bool:
    expected = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "page": item["page"],
        }
        for item in evidence
    ]

    actual = list(sources)

    return actual == expected


def build_sources(
    evidence: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "page": item["page"],
        }
        for item in evidence
    ]


def assess_evidence(
    retrieved_chunks: Iterable[dict[str, Any]],
    required_evidence: Iterable[EvidenceRequirement],
) -> EvidenceAssessment:
    retrieved_ids = {
        item["chunk_id"]
        for item in retrieved_chunks
    }

    requirements = tuple(required_evidence)

    if not requirements:
        return EvidenceAssessment(
            sufficient=False,
            coverage=0.0,
            required_evidence=requirements,
            supported_evidence=(),
            missing_evidence=(),
            assessment_reason="No explicit evidence requirements were supplied.",
        )

    supported = []
    missing = []

    for requirement in requirements:
        if requirement.supporting_chunk_ids and all(
            chunk_id in retrieved_ids
            for chunk_id in requirement.supporting_chunk_ids
        ):
            supported.append(requirement.evidence_id)
        else:
            missing.append(requirement.evidence_id)

    coverage = len(supported) / len(requirements)
    sufficient = coverage == 1.0

    if sufficient:
        reason = "All required evidence units are established by retrieved evidence."
    else:
        reason = (
            f"Required evidence coverage is incomplete: "
            f"{len(supported)}/{len(requirements)} evidence units established."
        )

    return EvidenceAssessment(
        sufficient=sufficient,
        coverage=coverage,
        required_evidence=requirements,
        supported_evidence=tuple(supported),
        missing_evidence=tuple(missing),
        assessment_reason=reason,
    )


def evaluate_grounding(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    required_evidence: Iterable[EvidenceRequirement] = (),
    llm: Callable[[list[dict[str, str]]], str] | None = None,
) -> GroundingResult:
    from backend.rag import (
        GROUNDED_SYSTEM_PROMPT,
        build_evidence,
        build_grounded_messages,
    )

    evidence = build_evidence(retrieved_chunks)

    assessment = assess_evidence(
        evidence,
        required_evidence,
    )

    if not assessment.sufficient:
        sources: list[dict[str, Any]] = []

        return GroundingResult(
            sufficient_evidence=False,
            llm_invoked=False,
            grounded=True,
            answer=REFUSAL_TEXT,
            sources=sources,
            provenance_valid=True,
            refusal=True,
            assessment=assessment,
        )

    if llm is None:
        raise ValueError(
            "An LLM callable is required when evidence is sufficient."
        )

    messages = build_grounded_messages(question, evidence)
    answer = llm(messages)

    sources = build_sources(evidence)

    return GroundingResult(
        sufficient_evidence=True,
        llm_invoked=True,
        grounded=True,
        answer=answer,
        sources=sources,
        provenance_valid=validate_provenance(evidence, sources),
        refusal=False,
        assessment=assessment,
    )


def instruction_like_text_is_data(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    from backend.rag import (
        GROUNDED_SYSTEM_PROMPT,
        build_evidence,
        build_grounded_messages,
    )

    evidence = build_evidence(retrieved_chunks)
    messages = build_grounded_messages(question, evidence)

    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    return (
        system_message == GROUNDED_SYSTEM_PROMPT
        and "Document evidence:" in user_message
        and "Question:" in user_message
    )