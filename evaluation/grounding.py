from dataclasses import dataclass
from typing import Any, Callable, Iterable

from backend.rag import (
    GROUNDED_SYSTEM_PROMPT,
    build_evidence,
    build_grounded_messages,
    has_sufficient_evidence,
)


REFUSAL_TEXT = (
    "The requested information cannot be established "
    "from the available document evidence."
)


@dataclass(frozen=True)
class GroundingResult:
    sufficient_evidence: bool
    llm_invoked: bool
    grounded: bool
    answer: str | None
    sources: list[dict[str, Any]]
    provenance_valid: bool
    refusal: bool


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


def build_sources(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "page": item["page"],
        }
        for item in evidence
    ]


def evaluate_grounding(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    llm: Callable[[list[dict[str, str]]], str] | None = None,
) -> GroundingResult:
    evidence = build_evidence(retrieved_chunks)
    sufficient = has_sufficient_evidence(question, evidence)

    if not sufficient:
        sources: list[dict[str, Any]] = []
        return GroundingResult(
            sufficient_evidence=False,
            llm_invoked=False,
            grounded=True,
            answer=REFUSAL_TEXT,
            sources=sources,
            provenance_valid=(sources == []),
            refusal=True,
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
    )


def instruction_like_text_is_data(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    evidence = build_evidence(retrieved_chunks)
    messages = build_grounded_messages(question, evidence)

    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    return (
        system_message == GROUNDED_SYSTEM_PROMPT
        and "Document evidence:" in user_message
        and "Question:" in user_message
    )