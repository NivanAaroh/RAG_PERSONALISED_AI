# backend/rag.py

from backend.llm import client, model
from backend.retrieval import retrieve

GROUNDED_SYSTEM_PROMPT = (
    "You are a document-grounded AI assistant.\n"
    "Answer the user's question using only the supplied document evidence.\n"
    "Do not use outside knowledge.\n"
    "If the supplied evidence does not establish the answer, say: "
    "\"The requested information cannot be established from the available "
    "document evidence.\"\n"
    "Do not invent facts, sources, pages, or citations. "
    "Never mention citations, sources, pages, chunk IDs, document names, "
    "or source references in your answer. "
    "The application provides source metadata separately."
)


def format_evidence(evidence):
    sections = []

    for item in evidence:
        sections.append(
            f"[Document: {item['document_name']} | "
            f"Page: {item['page']} | "
            f"Chunk: {item['chunk_id']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(sections)


def build_grounded_messages(question, evidence):
    evidence_text = format_evidence(evidence)

    return [
        {
            "role": "system",
            "content": GROUNDED_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Document evidence:\n"
                f"--- BEGIN EVIDENCE ---\n"
                f"{evidence_text}\n"
                f"--- END EVIDENCE ---"
            ),
        },
    ]


def generate_grounded_response(question, evidence):
    messages = build_grounded_messages(question, evidence)
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in response_stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


def build_evidence(retrieved_chunks):
    evidence = []

    for chunk in retrieved_chunks:
        evidence.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return evidence


def has_sufficient_evidence(question, evidence):
    if not evidence:
        return False

    question_terms = {
        term.lower()
        for term in question.split()
        if len(term) >= 4
    }

    evidence_text = " ".join(
        item["text"].lower()
        for item in evidence
    )

    matched_terms = sum(
        1
        for term in question_terms
        if term in evidence_text
    )

    return matched_terms >= 2


def run_rag(question, document_id, top_k=5):
    retrieved_chunks = retrieve(
        question,
        document_id,
        top_k,
    )

    evidence = build_evidence(retrieved_chunks)

    if not has_sufficient_evidence(question, evidence):
        return evidence, None

    return evidence, generate_grounded_response(question, evidence)
