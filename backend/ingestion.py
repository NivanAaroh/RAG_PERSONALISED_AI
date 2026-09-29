from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.split())


def ingest_pdf(
    pdf_path: str | Path,
    document_id: str,
    document_name: str | None = None,
) -> dict[str, Any]:
    """
    Extract a PDF into a structured, page-aware document representation.

    This function is intentionally independent of FastAPI, HTTP, LLMs,
    embeddings, vector databases, chunking, and retrieval.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    reader = PdfReader(str(path))

    resolved_document_name = document_name or path.name

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        normalized_text = _normalize_text(extracted_text)

        pages.append(
            {
                "document_id": document_id,
                "document_name": resolved_document_name,
                "page": page_number,
                "text": normalized_text,
            }
        )

    return {
        "document_id": document_id,
        "document_name": resolved_document_name,
        "page_count": len(pages),
        "pages": pages,
    }