import tiktoken


ENCODING = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 600
OVERLAP_TOKENS = 90


def chunk_document(document_id, document_name, pages):
    """
    Convert page-aware document text into deterministic token-based chunks.

    pages:
        List of dictionaries containing:
        {
            "page": int,
            "text": str
        }

    Returns:
        List of dictionaries containing:
        {
            "document_id": str,
            "document_name": str,
            "page": int,
            "chunk_id": str,
            "text": str
        }
    """

    chunks = []

    for page_data in pages:
        page = page_data["page"]
        text = page_data.get("text", "")

        if not text or not text.strip():
            continue

        token_ids = ENCODING.encode(text)

        start = 0
        chunk_index = 0

        while start < len(token_ids):
            end = min(start + TARGET_TOKENS, len(token_ids))

            chunk_token_ids = token_ids[start:end]
            chunk_text = ENCODING.decode(chunk_token_ids)

            chunk_id = f"{document_id}:p{page}:c{chunk_index}"

            chunks.append(
                {
                    "document_id": document_id,
                    "document_name": document_name,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

            if end >= len(token_ids):
                break

            start = end - OVERLAP_TOKENS
            chunk_index += 1

    return chunks