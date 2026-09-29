from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]

    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def embed_text(text):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty or whitespace-only.")

    embedding = _model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding[0].tolist()