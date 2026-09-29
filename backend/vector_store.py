import chromadb


CHROMA_PATH = "backend/chroma_db"
COLLECTION_NAME = "documents"


_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def upsert_chunks(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count must equal embedding count")

    if not chunks:
        return 0

    ids = [chunk["chunk_id"] for chunk in chunks]

    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "document_name": chunk["document_name"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)


def count_chunks():
    return _collection.count()


def get_chunks(ids):
    return _collection.get(ids=ids)