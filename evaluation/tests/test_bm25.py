from evaluation.bm25 import BM25Document, BM25Index, tokenize


def _index() -> BM25Index:
    return BM25Index(
        [
            BM25Document("chunk-a", "doc", "doc.pdf", 1, "BM25 lexical retrieval scoring"),
            BM25Document("chunk-b", "doc", "doc.pdf", 2, "dense neural retrieval embeddings"),
            BM25Document("chunk-c", "doc", "doc.pdf", 3, "BM25 ranking uses term frequency"),
        ]
    )


def test_tokenizer_is_normalized():
    assert tokenize("BM25, Retrieval!") == ["bm25", "retrieval"]


def test_exact_lexical_match_ranks_first():
    results = _index().search("BM25 lexical retrieval", 2)
    assert results[0][0].chunk_id == "chunk-a"


def test_top_k_is_respected():
    results = _index().search("retrieval", 1)
    assert len(results) == 1


def test_metadata_is_preserved():
    results = _index().search("BM25", 1)
    document = results[0][0]

    assert document.chunk_id == "chunk-a"
    assert document.document_id == "doc"
    assert document.page == 1
    assert document.text.startswith("BM25")


def test_results_are_deterministic():
    index = _index()

    first = index.search("retrieval", 3)
    second = index.search("retrieval", 3)

    assert [
        (doc.chunk_id, score) for doc, score in first
    ] == [
        (doc.chunk_id, score) for doc, score in second
    ]


def test_invalid_query_is_rejected():
    try:
        _index().search("   ", 5)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
