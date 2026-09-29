from collections.abc import Sequence


def _required_ids(required_evidence: Sequence[str]) -> set[str]:
    return set(required_evidence)


def hit_at_k(
    required_evidence: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> int:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    required = _required_ids(required_evidence)

    if not required:
        return 0

    retrieved = set(retrieved_chunk_ids[:k])
    return int(bool(required & retrieved))


def evidence_recall_at_k(
    required_evidence: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    required = _required_ids(required_evidence)

    if not required:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    recovered = required & retrieved

    return len(recovered) / len(required)


def precision_at_k(
    required_evidence: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    retrieved = list(retrieved_chunk_ids[:k])

    if not retrieved:
        return 0.0

    required = _required_ids(required_evidence)
    relevant_count = sum(
        1 for chunk_id in retrieved if chunk_id in required
    )

    return relevant_count / len(retrieved)


def reciprocal_rank(
    required_evidence: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> float:
    required = _required_ids(required_evidence)

    if not required:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in required:
            return 1.0 / rank

    return 0.0
