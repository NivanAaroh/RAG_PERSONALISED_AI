import math
import re
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Document:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str


class BM25Index:
    def __init__(
        self,
        documents: list[BM25Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 corpus cannot be empty.")
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(doc.text) for doc in self.documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avgdl = sum(self._lengths) / len(self._lengths)

        document_frequency: dict[str, int] = {}

        for tokens in self._tokens:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        self._idf = {
            term: math.log(
                1.0
                + (len(self.documents) - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[BM25Document, float]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_terms = tokenize(query)

        if not query_terms:
            return []

        scored: list[tuple[BM25Document, float]] = []

        for index, tokens in enumerate(self._tokens):
            term_frequency: dict[str, int] = {}

            for token in tokens:
                term_frequency[token] = term_frequency.get(token, 0) + 1

            document_length = self._lengths[index]
            score = 0.0

            for term in query_terms:
                if term not in term_frequency:
                    continue

                tf = term_frequency[term]
                idf = self._idf.get(term, 0.0)

                denominator = (
                    tf
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b * document_length / self._avgdl
                    )
                )

                score += idf * (
                    tf * (self.k1 + 1.0)
                ) / denominator

            scored.append((self.documents[index], score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))

        return scored[:top_k]
