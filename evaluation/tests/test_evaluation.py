import json
import unittest

from evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)
from evaluation.schemas import validate_dataset


class EvaluationTests(unittest.TestCase):

    def test_perfect_retrieval(self):
        required = ["A"]
        retrieved = ["A", "B", "C"]

        self.assertEqual(hit_at_k(required, retrieved, 3), 1)
        self.assertEqual(evidence_recall_at_k(required, retrieved, 3), 1.0)
        self.assertEqual(precision_at_k(required, retrieved, 3), 1 / 3)
        self.assertEqual(reciprocal_rank(required, retrieved), 1.0)

    def test_required_evidence_missing(self):
        required = ["A"]
        retrieved = ["B", "C", "D"]

        self.assertEqual(hit_at_k(required, retrieved, 3), 0)
        self.assertEqual(evidence_recall_at_k(required, retrieved, 3), 0.0)
        self.assertEqual(precision_at_k(required, retrieved, 3), 0.0)
        self.assertEqual(reciprocal_rank(required, retrieved), 0.0)

    def test_partial_evidence(self):
        required = ["A", "B"]
        retrieved = ["A", "C", "D"]

        self.assertEqual(hit_at_k(required, retrieved, 3), 1)
        self.assertEqual(evidence_recall_at_k(required, retrieved, 3), 0.5)
        self.assertEqual(precision_at_k(required, retrieved, 3), 1 / 3)
        self.assertEqual(reciprocal_rank(required, retrieved), 1.0)

    def test_ranking_difference(self):
        required = ["A"]
        retrieved = ["B", "C", "A"]

        self.assertEqual(hit_at_k(required, retrieved, 3), 1)
        self.assertEqual(evidence_recall_at_k(required, retrieved, 3), 1.0)
        self.assertEqual(precision_at_k(required, retrieved, 3), 1 / 3)
        self.assertEqual(reciprocal_rank(required, retrieved), 1 / 3)

    def test_metric_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            hit_at_k(["A"], ["A"], 0)

        with self.assertRaises(ValueError):
            evidence_recall_at_k(["A"], ["A"], 0)

        with self.assertRaises(ValueError):
            precision_at_k(["A"], ["A"], 0)

    def test_valid_dataset(self):
        data = {
            "dataset_version": "smoke-v1",
            "corpus": {"document_id": "doc-1"},
            "cases": [
                {
                    "id": "smoke-001",
                    "question": "What is this?",
                    "support_status": "supported",
                    "required_evidence": [
                        {
                            "document_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "page": 1,
                        }
                    ],
                },
                {
                    "id": "smoke-002",
                    "question": "Is this supported?",
                    "support_status": "unsupported",
                    "required_evidence": [],
                },
            ],
        }

        dataset = validate_dataset(data)

        self.assertEqual(dataset.dataset_version, "smoke-v1")
        self.assertEqual(dataset.document_id, "doc-1")
        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(dataset.cases[1].support_status, "unsupported")

    def test_invalid_dataset_missing_version(self):
        data = {
            "corpus": {"document_id": "doc-1"},
            "cases": [],
        }

        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_invalid_dataset_missing_required_fields(self):
        data = {
            "dataset_version": "smoke-v1",
            "corpus": {"document_id": "doc-1"},
            "cases": [
                {
                    "id": "smoke-001",
                    "question": "Question",
                    "support_status": "supported",
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_invalid_support_status(self):
        data = {
            "dataset_version": "smoke-v1",
            "corpus": {"document_id": "doc-1"},
            "cases": [
                {
                    "id": "smoke-001",
                    "question": "Question",
                    "support_status": "unknown",
                    "required_evidence": [],
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_dataset_serialization_round_trip(self):
        data = {
            "dataset_version": "smoke-v1",
            "corpus": {"document_id": "doc-1"},
            "cases": [
                {
                    "id": "smoke-001",
                    "question": "Question",
                    "support_status": "supported",
                    "required_evidence": [
                        {
                            "document_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "page": 1,
                        }
                    ],
                }
            ],
        }

        serialized = json.dumps(data)
        loaded = json.loads(serialized)
        dataset = validate_dataset(loaded)

        self.assertEqual(dataset.dataset_version, "smoke-v1")
        self.assertEqual(
            dataset.cases[0].required_evidence[0].chunk_id,
            "chunk-1",
        )

    def test_metric_results_are_deterministic(self):
        required = ["A", "B"]
        retrieved = ["C", "A", "B"]

        first = (
            hit_at_k(required, retrieved, 3),
            evidence_recall_at_k(required, retrieved, 3),
            precision_at_k(required, retrieved, 3),
            reciprocal_rank(required, retrieved),
        )

        second = (
            hit_at_k(required, retrieved, 3),
            evidence_recall_at_k(required, retrieved, 3),
            precision_at_k(required, retrieved, 3),
            reciprocal_rank(required, retrieved),
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
