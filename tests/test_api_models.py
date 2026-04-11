from __future__ import annotations

import unittest

from pydantic import ValidationError

from vector_gateway.models.api import (
    EmbedRequest,
    EnsureCollectionRequest,
    PayloadSetRequest,
    RetrieveRequest,
    SearchRequest,
    TransformSparseRequest,
    UpsertPoint,
)


class ApiModelValidationTest(unittest.TestCase):
    def test_embed_rejects_both_text_and_texts(self) -> None:
        with self.assertRaises(ValidationError):
            EmbedRequest(text="one", texts=["two"])

    def test_search_allows_query_text_only(self) -> None:
        request = SearchRequest(collection="knowledge", query_text="find this")
        self.assertEqual(request.query_text, "find this")
        self.assertEqual(request.search_mode, "auto")

    def test_search_rejects_invalid_mode(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(collection="knowledge", query_text="find this", search_mode="lexical")

    def test_search_rejects_sparse_mode_with_dense_vector(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(collection="knowledge", query_text="find this", vector=[0.1, 0.2], search_mode="sparse")

    def test_upsert_point_rejects_non_finite_dense_vector(self) -> None:
        with self.assertRaises(ValidationError):
            UpsertPoint(vector=[0.1, float("nan")], payload={})

    def test_upsert_point_rejects_invalid_sparse_vector_shape(self) -> None:
        with self.assertRaises(ValidationError):
            UpsertPoint(
                vector={"sparse": {"indices": [3, 2], "values": [0.1, 0.2]}},
                payload={},
            )

    def test_retrieve_rejects_empty_ids(self) -> None:
        with self.assertRaises(ValidationError):
            RetrieveRequest(collection="knowledge", ids=[])

    def test_payload_set_rejects_empty_payload(self) -> None:
        with self.assertRaises(ValidationError):
            PayloadSetRequest(collection="knowledge", ids=["k1"], payload={})

    def test_ensure_collection_rejects_conflicting_vector_names(self) -> None:
        with self.assertRaises(ValidationError):
            EnsureCollectionRequest(
                collection="decision_memory_v2",
                vector_size=1024,
                vector_name="dense",
                sparse_vector_name="dense",
            )

    def test_transform_sparse_rejects_blank_text(self) -> None:
        with self.assertRaises(ValidationError):
            TransformSparseRequest(texts=["ok", "  "])


if __name__ == "__main__":
    unittest.main()
