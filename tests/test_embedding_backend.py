from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from vector_gateway.backends.embedding_local import LocalEmbeddingBackend
from vector_gateway.config import EmbeddingConfig, EmbeddingModelConfig


def _fake_torch(*, cuda: bool, mps: bool):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )


class LocalEmbeddingBackendDeviceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LocalEmbeddingBackend(
            EmbeddingConfig(device="auto"),
            {
                "default": EmbeddingModelConfig(
                    model_name="BAAI/bge-m3",
                    vector_size=1024,
                    device="auto",
                )
            },
        )

    def test_auto_prefers_cuda(self) -> None:
        with patch.dict("sys.modules", {"torch": _fake_torch(cuda=True, mps=False)}):
            _, profile = self.backend.resolve_profile("default")
        self.assertEqual(profile.device, "cuda")

    def test_override_can_force_cpu(self) -> None:
        with patch.dict("sys.modules", {"torch": _fake_torch(cuda=True, mps=False)}):
            _, profile = self.backend.resolve_profile("default", "cpu")
        self.assertEqual(profile.device, "cpu")

    def test_unavailable_device_falls_back_to_cpu(self) -> None:
        with patch.dict("sys.modules", {"torch": _fake_torch(cuda=False, mps=False)}):
            _, profile = self.backend.resolve_profile("default", "cuda")
        self.assertEqual(profile.device, "cpu")


if __name__ == "__main__":
    unittest.main()
