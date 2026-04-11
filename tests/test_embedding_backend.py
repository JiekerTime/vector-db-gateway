from __future__ import annotations

import asyncio
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

    def test_registered_underlying_model_name_resolves_to_alias(self) -> None:
        with patch.dict("sys.modules", {"torch": _fake_torch(cuda=True, mps=False)}):
            profile_name, profile = self.backend.resolve_profile("BAAI/bge-m3")
        self.assertEqual(profile_name, "default@cuda")
        self.assertEqual(profile.model_name, "BAAI/bge-m3")

    def test_unavailable_device_falls_back_to_cpu(self) -> None:
        with patch.dict("sys.modules", {"torch": _fake_torch(cuda=False, mps=False)}):
            _, profile = self.backend.resolve_profile("default", "cuda")
        self.assertEqual(profile.device, "cpu")

    def test_warmup_uses_configured_profiles(self) -> None:
        warmed_profiles: list[tuple[str, str]] = []

        class _FakeModel:
            def encode(self, texts, **kwargs):
                return [[0.0] for _ in texts]

        backend = LocalEmbeddingBackend(
            EmbeddingConfig(
                device="auto",
                warmup_enabled=True,
                warmup_models=["default"],
                warmup_devices=["cpu"],
                warmup_probe_texts=["warmup probe"],
            ),
            {
                "default": EmbeddingModelConfig(
                    model_name="BAAI/bge-m3",
                    vector_size=1024,
                    device="auto",
                )
            },
        )

        def _fake_get_or_load(profile_name, profile):
            warmed_profiles.append((profile_name, profile.device or "unknown"))
            return _FakeModel()

        with patch.object(backend, "_get_or_load_model", side_effect=_fake_get_or_load):
            warmed = asyncio.run(backend.warmup())

        self.assertEqual(warmed_profiles, [("default@cpu", "cpu")])
        self.assertEqual(warmed[0]["profile"], "default@cpu")
        self.assertEqual(warmed[0]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
