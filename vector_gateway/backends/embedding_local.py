"""Local embedding backend backed by sentence-transformers."""

from __future__ import annotations

import asyncio
import logging
import threading

from vector_gateway.config import EmbeddingConfig, EmbeddingModelConfig

logger = logging.getLogger(__name__)


class LocalEmbeddingBackend:
    """Load sentence-transformers models lazily and serve batched inference."""

    def __init__(self, config: EmbeddingConfig, model_registry: dict[str, EmbeddingModelConfig]):
        self._config = config
        self._model_registry = model_registry
        self._models: dict[str, object] = {}
        self._model_devices: dict[str, str] = {}
        self._lock = threading.Lock()

    async def embed_texts(
        self,
        texts: list[str],
        model_name: str | None = None,
        device: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        profile_name, profile = self.resolve_profile(model_name, device)
        return await asyncio.to_thread(self._embed_sync, texts, profile_name, profile)

    def status(self) -> dict[str, object]:
        return {
            "backend": self._config.backend,
            "default_model": self._config.default_model,
            "device": self._config.device,
            "available_devices": self.available_devices(),
            "default_runtime_device": self._resolve_device(self._config.device),
            "normalize_embeddings": self._config.normalize_embeddings,
            "registered_models": sorted(self._model_registry.keys()),
            "loaded_models": {
                name: self._model_devices.get(name, "unknown")
                for name in sorted(self._models.keys())
            },
        }

    def resolve_profile(
        self,
        model_name: str | None,
        device: str | None = None,
    ) -> tuple[str, EmbeddingModelConfig]:
        if model_name and model_name in self._model_registry:
            profile = self._model_registry[model_name]
            profile_name = model_name
        elif model_name is None and "default" in self._model_registry:
            profile_name = "default"
            profile = self._model_registry["default"]
        else:
            profile_name = model_name or "runtime"
            profile = EmbeddingModelConfig(
                backend=self._config.backend,
                model_name=model_name or self._config.default_model,
                vector_size=None,
                distance="Cosine",
                normalize_embeddings=self._config.normalize_embeddings,
                device=self._config.device,
            )

        resolved_device = self._resolve_device(device or profile.device or self._config.device)
        runtime_profile_name = f"{profile_name}@{resolved_device}"
        runtime_profile = profile.model_copy(update={"device": resolved_device})
        return runtime_profile_name, runtime_profile

    def available_devices(self) -> list[str]:
        devices = ["cpu"]
        try:
            import torch
        except ImportError:
            return devices

        if torch.cuda.is_available():
            devices.insert(0, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            insert_at = 1 if devices and devices[0] == "cuda" else 0
            devices.insert(insert_at, "mps")
        return devices

    def _resolve_device(self, preferred: str | None) -> str:
        target = (preferred or "auto").lower()
        available = set(self.available_devices())
        if target == "auto":
            for candidate in ("cuda", "mps", "cpu"):
                if candidate in available:
                    return candidate
            return "cpu"
        if target in available:
            return target
        logger.warning("Requested device '%s' is unavailable, falling back to cpu", target)
        return "cpu"

    def _embed_sync(
        self,
        texts: list[str],
        profile_name: str,
        profile: EmbeddingModelConfig,
    ) -> list[list[float]]:
        model = self._get_or_load_model(profile_name, profile)
        vectors = model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=(
                self._config.normalize_embeddings
                if profile.normalize_embeddings is None
                else profile.normalize_embeddings
            ),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def _get_or_load_model(self, profile_name: str, profile: EmbeddingModelConfig):
        with self._lock:
            cached = self._models.get(profile_name)
            if cached is not None:
                return cached

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the local embedding backend"
                ) from exc

            logger.info("Loading embedding model: %s device=%s", profile.model_name, profile.device)
            model = SentenceTransformer(
                profile.model_name,
                device=profile.device or self._config.device,
            )
            self._models[profile_name] = model
            self._model_devices[profile_name] = profile.device or self._config.device
            return model
