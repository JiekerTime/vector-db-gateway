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
        self._lock = threading.Lock()

    async def embed_texts(self, texts: list[str], model_name: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        profile_name, profile = self.resolve_profile(model_name)
        return await asyncio.to_thread(self._embed_sync, texts, profile_name, profile)

    def status(self) -> dict[str, object]:
        return {
            "backend": self._config.backend,
            "default_model": self._config.default_model,
            "device": self._config.device,
            "normalize_embeddings": self._config.normalize_embeddings,
            "registered_models": sorted(self._model_registry.keys()),
            "loaded_models": sorted(self._models.keys()),
        }

    def resolve_profile(self, model_name: str | None) -> tuple[str, EmbeddingModelConfig]:
        if model_name and model_name in self._model_registry:
            profile = self._model_registry[model_name]
            return model_name, profile

        if model_name is None and "default" in self._model_registry:
            return "default", self._model_registry["default"]

        profile_name = model_name or "runtime"
        return profile_name, EmbeddingModelConfig(
            backend=self._config.backend,
            model_name=model_name or self._config.default_model,
            vector_size=None,
            distance="Cosine",
            normalize_embeddings=self._config.normalize_embeddings,
            device=self._config.device,
        )

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

            logger.info("Loading embedding model: %s", profile.model_name)
            model = SentenceTransformer(
                profile.model_name,
                device=profile.device or self._config.device,
            )
            self._models[profile_name] = model
            return model
