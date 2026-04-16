"""Local embedding backend backed by sentence-transformers."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import time

from vector_gateway.config import EmbeddingConfig, EmbeddingModelConfig

logger = logging.getLogger(__name__)


class LocalEmbeddingBackend:
    """Load sentence-transformers models lazily and serve batched inference."""

    def __init__(self, config: EmbeddingConfig, model_registry: dict[str, EmbeddingModelConfig]):
        self._config = config
        self._model_registry = model_registry
        self._models: dict[str, object] = {}
        self._model_devices: dict[str, str] = {}
        self._model_last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        if self._config.cpu_threads:
            thread_count = str(max(1, self._config.cpu_threads))
            os.environ.setdefault("OMP_NUM_THREADS", thread_count)
            os.environ.setdefault("MKL_NUM_THREADS", thread_count)

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
            "warmup": {
                "enabled": self._config.warmup_enabled,
                "models": self._config.warmup_models or sorted(self._model_registry.keys()),
                "devices": self._config.warmup_devices,
            },
            "idle_unload": {
                "seconds": self._config.idle_unload_seconds,
                "devices": self._config.idle_unload_devices,
            },
        }

    async def warmup(self) -> list[dict[str, str]]:
        if not self._config.warmup_enabled:
            return []

        configured_models = self._config.warmup_models or sorted(self._model_registry.keys())
        if not configured_models:
            configured_models = ["default"]
        configured_devices = self._config.warmup_devices or ["auto"]
        probe_texts = self._config.warmup_probe_texts or ["warmup"]

        warmed: list[dict[str, str]] = []
        seen_profiles: set[str] = set()
        for model_name in configured_models:
            for requested_device in configured_devices:
                profile_name, profile = self.resolve_profile(model_name, requested_device)
                if profile_name in seen_profiles:
                    continue
                await asyncio.to_thread(
                    self._warmup_sync,
                    probe_texts,
                    profile_name,
                    profile,
                )
                seen_profiles.add(profile_name)
                warmed.append(
                    {
                        "profile": profile_name,
                        "model": profile.model_name,
                        "device": profile.device or self._config.device,
                    }
                )
        return warmed

    def resolve_profile(
        self,
        model_name: str | None,
        device: str | None = None,
    ) -> tuple[str, EmbeddingModelConfig]:
        if model_name and model_name in self._model_registry:
            profile = self._model_registry[model_name]
            profile_name = model_name
        elif model_name:
            alias = self._profile_alias_for_model_name(model_name)
            if alias is not None:
                profile_name = alias
                profile = self._model_registry[alias]
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

    def _profile_alias_for_model_name(self, model_name: str) -> str | None:
        for alias, profile in self._model_registry.items():
            if profile.model_name == model_name:
                return alias
        return None

    def available_devices(self) -> list[str]:
        devices = ["cpu"]
        try:
            import torch
        except (ImportError, OSError):
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

    def _configure_runtime(self, profile: EmbeddingModelConfig) -> None:
        if (profile.device or self._config.device) != "cpu":
            return
        try:
            import torch
        except (ImportError, OSError):
            return
        if self._config.cpu_threads:
            torch.set_num_threads(max(1, self._config.cpu_threads))
        if self._config.cpu_interop_threads:
            try:
                torch.set_num_interop_threads(max(1, self._config.cpu_interop_threads))
            except RuntimeError:
                pass

    def _embed_sync(
        self,
        texts: list[str],
        profile_name: str,
        profile: EmbeddingModelConfig,
    ) -> list[list[float]]:
        try:
            return self._encode_sync(texts, profile_name, profile)
        except Exception as exc:
            fallback = self._fallback_profile(profile_name, profile)
            if fallback is None:
                raise
            fallback_name, fallback_profile = fallback
            logger.warning(
                "Embedding failed on %s for %s, retrying on cpu: %s",
                profile.device,
                profile.model_name,
                exc,
            )
            return self._encode_sync(texts, fallback_name, fallback_profile)

    def _warmup_sync(
        self,
        texts: list[str],
        profile_name: str,
        profile: EmbeddingModelConfig,
    ) -> None:
        try:
            self._encode_sync(texts, profile_name, profile, batch_size=1)
        except Exception as exc:
            fallback = self._fallback_profile(profile_name, profile)
            if fallback is None:
                raise
            fallback_name, fallback_profile = fallback
            logger.warning(
                "Warmup failed on %s for %s, retrying on cpu: %s",
                profile.device,
                profile.model_name,
                exc,
            )
            self._encode_sync(texts, fallback_name, fallback_profile, batch_size=1)

    def _encode_sync(
        self,
        texts: list[str],
        profile_name: str,
        profile: EmbeddingModelConfig,
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        self._configure_runtime(profile)
        model = self._get_or_load_model(profile_name, profile)
        vectors = model.encode(
            texts,
            batch_size=batch_size or self._config.batch_size,
            normalize_embeddings=(
                self._config.normalize_embeddings
                if profile.normalize_embeddings is None
                else profile.normalize_embeddings
            ),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._touch_model(profile_name)
        return vectors.tolist()

    def unload_idle_models(
        self,
        idle_seconds: int | None = None,
        *,
        devices: set[str] | None = None,
    ) -> list[str]:
        max_idle = idle_seconds if idle_seconds is not None else self._config.idle_unload_seconds
        if not max_idle or max_idle <= 0:
            return []

        allowed_devices = {device.lower() for device in (devices or self._config.idle_unload_devices or [])}
        now = time.monotonic()
        unloaded: list[str] = []
        released_cuda = False

        with self._lock:
            for profile_name in list(self._models.keys()):
                last_used = self._model_last_used.get(profile_name, now)
                if now - last_used < max_idle:
                    continue
                device = (self._model_devices.get(profile_name) or "").lower()
                if allowed_devices and device not in allowed_devices:
                    continue
                self._models.pop(profile_name, None)
                self._model_devices.pop(profile_name, None)
                self._model_last_used.pop(profile_name, None)
                unloaded.append(profile_name)
                if device == "cuda":
                    released_cuda = True

        if not unloaded:
            return []

        gc.collect()
        if released_cuda:
            try:
                import torch
            except (ImportError, OSError):
                return unloaded
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.exception("Failed to clear CUDA cache after unloading idle profiles")
        return unloaded

    def _fallback_profile(
        self,
        profile_name: str,
        profile: EmbeddingModelConfig,
    ) -> tuple[str, EmbeddingModelConfig] | None:
        current_device = (profile.device or self._config.device or "auto").lower()
        if current_device == "cpu":
            return None
        if "cpu" not in self.available_devices():
            return None
        base_name = profile_name.split("@", 1)[0]
        fallback_profile = profile.model_copy(update={"device": "cpu"})
        return f"{base_name}@cpu", fallback_profile

    def _get_or_load_model(self, profile_name: str, profile: EmbeddingModelConfig):
        with self._lock:
            cached = self._models.get(profile_name)
            if cached is not None:
                self._model_last_used[profile_name] = time.monotonic()
                return cached

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the local embedding backend"
                ) from exc

            logger.info("Loading embedding model: %s device=%s", profile.model_name, profile.device)
            try:
                model = SentenceTransformer(
                    profile.model_name,
                    device=profile.device or self._config.device,
                )
            except Exception as exc:
                logger.warning(
                    "Primary model load failed for %s on %s, retrying with local cache only: %s",
                    profile.model_name,
                    profile.device,
                    exc,
                )
                model = SentenceTransformer(
                    profile.model_name,
                    device=profile.device or self._config.device,
                    local_files_only=True,
                )
            self._models[profile_name] = model
            self._model_devices[profile_name] = profile.device or self._config.device
            self._model_last_used[profile_name] = time.monotonic()
            return model

    def _touch_model(self, profile_name: str) -> None:
        with self._lock:
            if profile_name in self._models:
                self._model_last_used[profile_name] = time.monotonic()
