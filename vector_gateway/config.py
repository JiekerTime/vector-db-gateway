"""Configuration models and loader."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    backend: str = "sentence_transformers"
    default_model: str = "BAAI/bge-m3"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 64


class EmbeddingModelConfig(BaseModel):
    backend: str = "sentence_transformers"
    model_name: str
    vector_size: int | None = None
    distance: str = "Cosine"
    normalize_embeddings: bool | None = None
    device: str | None = None


class QdrantConfig(BaseModel):
    url: str = "http://qdrant:6333"
    timeout: int = 20


class QueueConfig(BaseModel):
    max_batch_size: int = 8
    max_wait_ms: int = 15
    max_concurrent_jobs: int = 1
    preferred_device: str | None = None


class RoutingRule(BaseModel):
    caller_pattern: str
    queue: str
    service_priority: int = 1
    operation: str = "search"


class FairnessConfig(BaseModel):
    aging_step_ms: int = 5000
    max_consecutive_realtime_batches: int = 8
    reserve_batch_share: float = 0.10


class CollectionConfig(BaseModel):
    vector_size: int
    distance: str = "Cosine"
    owner: str = "default"
    vector_name: str | None = None
    model: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class GatewayConfig(BaseModel):
    port: int = 8526
    api_key: str = "change-me"
    log_level: str = "INFO"
    log_dir: str = "logs"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    models: dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    queues: dict[str, QueueConfig]
    routing_rules: list[RoutingRule]
    operation_priority: dict[str, int]
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    collections: dict[str, CollectionConfig]


def load_config(path: str = "config.yaml") -> GatewayConfig:
    """Read the YAML config file and return validated settings."""
    config_path = Path(os.environ.get("VECTOR_GATEWAY_CONFIG", path))
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if "models" not in raw:
        raw["models"] = {}
    return GatewayConfig.model_validate(raw)
