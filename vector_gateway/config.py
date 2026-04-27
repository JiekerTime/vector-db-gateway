"""Configuration models and loader."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    backend: str = "sentence_transformers"
    default_model: str = "BAAI/bge-m3"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 64
    cpu_threads: int | None = None
    cpu_interop_threads: int | None = None
    warmup_enabled: bool = True
    warmup_models: list[str] = Field(default_factory=lambda: ["default"])
    warmup_devices: list[str] = Field(default_factory=lambda: ["auto"])
    warmup_probe_texts: list[str] = Field(default_factory=lambda: ["warmup"])
    idle_unload_seconds: int | None = None
    idle_unload_devices: list[str] = Field(default_factory=lambda: ["cuda"])


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
    sparse_vector_name: str | None = None
    sparse_modifier: str | None = None
    payload_indexes: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class LogicalCollectionMigrationConfig(BaseModel):
    next_target: str | None = None
    scheduler: str | None = None
    job_name: str | None = None


class MetadataPrefixPartConfig(BaseModel):
    payload_key: str
    label: str | None = None


class MetadataPrefixConfig(BaseModel):
    enabled: bool = False
    parts: list[MetadataPrefixPartConfig] = Field(default_factory=list)
    separator: str = " | "
    prefix: str = "["
    suffix: str = "]"
    text_payload_key: str = "text"
    prefix_payload_key: str = "metadata_prefix"
    raw_text_payload_key: str | None = "text_raw"

    @model_validator(mode="after")
    def validate_payload(self) -> "MetadataPrefixConfig":
        if self.enabled and not self.parts:
            raise ValueError("'metadata_prefix.parts' must not be empty when enabled")
        return self


class LogicalCollectionConfig(BaseModel):
    read_targets: list[str] = Field(default_factory=list)
    write_targets: list[str] = Field(default_factory=list)
    default_query_mode: str = "dense"
    alias_name: str | None = None
    query_model: str | None = None
    write_model: str | None = None
    migration: LogicalCollectionMigrationConfig = Field(default_factory=LogicalCollectionMigrationConfig)
    metadata_prefix: MetadataPrefixConfig = Field(default_factory=MetadataPrefixConfig)


class ServiceEndpointConfig(BaseModel):
    url: str
    api_key: str
    timeout: int = 20


class DoMigConfig(BaseModel):
    enabled: bool = False
    queue_channel: str = "migration_queue"
    batch_limit: int = 200


class GatewayConfig(BaseModel):
    port: int = 8526
    api_key: str = "change-me"
    log_level: str = "INFO"
    log_dir: str = "logs"
    state_dir: str = "state"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    models: dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    queues: dict[str, QueueConfig]
    routing_rules: list[RoutingRule]
    operation_priority: dict[str, int]
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    collections: dict[str, CollectionConfig]
    logical_collections: dict[str, LogicalCollectionConfig] = Field(default_factory=dict)
    write_disk: ServiceEndpointConfig | None = None
    db_migrator: ServiceEndpointConfig | None = None
    do_mig: DoMigConfig = Field(default_factory=DoMigConfig)


def load_config(path: str = "config.yaml") -> GatewayConfig:
    """Read the YAML config file and return validated settings."""
    config_path = Path(os.environ.get("VECTOR_GATEWAY_CONFIG", path))
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if "models" not in raw:
        raw["models"] = {}
    return GatewayConfig.model_validate(raw)
