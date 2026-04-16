"""Metadata-prefix helpers for contextual chunk text enrichment."""

from __future__ import annotations

from typing import Any

from vector_gateway.config import MetadataPrefixConfig


def render_metadata_prefix(payload: dict[str, Any], policy: MetadataPrefixConfig) -> str | None:
    """Render a metadata prefix string from payload fields and policy."""
    if not policy.enabled:
        return None

    parts: list[str] = []
    for item in policy.parts:
        raw_value = payload.get(item.payload_key)
        value = _string_value(raw_value)
        if value is None:
            continue
        if item.label:
            parts.append(f"{item.label}: {value}")
        else:
            parts.append(value)

    if not parts:
        return None
    return f"{policy.prefix}{policy.separator.join(parts)}{policy.suffix}"


def apply_metadata_prefix(
    *,
    text: str | None,
    payload: dict[str, Any],
    policy: MetadataPrefixConfig,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Apply metadata prefix to chunk text and return updated text/payload."""
    updated_payload = dict(payload)
    base_text = _source_text(text, updated_payload, policy.text_payload_key)
    if base_text is None:
        return text, updated_payload, None

    prefix = render_metadata_prefix(updated_payload, policy)
    if prefix is None:
        return base_text, updated_payload, None

    if base_text.startswith(prefix):
        prefixed_text = base_text
    else:
        prefixed_text = f"{prefix}\n{base_text}"

    if policy.raw_text_payload_key and policy.raw_text_payload_key not in updated_payload:
        updated_payload[policy.raw_text_payload_key] = base_text
    updated_payload[policy.prefix_payload_key] = prefix
    updated_payload[policy.text_payload_key] = prefixed_text
    return prefixed_text, updated_payload, prefix


def _source_text(text: str | None, payload: dict[str, Any], text_payload_key: str) -> str | None:
    if isinstance(text, str) and text.strip():
        return text
    from_payload = payload.get(text_payload_key)
    if isinstance(from_payload, str) and from_payload.strip():
        return from_payload
    return None


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        values = [_string_value(item) for item in value]
        compact = [item for item in values if item]
        if compact:
            return ", ".join(compact)
        return None
    return None
