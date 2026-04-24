from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

_QUERY_TYPE_MAPPING = {
    "ASR": "ASR",
    "AUDIO": "ASR",
    "OCR": "OCR",
    "TEXT": "OCR",
    "VISUAL": "Visual",
    "VISION": "Visual",
    "MIXED": "Visual",
    "MULTIMODAL": "Visual",
    "RAG": "Visual",
}


def coalesce_query_value(payload: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def normalize_query_type(raw_query_type: str) -> str:
    return _QUERY_TYPE_MAPPING.get((raw_query_type or "").strip().upper(), "Visual")


def is_query_dict(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    keys = set(item.keys())
    signals = {
        "id",
        "query_text",
        "query",
        "query_type",
        "time_hint",
        "why_this_query",
    }
    return bool(keys & signals)


def extract_query_list(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    for key in ("retrieval_queries", "queries"):
        value = payload.get(key)
        if isinstance(value, list) and all(is_query_dict(x) for x in value):
            return value

    for value in payload.values():
        if isinstance(value, list) and value and all(is_query_dict(x) for x in value):
            return value
    return None


def normalize_query_input(query: Any) -> dict[str, Any]:
    query_dict: dict[str, Any] | None = None
    if isinstance(query, dict):
        query_dict = dict(query)
    elif is_dataclass(query):
        obj = asdict(query)
        if isinstance(obj, dict):
            query_dict = obj
    else:
        required = ("id", "time_hint", "query_type", "query_text", "why_this_query")
        if all(hasattr(query, key) for key in required):
            query_dict = {key: getattr(query, key) for key in required}
            if hasattr(query, "extra_fields") and isinstance(getattr(query, "extra_fields"), dict):
                query_dict.update(getattr(query, "extra_fields"))

    if query_dict is None:
        raise TypeError("Each query must be a dict or a RetrievalQuery-like object.")

    if isinstance(query_dict.get("extra_fields"), dict):
        merged = dict(query_dict["extra_fields"])
        merged.update({k: v for k, v in query_dict.items() if k != "extra_fields"})
        query_dict = merged

    query_id = str(coalesce_query_value(query_dict, ["id", "query_id"], default="")).strip()
    query_text = str(
        coalesce_query_value(
            query_dict,
            ["query_text", "query", "text", "retrieval_query", "query_content"],
            default="",
        )
    ).strip()
    time_hint = str(coalesce_query_value(query_dict, ["time_hint", "time", "phase_hint"], default="")).strip()
    query_type = normalize_query_type(
        str(coalesce_query_value(query_dict, ["query_type", "type", "route"], default="Visual"))
    )
    why_this_query = str(
        coalesce_query_value(
            query_dict,
            ["why_this_query", "reason", "rationale", "why"],
            default="",
        )
    ).strip()

    normalized = dict(query_dict)
    normalized["id"] = query_id
    normalized["query_text"] = query_text
    normalized["query_type"] = query_type
    normalized["time_hint"] = time_hint
    normalized["why_this_query"] = why_this_query
    return normalized
