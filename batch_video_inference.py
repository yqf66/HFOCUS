#!/usr/bin/env python3
"""批量视频两阶段推理框架（基于 base_Omni + focus_localizer + evidence_judge_pipeline）。"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch

from base_Omni import (
    DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH,
    DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH,
    DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
    ModelRegistry,
    _build_localizer_config,
    _get_or_load_evidence_judge_vl_runtime,
    _is_same_model_path,
    _release_omni_runtime,
    _resolve_harm_rules_text,
    initialize_model_registry,
    run_global_understanding_and_query_extraction,
    run_query_evidence_judge,
)
from evidence_judge_pipeline import _load_judge_runtime

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]


ROMAN_CATEGORIES = ("I", "II", "III", "IV")
C_TO_GT_CATEGORY = {
    "C1": "I",
    "C2": "II",
    "C3": "III",
    "C4": "IV",
}


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"JSONL 解析失败: {path} line={line_no}, err={exc}") from exc
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _append_jsonl_line(fp: Any, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _iter_latest_index_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(index_path)
    dedup: dict[int, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        idx = _safe_int(row.get("index"))
        if idx is None:
            passthrough.append(row)
            continue
        dedup[idx] = row
    ordered = [dedup[idx] for idx in sorted(dedup)]
    return ordered + passthrough


def _overwrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for row in rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")


def _filter_rows_by_indices(rows: list[dict[str, Any]], indices: set[int]) -> list[dict[str, Any]]:
    filtered: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = _safe_int(row.get("index"))
        if idx is None or idx not in indices:
            continue
        filtered[idx] = row
    return [filtered[idx] for idx in sorted(filtered)]


def _iter_with_progress(items: list[dict[str, Any]], *, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, total=len(items), dynamic_ncols=True, leave=True)


def _run_with_optional_capture(verbose: bool, log_path: Path, fn, *args, **kwargs):
    if verbose:
        return fn(*args, **kwargs)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("\n" + "=" * 80 + "\n")
        with redirect_stdout(lf):
            return fn(*args, **kwargs)


def _roman_sort_key(cat: str) -> int:
    order = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return order.get(cat, 999)


def _normalize_roman_category(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip().upper()
        if not text:
            continue

        match = re.match(r"^(IV|III|II|I)(?:\b|[\s\.:,_-])", text)
        if match:
            return match.group(1)

        match = re.search(r"\b(IV|III|II|I)\b", text)
        if match:
            return match.group(1)
    return ""


def _extract_dataset_label(video_path: str) -> str:
    text = str(video_path or "")
    match = re.search(r"(?:^|[\\/])(C[1-5])(?:[\\/]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(C[1-5])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def _derive_gt_from_path(video_path: str) -> dict[str, Any]:
    dataset_label = _extract_dataset_label(video_path)

    if dataset_label in C_TO_GT_CATEGORY:
        return {
            "dataset_label": dataset_label,
            "binary": "Harmful",
            "category": C_TO_GT_CATEGORY[dataset_label],
        }

    if dataset_label == "C5":
        return {
            "dataset_label": dataset_label,
            "binary": "Harmless",
            "category": "",
        }

    return {
        "dataset_label": dataset_label,
        "binary": "Unknown",
        "category": "",
    }


def _extract_pred_applicable_categories(judge_result: dict[str, Any]) -> list[str]:
    judgements = judge_result.get("category_judgements")
    if not isinstance(judgements, list):
        return []

    cats: set[str] = set()
    for item in judgements:
        if not isinstance(item, dict):
            continue
        applicability = str(item.get("applicability", "")).strip()
        if applicability != "<适用>":
            continue
        cat = _normalize_roman_category(item.get("category_id"), item.get("category_name"))
        if cat in ROMAN_CATEGORIES:
            cats.add(cat)

    return sorted(cats, key=_roman_sort_key)


def _extract_pred_binary(judge_result: dict[str, Any], pred_categories: list[str]) -> str:
    text = str(judge_result.get("judge_normalized", "") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        if lines[0] == "<Harmful>":
            return "Harmful"
        if lines[0] == "<Harmless>":
            return "Harmless"

    lowered = text.lower()
    if "<harmful>" in lowered:
        return "Harmful"
    if "<harmless>" in lowered:
        return "Harmless"

    if pred_categories:
        return "Harmful"
    if isinstance(judge_result.get("category_judgements"), list):
        return "Harmless"

    return "Unknown"


def _release_whisper_runtime(registry: ModelRegistry | None) -> None:
    if registry is None or registry.whisper is None:
        return

    runtime = registry.whisper
    registry.whisper = None
    try:
        model = runtime.model
        del model
    except Exception:
        pass
    del runtime

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _release_query_runtime(registry: ModelRegistry | None) -> None:
    if registry is None or registry.query is None:
        return

    runtime = registry.query
    registry.query = None
    try:
        model = runtime.model
        tokenizer = runtime.tokenizer
        del model
        del tokenizer
    except Exception:
        pass
    del runtime

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    payloads: list[Any]

    if suffix == ".jsonl":
        payloads = []
        for line_no, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payloads.append(json.loads(text))
            except Exception as exc:
                raise ValueError(f"dataset jsonl 解析失败 line={line_no}: {exc}") from exc
    else:
        raw = _read_json(dataset_path)
        if isinstance(raw, list):
            payloads = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("data"), list):
                payloads = raw["data"]
            elif isinstance(raw.get("items"), list):
                payloads = raw["items"]
            else:
                payloads = [raw]
        else:
            raise ValueError("dataset 文件必须是 JSON 对象 / JSON 数组 / JSONL")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(payloads, start=1):
        if not isinstance(item, dict):
            continue
        video_path = str(item.get("file", "") or "").strip()
        if not video_path:
            continue

        gt = _derive_gt_from_path(video_path)
        records.append(
            {
                "index": idx,
                "video_path": video_path,
                "gt": gt,
                "raw_item": item,
            }
        )

    return records


def _iter_stage1_index(stage1_index_path: Path) -> list[dict[str, Any]]:
    if not stage1_index_path.exists():
        raise FileNotFoundError(f"stage1 索引文件不存在: {stage1_index_path}")
    return _iter_latest_index_rows(stage1_index_path)


def _extract_stage1_payload_counts(detail: dict[str, Any]) -> tuple[int, int] | None:
    queries_payload = detail.get("queries_payload")
    localization_payload = detail.get("localization_payload")
    report_text = str(detail.get("report_text", "") or "")

    if not isinstance(queries_payload, dict):
        return None
    if not isinstance(localization_payload, dict):
        return None
    if not report_text.strip():
        return None

    retrieval_queries = queries_payload.get("retrieval_queries")
    evidence_results = localization_payload.get("evidence_results")
    if not isinstance(retrieval_queries, list) or not retrieval_queries:
        return None
    if not isinstance(evidence_results, list) or not evidence_results:
        return None

    return len(retrieval_queries), len(evidence_results)


def _load_valid_stage1_sample(
    sample_path: Path,
    *,
    expected_index: int | None = None,
    expected_video_path: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not sample_path.exists():
        return None, f"stage1 文件不存在: {sample_path}"

    try:
        detail = _read_json(sample_path)
    except Exception as exc:
        return None, f"stage1 文件无法解析: {sample_path}, err={exc}"
    if not isinstance(detail, dict):
        return None, f"stage1 文件格式错误: {sample_path}"

    if expected_index is not None:
        parsed_idx = _safe_int(detail.get("index"))
        if parsed_idx is not None and parsed_idx != expected_index:
            return None, f"stage1 index 不一致: expect={expected_index}, got={parsed_idx}"

    if expected_video_path is not None:
        detail_video = str(detail.get("video_path", "") or "")
        if detail_video != expected_video_path:
            return None, f"stage1 video_path 不一致: expect={expected_video_path}, got={detail_video}"

    counts = _extract_stage1_payload_counts(detail)
    if counts is None:
        return None, f"stage1 关键字段缺失或为空: {sample_path}"

    query_result = detail.get("query_result")
    parse_error = False
    parse_error_message = ""
    if isinstance(query_result, dict):
        parse_error = bool(query_result.get("parse_error", False))
        parse_error_message = str(query_result.get("parse_error_message", "") or "")

    return {
        "detail": detail,
        "query_count": counts[0],
        "evidence_count": counts[1],
        "query_parse_error": parse_error,
        "query_parse_error_message": parse_error_message,
    }, ""


def _recover_stage1_rows_from_samples(stage1_dir: Path, stage1_log_dir: Path) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    if not stage1_dir.exists():
        return recovered

    for sample_path in sorted(stage1_dir.glob("*.json")):
        stem_idx = _safe_int(sample_path.stem)
        loaded, _ = _load_valid_stage1_sample(sample_path, expected_index=stem_idx)
        if loaded is None:
            continue

        detail = loaded["detail"]
        if not isinstance(detail, dict):
            continue
        index = _safe_int(detail.get("index")) or stem_idx
        if index is None:
            continue
        video_path = str(detail.get("video_path", "") or "")
        gt = detail.get("gt") if isinstance(detail.get("gt"), dict) else {}
        sample_log_path = stage1_log_dir / f"{index:06d}.log"

        recovered.append(
            {
                "index": index,
                "video_path": video_path,
                "gt": gt,
                "status": "ok",
                "stage1_path": str(sample_path),
                "sample_log_path": str(sample_log_path),
                "query_count": int(loaded["query_count"]),
                "evidence_count": int(loaded["evidence_count"]),
                "query_parse_error": bool(loaded["query_parse_error"]),
                "query_parse_error_message": str(loaded["query_parse_error_message"]),
                "recovered_from_samples": True,
            }
        )

    return recovered


def _merge_stage1_rows(
    index_rows: list[dict[str, Any]],
    recovered_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in index_rows:
        idx = _safe_int(row.get("index"))
        if idx is None:
            continue
        merged[idx] = row

    for row in recovered_rows:
        idx = _safe_int(row.get("index"))
        if idx is None:
            continue
        existing = merged.get(idx)
        existing_status = str(existing.get("status", "") or "") if isinstance(existing, dict) else ""
        if existing_status != "ok":
            merged[idx] = row
            continue

        stage1_path_text = str(existing.get("stage1_path", "") or "")
        existing_path = Path(stage1_path_text).expanduser() if stage1_path_text else None
        loaded, _ = (
            _load_valid_stage1_sample(existing_path, expected_index=idx)
            if existing_path is not None
            else (None, "missing")
        )
        if loaded is None:
            merged[idx] = row

    return [merged[idx] for idx in sorted(merged)]


def _iter_stage2_index(stage2_index_path: Path) -> list[dict[str, Any]]:
    if not stage2_index_path.exists():
        raise FileNotFoundError(f"stage2 索引文件不存在: {stage2_index_path}")
    return _iter_latest_index_rows(stage2_index_path)


def _load_valid_stage2_sample(
    sample_path: Path,
    *,
    expected_index: int | None = None,
    expected_video_path: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not sample_path.exists():
        return None, f"stage2 文件不存在: {sample_path}"
    try:
        detail = _read_json(sample_path)
    except Exception as exc:
        return None, f"stage2 文件无法解析: {sample_path}, err={exc}"
    if not isinstance(detail, dict):
        return None, f"stage2 文件格式错误: {sample_path}"

    if expected_index is not None:
        parsed_idx = _safe_int(detail.get("index"))
        if parsed_idx is not None and parsed_idx != expected_index:
            return None, f"stage2 index 不一致: expect={expected_index}, got={parsed_idx}"

    if expected_video_path is not None:
        detail_video = str(detail.get("video_path", "") or "")
        if detail_video != expected_video_path:
            return None, f"stage2 video_path 不一致: expect={expected_video_path}, got={detail_video}"

    pred_binary = str(detail.get("pred_binary", "") or "")
    pred_categories = detail.get("pred_categories")
    judge_result = detail.get("judge_result")
    if pred_binary not in {"Harmful", "Harmless", "Unknown"}:
        return None, f"stage2 pred_binary 非法: {pred_binary}"
    if not isinstance(pred_categories, list):
        return None, "stage2 pred_categories 缺失或格式错误"
    if not isinstance(judge_result, dict):
        return None, "stage2 judge_result 缺失或格式错误"
    return detail, ""


def run_stage1(args: argparse.Namespace, records: list[dict[str, Any]]) -> ModelRegistry | None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    stage1_dir = out_dir / "stage1_samples"
    stage1_index_path = out_dir / "stage1_index.jsonl"
    stage1_meta_path = out_dir / "stage1_meta.json"
    stage1_log_dir = out_dir / "logs" / "stage1"
    stage1_error_path = out_dir / "stage1_errors.jsonl"

    localizer_cfg = _build_localizer_config(
        blip_model=args.localizer_blip_model,
        device=args.localizer_device,
        batch_size=args.localizer_batch_size,
        config_json=args.localizer_config_json,
        whisper_model_path=args.localizer_whisper_model,
        whisper_device=args.localizer_whisper_device,
    )

    registry = _run_with_optional_capture(
        args.verbose,
        stage1_log_dir / "_runtime_init.log",
        initialize_model_registry,
        omni_model_path=args.omni_model,
        omni_device=args.omni_device,
        omni_device_map=args.omni_device_map,
        load_query=True,
        query_model_path=args.query_model,
        query_device=args.query_device,
        load_whisper=True,
        whisper_model_path=args.localizer_whisper_model,
        whisper_device=args.localizer_whisper_device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, Any]] = []
    existing_by_index: dict[int, dict[str, Any]] = {}
    if args.resume:
        if stage1_index_path.exists():
            existing_rows = _iter_latest_index_rows(stage1_index_path)
        recovered_rows = _recover_stage1_rows_from_samples(stage1_dir, stage1_log_dir)
        merged_rows = _merge_stage1_rows(existing_rows, recovered_rows)
        for row in merged_rows:
            idx = _safe_int(row.get("index"))
            if idx is None:
                continue
            existing_by_index[idx] = row
        if existing_by_index:
            print(f"[Stage1] resume 已启用，检测到可复用样本: {len(existing_by_index)}")

    ok_count = 0
    err_count = 0
    resumed_count = 0
    with stage1_index_path.open("a", encoding="utf-8") as wf:
        with stage1_error_path.open("a", encoding="utf-8") as ef:
            progress_items = _iter_with_progress(records, desc="Stage1")
            for rank, sample in enumerate(progress_items, start=1):
                video_path = str(sample["video_path"])
                index = int(sample["index"])
                gt = dict(sample.get("gt", {}))

                if tqdm is None:
                    print("\n" + "=" * 72)
                    print(f"[Stage1 {rank}/{len(records)}] {video_path}")
                    print("=" * 72)

                record_line: dict[str, Any] = {
                    "index": index,
                    "video_path": video_path,
                    "gt": gt,
                }

                if args.resume:
                    cached_row = existing_by_index.get(index)
                    sample_path: Path | None = None
                    sample_log_path: Path | None = None
                    if isinstance(cached_row, dict):
                        sample_path_text = str(cached_row.get("stage1_path", "") or "")
                        sample_log_text = str(cached_row.get("sample_log_path", "") or "")
                        if sample_path_text:
                            sample_path = Path(sample_path_text).expanduser()
                        if sample_log_text:
                            sample_log_path = Path(sample_log_text).expanduser()
                    if sample_path is None:
                        sample_path = stage1_dir / f"{index:06d}.json"
                    if sample_log_path is None:
                        sample_log_path = stage1_log_dir / f"{index:06d}.log"

                    loaded, _ = _load_valid_stage1_sample(
                        sample_path,
                        expected_index=index,
                        expected_video_path=video_path,
                    )
                    if loaded is not None:
                        reused_line = {
                            "index": index,
                            "video_path": video_path,
                            "gt": gt,
                            "status": "ok",
                            "stage1_path": str(sample_path),
                            "sample_log_path": str(sample_log_path),
                            "query_count": int(loaded["query_count"]),
                            "evidence_count": int(loaded["evidence_count"]),
                            "query_parse_error": bool(loaded["query_parse_error"]),
                            "query_parse_error_message": str(loaded["query_parse_error_message"] or ""),
                            "resumed_skip": True,
                        }
                        ok_count += 1
                        resumed_count += 1
                        _append_jsonl_line(wf, reused_line)
                        if tqdm is not None:
                            progress_items.set_postfix_str(
                                f"ok={ok_count} err={err_count} resumed={resumed_count}"
                            )
                        continue

                try:
                    sample_log_path = stage1_log_dir / f"{index:06d}.log"
                    result = _run_with_optional_capture(
                        args.verbose,
                        sample_log_path,
                        run_global_understanding_and_query_extraction,
                        video_path=video_path,
                        user_focus=args.focus,
                        omni_model_path=args.omni_model,
                        query_model_path=args.query_model,
                        omni_device=args.omni_device,
                        omni_device_map=args.omni_device_map,
                        query_device=args.query_device,
                        omni_max_new_tokens=args.omni_max_new_tokens,
                        query_max_new_tokens=args.query_max_new_tokens,
                        video_fps=args.video_fps,
                        video_min_frames=args.video_min_frames,
                        video_max_frames=args.video_max_frames,
                        video_nframes=args.video_nframes,
                        video_start=args.video_start,
                        video_end=args.video_end,
                        segment_seconds=args.segment_seconds,
                        segment_overlap=args.segment_overlap,
                        merge_batch_size=args.merge_batch_size,
                        merge_max_new_tokens=args.merge_max_new_tokens,
                        merge_max_continuations=args.merge_max_continuations,
                        run_localizer=True,
                        run_evidence_judge=False,
                        localizer_config=localizer_cfg,
                        localizer_whisper_model_path=args.localizer_whisper_model,
                        localizer_whisper_device=args.localizer_whisper_device,
                        model_registry=registry,
                    )

                    query_dict = result.query_result.to_dict()
                    queries_payload = {
                        "retrieval_queries": list(query_dict.get("retrieval_queries", [])),
                    }
                    localization_payload = {
                        "evidence_results": list(result.evidence_results or []),
                    }

                    if not queries_payload["retrieval_queries"]:
                        raise RuntimeError("query 提炼为空，无法进入第二阶段。")
                    if not localization_payload["evidence_results"]:
                        raise RuntimeError("localizer 结果为空，无法进入第二阶段。")

                    detail = {
                        "index": index,
                        "video_path": video_path,
                        "gt": gt,
                        "report_text": result.report_text,
                        "query_result": query_dict,
                        "queries_payload": queries_payload,
                        "localization_payload": localization_payload,
                        "raw_item": sample.get("raw_item"),
                    }

                    sample_path = stage1_dir / f"{index:06d}.json"
                    _safe_json_dump(sample_path, detail)

                    record_line.update(
                        {
                            "status": "ok",
                            "stage1_path": str(sample_path),
                            "sample_log_path": str(sample_log_path),
                            "query_count": len(queries_payload["retrieval_queries"]),
                            "evidence_count": len(localization_payload["evidence_results"]),
                            "query_parse_error": bool(query_dict.get("parse_error", False)),
                            "query_parse_error_message": str(query_dict.get("parse_error_message", "") or ""),
                        }
                    )
                    ok_count += 1
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} resumed={resumed_count}"
                        )
                except Exception as exc:
                    err_count += 1
                    record_line.update(
                        {
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} resumed={resumed_count}"
                        )
                        tqdm.write(f"[Stage1-Error] idx={index} video={video_path} err={exc}")
                    else:
                        print(f"[Stage1-Error] {video_path}\n{exc}")
                    _append_jsonl_line(ef, record_line)

                _append_jsonl_line(wf, record_line)

    current_indices = {int(sample["index"]) for sample in records}
    normalized_rows = _filter_rows_by_indices(_iter_latest_index_rows(stage1_index_path), current_indices)
    _overwrite_jsonl(stage1_index_path, normalized_rows)

    stage1_meta = {
        "total": len(records),
        "ok": ok_count,
        "error": err_count,
        "resumed_skip": resumed_count,
        "stage1_index": str(stage1_index_path),
        "stage1_samples_dir": str(stage1_dir),
    }
    _safe_json_dump(stage1_meta_path, stage1_meta)

    print("\n" + "=" * 72)
    print("Stage1 完成")
    print(json.dumps(stage1_meta, ensure_ascii=False, indent=2))

    return registry


def run_stage2(args: argparse.Namespace, registry: ModelRegistry | None = None) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    stage1_index_path = out_dir / "stage1_index.jsonl"
    stage1_dir = out_dir / "stage1_samples"
    stage2_dir = out_dir / "stage2_samples"
    stage2_index_path = out_dir / "stage2_index.jsonl"
    stage2_meta_path = out_dir / "stage2_meta.json"
    stage2_log_dir = out_dir / "logs" / "stage2"
    stage2_error_path = out_dir / "stage2_errors.jsonl"

    stage1_rows: list[dict[str, Any]] = []
    if stage1_index_path.exists():
        stage1_rows = _iter_stage1_index(stage1_index_path)
    if args.resume:
        recovered_rows = _recover_stage1_rows_from_samples(stage1_dir, out_dir / "logs" / "stage1")
        stage1_rows = _merge_stage1_rows(stage1_rows, recovered_rows)
    if not stage1_rows:
        raise RuntimeError(
            f"未找到可用于 Stage2 的 stage1 数据，请先运行 stage1。index={stage1_index_path}, dir={stage1_dir}"
        )

    existing_stage2_by_index: dict[int, dict[str, Any]] = {}
    if args.resume and stage2_index_path.exists():
        existing_stage2_rows = _iter_stage2_index(stage2_index_path)
        for row in existing_stage2_rows:
            idx = _safe_int(row.get("index"))
            if idx is None:
                continue
            existing_stage2_by_index[idx] = row
        if existing_stage2_by_index:
            print(f"[Stage2] resume 已启用，检测到已有 stage2 记录: {len(existing_stage2_by_index)}")

    if registry is None:
        registry = ModelRegistry()

    print("\n" + "=" * 72)
    print("Stage2 准备：释放 Omni/Whisper 并加载 Judge 模型")
    print("=" * 72)

    _run_with_optional_capture(args.verbose, stage2_log_dir / "_runtime_release.log", _release_omni_runtime, registry)
    _run_with_optional_capture(args.verbose, stage2_log_dir / "_runtime_release.log", _release_whisper_runtime, registry)

    same_query_judge_model = _is_same_model_path(args.query_model, args.evidence_judge_model)
    if not same_query_judge_model:
        _release_query_runtime(registry)

    harm_rules_text = _resolve_harm_rules_text(
        harm_rules_text=args.harm_rules_text,
        harm_rules_txt_path=args.harm_rules_txt,
    )

    vl_runtime = _run_with_optional_capture(
        args.verbose,
        stage2_log_dir / "_vl_runtime.log",
        _get_or_load_evidence_judge_vl_runtime,
        registry=registry,
        model_path=args.evidence_judge_vl_model,
        device=args.evidence_judge_vl_device,
    )

    judge_runtime: Any
    if same_query_judge_model and registry.query is not None:
        print("[Stage2] 复用 Stage1 已加载的 Query 模型作为最终审判模型")
        judge_runtime = registry.query
    else:
        print("[Stage2] 加载独立最终审判模型")
        judge_runtime = _run_with_optional_capture(
            args.verbose,
            stage2_log_dir / "_judge_runtime.log",
            _load_judge_runtime,
            model_path=args.evidence_judge_model,
            device=args.evidence_judge_device,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    err_count = 0
    skip_count = 0
    resumed_count = 0
    with stage2_index_path.open("a", encoding="utf-8") as wf:
        with stage2_error_path.open("a", encoding="utf-8") as ef:
            progress_items = _iter_with_progress(stage1_rows, desc="Stage2")
            for rank, row in enumerate(progress_items, start=1):
                index = int(row.get("index", rank))
                video_path = str(row.get("video_path", "") or "")
                gt = row.get("gt") if isinstance(row.get("gt"), dict) else {}

                if tqdm is None:
                    print("\n" + "=" * 72)
                    print(f"[Stage2 {rank}/{len(stage1_rows)}] {video_path}")
                    print("=" * 72)

                out_record: dict[str, Any] = {
                    "index": index,
                    "video_path": video_path,
                    "gt": gt,
                    "stage1_status": str(row.get("status", "")),
                }

                if args.resume:
                    cached_row = existing_stage2_by_index.get(index)
                    if isinstance(cached_row, dict) and str(cached_row.get("status", "") or "") == "ok":
                        cached_path_text = str(cached_row.get("stage2_path", "") or "")
                        cached_sample_path = (
                            Path(cached_path_text).expanduser()
                            if cached_path_text
                            else (stage2_dir / f"{index:06d}.json")
                        )
                        loaded_stage2, _ = _load_valid_stage2_sample(
                            cached_sample_path,
                            expected_index=index,
                            expected_video_path=video_path,
                        )
                        if loaded_stage2 is not None:
                            pred_binary = str(loaded_stage2.get("pred_binary", "Unknown") or "Unknown")
                            pred_categories = (
                                loaded_stage2.get("pred_categories")
                                if isinstance(loaded_stage2.get("pred_categories"), list)
                                else []
                            )
                            gt_binary = str(gt.get("binary", "") or "")
                            gt_category = str(gt.get("category", "") or "")
                            binary_correct = gt_binary in {"Harmful", "Harmless"} and pred_binary == gt_binary
                            harmful_hit_correct = False
                            if gt_binary == "Harmful" and gt_category in ROMAN_CATEGORIES:
                                harmful_hit_correct = gt_category in set(pred_categories)

                            sample_log_path = stage2_log_dir / f"{index:06d}.log"
                            out_record.update(
                                {
                                    "status": "ok",
                                    "stage2_path": str(cached_sample_path),
                                    "sample_log_path": str(sample_log_path),
                                    "pred_binary": pred_binary,
                                    "pred_categories": pred_categories,
                                    "binary_correct": bool(binary_correct),
                                    "harmful_category_hit_correct": bool(harmful_hit_correct),
                                    "resumed_skip": True,
                                }
                            )
                            ok_count += 1
                            resumed_count += 1
                            _append_jsonl_line(wf, out_record)
                            if tqdm is not None:
                                progress_items.set_postfix_str(
                                    f"ok={ok_count} err={err_count} skip={skip_count} resumed={resumed_count}"
                                )
                            continue

                if str(row.get("status", "")) != "ok":
                    out_record.update(
                        {
                            "status": "skipped",
                            "error": "stage1_failed",
                            "pred_binary": "Unknown",
                            "pred_categories": [],
                            "binary_correct": False,
                            "harmful_category_hit_correct": False,
                        }
                    )
                    skip_count += 1
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} skip={skip_count} resumed={resumed_count}"
                        )
                    _append_jsonl_line(wf, out_record)
                    continue

                stage1_path = Path(str(row.get("stage1_path", "") or "")).expanduser()
                if not stage1_path.exists():
                    out_record.update(
                        {
                            "status": "error",
                            "error": f"stage1 详情文件不存在: {stage1_path}",
                            "pred_binary": "Unknown",
                            "pred_categories": [],
                            "binary_correct": False,
                            "harmful_category_hit_correct": False,
                        }
                    )
                    err_count += 1
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} skip={skip_count} resumed={resumed_count}"
                        )
                    _append_jsonl_line(ef, out_record)
                    _append_jsonl_line(wf, out_record)
                    continue

                try:
                    detail = _read_json(stage1_path)
                    queries_payload = detail.get("queries_payload")
                    localization_payload = detail.get("localization_payload")
                    report_text = str(detail.get("report_text", "") or "")

                    if not isinstance(queries_payload, dict):
                        raise ValueError("stage1 queries_payload 缺失或格式错误")
                    if not isinstance(localization_payload, dict):
                        raise ValueError("stage1 localization_payload 缺失或格式错误")
                    if not report_text.strip():
                        raise ValueError("stage1 report_text 为空")

                    sample_log_path = stage2_log_dir / f"{index:06d}.log"
                    judge_result = _run_with_optional_capture(
                        args.verbose,
                        sample_log_path,
                        run_query_evidence_judge,
                        video_path=video_path,
                        retrieval_queries=queries_payload.get("retrieval_queries", []),
                        evidence_results=localization_payload.get("evidence_results", []),
                        report_text=report_text,
                        harm_rules_text=harm_rules_text,
                        qwen3_vl_model_path=args.evidence_judge_vl_model,
                        qwen3_vl_device=args.evidence_judge_vl_device,
                        qwen3_vl_max_new_tokens=args.evidence_judge_vl_max_new_tokens,
                        max_visual_frames=args.evidence_judge_max_visual_frames,
                        judge_model_path=args.evidence_judge_model,
                        judge_device=args.evidence_judge_device,
                        judge_max_new_tokens=args.evidence_judge_max_new_tokens,
                        vl_runtime=vl_runtime,
                        judge_runtime=judge_runtime,
                    )

                    pred_categories = _extract_pred_applicable_categories(judge_result)
                    pred_binary = _extract_pred_binary(judge_result, pred_categories)

                    gt_binary = str(gt.get("binary", "") or "")
                    gt_category = str(gt.get("category", "") or "")

                    binary_correct = gt_binary in {"Harmful", "Harmless"} and pred_binary == gt_binary
                    harmful_hit_correct = False
                    if gt_binary == "Harmful" and gt_category in ROMAN_CATEGORIES:
                        harmful_hit_correct = gt_category in set(pred_categories)

                    sample_path = stage2_dir / f"{index:06d}.json"
                    stage2_detail = {
                        "index": index,
                        "video_path": video_path,
                        "gt": gt,
                        "pred_binary": pred_binary,
                        "pred_categories": pred_categories,
                        "judge_result": judge_result,
                    }
                    _safe_json_dump(sample_path, stage2_detail)

                    out_record.update(
                        {
                            "status": "ok",
                            "stage2_path": str(sample_path),
                            "sample_log_path": str(sample_log_path),
                            "pred_binary": pred_binary,
                            "pred_categories": pred_categories,
                            "binary_correct": bool(binary_correct),
                            "harmful_category_hit_correct": bool(harmful_hit_correct),
                        }
                    )
                    ok_count += 1
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} skip={skip_count} resumed={resumed_count}"
                        )
                except Exception as exc:
                    out_record.update(
                        {
                            "status": "error",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "pred_binary": "Unknown",
                            "pred_categories": [],
                            "binary_correct": False,
                            "harmful_category_hit_correct": False,
                        }
                    )
                    err_count += 1
                    if tqdm is not None:
                        progress_items.set_postfix_str(
                            f"ok={ok_count} err={err_count} skip={skip_count} resumed={resumed_count}"
                        )
                        tqdm.write(f"[Stage2-Error] idx={index} video={video_path} err={exc}")
                    else:
                        print(f"[Stage2-Error] {video_path}\n{exc}")
                    _append_jsonl_line(ef, out_record)

                _append_jsonl_line(wf, out_record)

    stage1_indices = {
        idx
        for idx in (_safe_int(row.get("index")) for row in stage1_rows)
        if idx is not None
    }
    normalized_rows = _filter_rows_by_indices(_iter_latest_index_rows(stage2_index_path), stage1_indices)
    _overwrite_jsonl(stage2_index_path, normalized_rows)

    stage2_meta = {
        "total": len(stage1_rows),
        "ok": ok_count,
        "error": err_count,
        "skipped": skip_count,
        "resumed_skip": resumed_count,
        "stage2_index": str(stage2_index_path),
        "stage2_samples_dir": str(stage2_dir),
    }
    _safe_json_dump(stage2_meta_path, stage2_meta)

    print("\n" + "=" * 72)
    print("Stage2 完成")
    print(json.dumps(stage2_meta, ensure_ascii=False, indent=2))


def evaluate_from_stage2(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    stage2_index_path = out_dir / "stage2_index.jsonl"
    metrics_path = out_dir / "metrics.json"
    metrics_txt_path = out_dir / "metrics_report.txt"

    rows = _iter_stage2_index(stage2_index_path)

    total_samples = 0
    known_gt_samples = 0

    binary_total = 0
    binary_correct = 0

    harmful_total = 0
    harmful_hit_correct = 0

    category_stats: dict[str, dict[str, int]] = {
        cat: {"total": 0, "correct": 0, "gt_positive": 0, "pred_positive": 0} for cat in ROMAN_CATEGORIES
    }

    stage_status_counts: dict[str, int] = {}
    unknown_gt_count = 0

    for row in rows:
        total_samples += 1

        status = str(row.get("status", "") or "")
        stage_status_counts[status] = stage_status_counts.get(status, 0) + 1

        gt = row.get("gt") if isinstance(row.get("gt"), dict) else {}
        gt_binary = str(gt.get("binary", "") or "")
        gt_category = str(gt.get("category", "") or "")

        if gt_binary not in {"Harmful", "Harmless"}:
            unknown_gt_count += 1
            continue

        known_gt_samples += 1

        pred_binary = str(row.get("pred_binary", "Unknown") or "Unknown")
        pred_categories = row.get("pred_categories") if isinstance(row.get("pred_categories"), list) else []
        pred_set = {c for c in pred_categories if c in ROMAN_CATEGORIES}

        binary_total += 1
        is_binary_correct = pred_binary == gt_binary
        if is_binary_correct:
            binary_correct += 1

        if gt_binary == "Harmful" and gt_category in ROMAN_CATEGORIES:
            harmful_total += 1
            if gt_category in pred_set:
                harmful_hit_correct += 1

        for cat in ROMAN_CATEGORIES:
            cat_stat = category_stats[cat]
            cat_stat["total"] += 1

            gt_applicable = gt_category == cat
            pred_applicable = cat in pred_set

            if gt_applicable:
                cat_stat["gt_positive"] += 1
            if pred_applicable:
                cat_stat["pred_positive"] += 1
            if gt_applicable == pred_applicable:
                cat_stat["correct"] += 1

    metrics = {
        "total_samples": total_samples,
        "known_gt_samples": known_gt_samples,
        "unknown_gt_samples": unknown_gt_count,
        "stage_status_counts": stage_status_counts,
        "binary": {
            "correct": binary_correct,
            "total": binary_total,
            "accuracy": (binary_correct / binary_total) if binary_total else None,
        },
        "harmful_category_hit": {
            "correct": harmful_hit_correct,
            "total": harmful_total,
            "accuracy": (harmful_hit_correct / harmful_total) if harmful_total else None,
            "rule": "仅在 GT 为 C1~C4 时统计，若预测适用类别包含 GT 类别（允许多选）即判正确",
        },
        "per_category_accuracy": {
            cat: {
                "correct": stat["correct"],
                "total": stat["total"],
                "accuracy": (stat["correct"] / stat["total"]) if stat["total"] else None,
                "gt_positive": stat["gt_positive"],
                "pred_positive": stat["pred_positive"],
            }
            for cat, stat in category_stats.items()
        },
    }

    _safe_json_dump(metrics_path, metrics)

    lines: list[str] = []
    lines.append("Batch Video Inference Metrics")
    lines.append("=" * 48)
    lines.append(f"total_samples: {total_samples}")
    lines.append(f"known_gt_samples: {known_gt_samples}")
    lines.append(f"unknown_gt_samples: {unknown_gt_count}")
    lines.append(f"stage_status_counts: {json.dumps(stage_status_counts, ensure_ascii=False)}")
    lines.append("")

    binary_acc = metrics["binary"]["accuracy"]
    lines.append("[Binary Accuracy]")
    if isinstance(binary_acc, float):
        lines.append(
            f"correct={metrics['binary']['correct']} total={metrics['binary']['total']} "
            f"acc={binary_acc:.4f}"
        )
    else:
        lines.append(
            f"correct={metrics['binary']['correct']} total={metrics['binary']['total']} acc=N/A"
        )
    lines.append("")

    hit_acc = metrics["harmful_category_hit"]["accuracy"]
    lines.append("[Harmful Category Hit Accuracy]")
    if isinstance(hit_acc, float):
        lines.append(
            f"correct={metrics['harmful_category_hit']['correct']} "
            f"total={metrics['harmful_category_hit']['total']} acc={hit_acc:.4f}"
        )
    else:
        lines.append("acc=N/A")
    lines.append("")

    lines.append("[Per-Category Accuracy: I/II/III/IV]")
    for cat in ROMAN_CATEGORIES:
        stat = metrics["per_category_accuracy"][cat]
        acc = stat["accuracy"]
        acc_text = f"{acc:.4f}" if isinstance(acc, float) else "N/A"
        lines.append(
            f"{cat}: correct={stat['correct']} total={stat['total']} acc={acc_text} "
            f"gt_positive={stat['gt_positive']} pred_positive={stat['pred_positive']}"
        )

    metrics_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("评估完成")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"metrics_json: {metrics_path}")
    print(f"metrics_report: {metrics_txt_path}")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量视频两阶段推理与评估框架")
    parser.add_argument("--stage", choices=["all", "stage1", "stage2", "eval"], default="all")
    parser.add_argument("--dataset_json", type=str, default="", help="数据集标注 JSON/JSONL 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="批量输出目录")

    parser.add_argument("--focus", type=str, default="", help="全局理解附加关注点")

    parser.add_argument("--omni_model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni")
    parser.add_argument("--omni_device", type=str, default=None)
    parser.add_argument("--omni_device_map", type=str, default="auto")
    parser.add_argument("--omni_max_new_tokens", type=int, default=1024)

    parser.add_argument("--query_model", type=str, default=DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH)
    parser.add_argument("--query_device", type=str, default=None)
    parser.add_argument("--query_max_new_tokens", type=int, default=1024)

    parser.add_argument("--video_fps", type=float, default=4.0)
    parser.add_argument("--video_min_frames", type=int, default=32)
    parser.add_argument("--video_max_frames", type=int, default=384)
    parser.add_argument("--video_nframes", type=int, default=None)
    parser.add_argument("--video_start", type=float, default=0.0)
    parser.add_argument("--video_end", type=float, default=None)
    parser.add_argument("--segment_seconds", type=float, default=20.0)
    parser.add_argument("--segment_overlap", type=float, default=2.0)
    parser.add_argument("--merge_batch_size", type=int, default=6)
    parser.add_argument("--merge_max_new_tokens", type=int, default=1024)
    parser.add_argument("--merge_max_continuations", type=int, default=3)

    parser.add_argument("--localizer_blip_model", type=str, default="large")
    parser.add_argument("--localizer_device", type=str, default=None)
    parser.add_argument("--localizer_batch_size", type=int, default=32)
    parser.add_argument("--localizer_config_json", type=str, default=None)
    parser.add_argument("--localizer_whisper_model", type=str, default=DEFAULT_LOCALIZER_WHISPER_MODEL_PATH)
    parser.add_argument("--localizer_whisper_device", type=str, default=None)

    parser.add_argument("--harm_rules_txt", type=str, default=None)
    parser.add_argument("--harm_rules_text", type=str, default="")

    parser.add_argument("--evidence_judge_vl_model", type=str, default=DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH)
    parser.add_argument("--evidence_judge_vl_device", type=str, default=None)
    parser.add_argument("--evidence_judge_vl_max_new_tokens", type=int, default=768)
    parser.add_argument("--evidence_judge_max_visual_frames", type=int, default=6)

    parser.add_argument("--evidence_judge_model", type=str, default=DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH)
    parser.add_argument("--evidence_judge_device", type=str, default=None)
    parser.add_argument("--evidence_judge_max_new_tokens", type=int, default=1024)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="启用断点续跑（默认开启）")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="关闭断点续跑，强制重跑")
    parser.add_argument("--verbose", action="store_true", help="打印底层模型详细日志（默认关闭，仅显示基本流程+进度条）")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.verbose:
        # 默认安静模式：压低第三方库日志与常见非关键 warning。
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("librosa").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa(\.|$)")

    if args.stage in {"all", "stage1"} and not str(args.dataset_json or "").strip():
        parser.error("stage=all/stage1 时必须提供 --dataset_json")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    registry: ModelRegistry | None = None

    if args.stage in {"all", "stage1"}:
        dataset_path = Path(args.dataset_json).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset 文件不存在: {dataset_path}")

        records = _load_dataset_records(dataset_path)
        if not records:
            raise RuntimeError(f"dataset 中未解析到有效 file 路径: {dataset_path}")

        dataset_preview_path = out_dir / "dataset_preview.json"
        _safe_json_dump(
            dataset_preview_path,
            {
                "dataset_path": str(dataset_path),
                "count": len(records),
                "first_5": records[:5],
            },
        )
        print(f"dataset_count={len(records)} preview_saved={dataset_preview_path}")

        registry = run_stage1(args, records)

    if args.stage in {"all", "stage2"}:
        run_stage2(args, registry=registry)

    if args.stage in {"all", "eval"}:
        evaluate_from_stage2(args)


if __name__ == "__main__":
    # 降低部分环境下 tokenizers 并行告警对日志的干扰。
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
