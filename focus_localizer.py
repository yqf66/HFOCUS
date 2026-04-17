"""
Lightweight query-guided local evidence localization built from FOCUS ideas.

This module adapts keyframe-focused FOCUS into a short-video local evidence
localizer with route-aware query handling:
- locate one main evidence segment per query
- return only a few supporting frames inside that segment
- always localize visually first for every supported query route
- currently supports Visual, ASR, and OCR query routes

Public APIs:
- localize_query_evidence(video_path, query, config)
- localize_all_queries(video_path, retrieval_queries, config)
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from query_utils import normalize_query_input

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - optional import guard
    VideoReader = None  # type: ignore[assignment]
    cpu = None  # type: ignore[assignment]


SimilarityFn = Callable[[Any, str, list[int]], list[float]]
_SUPPORTED_QUERY_TYPES = {"Visual", "ASR", "OCR"}


default_config: dict[str, Any] = {
    "arm_seconds": 1.0,
    "coarse_sample_fps": 4.0,
    "fine_sample_fps": 8.0,
    "peak_expand_ratio": 0.7,
    "min_segment_len": 1.0,
    "max_segment_len": 6.0,
    "segment_score_threshold": 0.35,
    "min_frame_gap_sec": 0.3,
    "max_support_frames": 6,
    "asr_sample_rate": 16000,
    "asr_clip_pad_before_sec": 0.30,
    "asr_clip_pad_after_sec": 0.60,
    "asr_clip_min_len": 1.2,
    "asr_clip_max_len": 12.0,
    "asr_keep_audio_clip": False,
    "ocr_max_frames": 4,
}


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(default_config)
    if config:
        merged.update(config)
    return merged


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 1e-8:
        return 0.0
    return inter / union


def _parse_time_hint_anchor(time_hint: str, duration_sec: float) -> float | None:
    if not time_hint:
        return None
    hint = time_hint.lower().split("|", 1)[0].strip()
    phase_to_ratio = {
        "beginning": 0.1,
        "early-middle": 0.3,
        "middle": 0.5,
        "late-middle": 0.7,
        "ending": 0.9,
    }
    if hint not in phase_to_ratio:
        return None
    return phase_to_ratio[hint] * max(0.0, duration_sec)


def _time_hint_prior(center_sec: float, duration_sec: float, time_hint: str) -> float:
    anchor = _parse_time_hint_anchor(time_hint=time_hint, duration_sec=duration_sec)
    if anchor is None or duration_sec <= 1e-8:
        return 0.5
    sigma = max(0.5, 0.2 * duration_sec)
    dist = abs(center_sec - anchor)
    return float(math.exp(-(dist * dist) / (2.0 * sigma * sigma)))


def _enforce_segment_length(
    start_sec: float,
    end_sec: float,
    center_sec: float,
    min_len: float,
    max_len: float,
    duration_sec: float,
) -> tuple[float, float]:
    start = max(0.0, min(start_sec, duration_sec))
    end = max(0.0, min(end_sec, duration_sec))
    if end < start:
        start, end = end, start

    seg_len = end - start
    if seg_len < min_len:
        half = min_len / 2.0
        start = center_sec - half
        end = center_sec + half
    seg_len = end - start
    if seg_len > max_len:
        half = max_len / 2.0
        start = center_sec - half
        end = center_sec + half

    start = max(0.0, start)
    end = min(duration_sec, end)

    if end - start < min_len:
        deficit = min_len - (end - start)
        left_room = start
        right_room = duration_sec - end
        move_left = min(deficit / 2.0, left_room)
        move_right = min(deficit - move_left, right_room)
        start -= move_left
        end += move_right
        remaining = min_len - (end - start)
        if remaining > 0 and left_room - move_left > 0:
            extra_left = min(remaining, start)
            start -= extra_left
            remaining -= extra_left
        if remaining > 0 and right_room - move_right > 0:
            end += min(remaining, duration_sec - end)

    start = max(0.0, min(start, duration_sec))
    end = max(start, min(end, duration_sec))
    return start, end


def _compute_arm_ucb(
    frame_indices: list[int],
    frame_scores: list[float],
    total_frames: int,
    fps: float,
    arm_seconds: float,
) -> tuple[list[int], list[float]]:
    arm_frames = max(1, int(round(max(1e-3, arm_seconds) * fps)))
    arm_count = max(1, int(math.ceil(total_frames / float(arm_frames))))
    arm_values: list[list[float]] = [[] for _ in range(arm_count)]

    point_arm_ids: list[int] = []
    for idx, score in zip(frame_indices, frame_scores):
        arm_id = min(arm_count - 1, max(0, idx // arm_frames))
        point_arm_ids.append(arm_id)
        arm_values[arm_id].append(float(score))

    arm_ucb = [0.0 for _ in range(arm_count)]
    total_samples = len(frame_scores)
    min_var = 1e-6

    for arm_id, values in enumerate(arm_values):
        if not values:
            continue
        n_i = float(len(values))
        mean_i = float(np.mean(values))
        var_i = float(np.var(values)) if len(values) > 1 else 0.0
        var_i = max(var_i, min_var)
        score = mean_i
        if total_samples > 1:
            score += math.sqrt(max(0.0, 2.0 * math.log(total_samples) * var_i / n_i))
            score += 3.0 * math.log(total_samples) / n_i
        arm_ucb[arm_id] = score

    return point_arm_ids, arm_ucb


def _find_local_peaks(values: list[float]) -> list[int]:
    if not values:
        return []
    if len(values) == 1:
        return [0]
    peaks: list[int] = []
    for i, v in enumerate(values):
        left = values[i - 1] if i > 0 else -1e9
        right = values[i + 1] if i < len(values) - 1 else -1e9
        if v >= left and v >= right:
            peaks.append(i)
    if not peaks:
        peaks = [int(np.argmax(values))]
    return peaks


def _build_candidates(
    coarse_points: list[dict[str, Any]],
    cfg: dict[str, Any],
    duration_sec: float,
    time_hint: str,
) -> list[dict[str, Any]]:
    if not coarse_points:
        return []

    values = [float(p["fused_score"]) for p in coarse_points]
    times = [float(p["time_sec"]) for p in coarse_points]
    peaks = _find_local_peaks(values)
    peaks = sorted(peaks, key=lambda i: values[i], reverse=True)

    peak_expand_ratio = float(cfg["peak_expand_ratio"])
    min_seg = float(cfg["min_segment_len"])
    max_seg = float(cfg["max_segment_len"])
    candidates: list[dict[str, Any]] = []

    for peak_idx in peaks:
        peak_value = values[peak_idx]
        stop_value = peak_value * peak_expand_ratio
        left = peak_idx
        right = peak_idx

        while left > 0 and values[left - 1] >= stop_value:
            left -= 1
        while right < len(values) - 1 and values[right + 1] >= stop_value:
            right += 1

        center = times[peak_idx]
        start, end = _enforce_segment_length(
            start_sec=times[left],
            end_sec=times[right],
            center_sec=center,
            min_len=min_seg,
            max_len=max_seg,
            duration_sec=duration_sec,
        )

        mean_score = float(np.mean([p["score"] for p in coarse_points if start <= p["time_sec"] <= end]))
        peak_score = float(coarse_points[peak_idx]["score"])
        seg_len = max(1e-6, end - start)
        compactness = 1.0
        if max_seg > min_seg:
            compactness = 1.0 - _clamp((seg_len - min_seg) / (max_seg - min_seg), 0.0, 1.0)
        expected_points = max(1.0, seg_len * float(cfg["coarse_sample_fps"]))
        point_count = sum(1 for p in coarse_points if start <= p["time_sec"] <= end)
        continuity = _clamp(point_count / expected_points, 0.0, 1.0)
        hint_prior = _time_hint_prior(center_sec=(start + end) / 2.0, duration_sec=duration_sec, time_hint=time_hint)

        segment_score = (
            0.45 * mean_score
            + 0.30 * peak_score
            + 0.15 * compactness
            + 0.05 * continuity
            + 0.05 * hint_prior
        )
        segment_score = float(_clamp(segment_score, 0.0, 1.0))

        candidate = {
            "start_sec": float(start),
            "end_sec": float(end),
            "score": segment_score,
            "peak_idx": int(coarse_points[peak_idx]["frame_idx"]),
        }

        is_duplicate = False
        for kept in candidates:
            if _segment_iou(start, end, kept["start_sec"], kept["end_sec"]) >= 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            candidates.append(candidate)
        if len(candidates) >= 12:
            break

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def _select_supporting_frames(
    frame_indices: list[int],
    frame_scores: list[float],
    fps: float,
    peak_idx: int,
    min_gap_sec: float,
    max_support_frames: int,
) -> list[dict[str, Any]]:
    if not frame_indices or not frame_scores or max_support_frames <= 0:
        return []

    idx_to_score = {int(i): float(s) for i, s in zip(frame_indices, frame_scores)}
    scored = sorted(idx_to_score.items(), key=lambda x: x[1], reverse=True)
    selected: list[tuple[int, float]] = []
    gap_frames = max(0, int(round(min_gap_sec * fps)))

    if peak_idx in idx_to_score:
        selected.append((peak_idx, idx_to_score[peak_idx]))

    for idx, score in scored:
        if len(selected) >= max_support_frames:
            break
        if any(abs(idx - s_idx) < gap_frames for s_idx, _ in selected):
            continue
        selected.append((idx, score))

    if not selected:
        selected = scored[:max_support_frames]

    selected = sorted(selected, key=lambda x: x[0])
    return [
        {
            "frame_idx": int(idx),
            "time_sec": float(idx / max(1e-6, fps)),
            "score": float(score),
        }
        for idx, score in selected[:max_support_frames]
    ]


def _sample_indices(start_idx: int, end_idx: int, step: int, anchors: list[int] | None = None) -> list[int]:
    if end_idx < start_idx:
        return []
    indices = list(range(start_idx, end_idx + 1, max(1, step)))
    if anchors:
        for a in anchors:
            if start_idx <= a <= end_idx:
                indices.append(int(a))
    return sorted(set(indices))


def _extract_audio_segment_to_wav(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
) -> Path:
    start = float(max(0.0, start_sec))
    end = float(max(start + 1e-3, end_sec))
    duration = float(max(1e-3, end - start))

    with tempfile.NamedTemporaryFile(prefix="focus_asr_clip_", suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(max(8000, int(sample_rate))),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        tail = err[-1] if err else "unknown ffmpeg error"
        if wav_path.exists():
            wav_path.unlink()
        raise RuntimeError(f"ffmpeg segment extraction failed: {tail}")
    return wav_path


def _default_asr_infer_placeholder(
    audio_path: str,
    *,
    query: dict[str, Any],
    segment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = query, segment, config
    return {
        "status": "placeholder",
        "text": "",
        "note": "ASR placeholder: provide config['asr_infer_fn'] to run real ASR.",
        "audio_path": audio_path,
    }


def _default_ocr_infer_placeholder(
    video: Any,
    keyframes: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = video, query, config
    return {
        "status": "placeholder",
        "texts": [],
        "note": "OCR placeholder: provide config['ocr_infer_fn'] to run real OCR.",
        "frame_count": len(keyframes),
    }


def _resolve_asr_infer_fn(config: dict[str, Any]) -> Callable[..., Any]:
    fn = config.get("asr_infer_fn")
    if callable(fn):
        return fn
    return _default_asr_infer_placeholder


def _resolve_ocr_infer_fn(config: dict[str, Any]) -> Callable[..., Any]:
    fn = config.get("ocr_infer_fn")
    if callable(fn):
        return fn
    return _default_ocr_infer_placeholder


def _derive_asr_clip_segment(
    visual_result: dict[str, Any],
    cfg: dict[str, Any],
    duration_sec: float,
) -> dict[str, Any] | None:
    main_segment = visual_result.get("main_segment")
    if not isinstance(main_segment, dict):
        return None

    try:
        main_start = float(main_segment.get("start_sec", 0.0))
        main_end = float(main_segment.get("end_sec", main_start))
    except Exception:
        return None
    if main_end < main_start:
        main_start, main_end = main_end, main_start
    main_start = float(_clamp(main_start, 0.0, duration_sec))
    main_end = float(_clamp(main_end, 0.0, duration_sec))
    if main_end <= main_start:
        return None

    support = visual_result.get("supporting_frames")
    support_frames = support if isinstance(support, list) else []
    support_times: list[float] = []
    for item in support_frames:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("time_sec"))
        except Exception:
            continue
        if math.isfinite(t):
            support_times.append(float(_clamp(t, 0.0, duration_sec)))

    raw_start = min(support_times) if support_times else main_start
    raw_end = max(support_times) if support_times else main_end
    raw_start = min(raw_start, main_start)
    raw_end = max(raw_end, main_end)

    pad_before = float(max(0.0, cfg.get("asr_clip_pad_before_sec", 0.30)))
    pad_after = float(max(0.0, cfg.get("asr_clip_pad_after_sec", 0.60)))
    start_sec, end_sec = _enforce_segment_length(
        start_sec=raw_start - pad_before,
        end_sec=raw_end + pad_after,
        center_sec=0.5 * (raw_start + raw_end),
        min_len=float(max(0.1, cfg.get("asr_clip_min_len", 1.2))),
        max_len=float(max(0.2, cfg.get("asr_clip_max_len", 12.0))),
        duration_sec=duration_sec,
    )
    if end_sec <= start_sec:
        return None
    return {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "raw_start_sec": float(raw_start),
        "raw_end_sec": float(raw_end),
        "pad_before_sec": float(pad_before),
        "pad_after_sec": float(pad_after),
    }


def _select_ocr_keyframes(
    visual_result: dict[str, Any],
    cfg: dict[str, Any],
    fps: float,
    total_frames: int,
) -> list[dict[str, Any]]:
    support = visual_result.get("supporting_frames")
    support_frames = support if isinstance(support, list) else []

    normalized: list[dict[str, Any]] = []
    for item in support_frames:
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_idx"))
            time_sec = float(item.get("time_sec", frame_idx / max(1e-6, fps)))
            score = float(item.get("score", 0.0))
        except Exception:
            continue
        if 0 <= frame_idx < total_frames:
            normalized.append(
                {
                    "frame_idx": int(frame_idx),
                    "time_sec": float(time_sec),
                    "score": float(score),
                }
            )

    max_frames = max(1, int(cfg.get("ocr_max_frames", 4)))
    if normalized:
        picked = sorted(normalized, key=lambda x: float(x["score"]), reverse=True)[:max_frames]
        return sorted(picked, key=lambda x: int(x["frame_idx"]))

    main_segment = visual_result.get("main_segment")
    if isinstance(main_segment, dict):
        center = 0.5 * (float(main_segment.get("start_sec", 0.0)) + float(main_segment.get("end_sec", 0.0)))
        frame_idx = int(round(center * max(1e-6, fps)))
        frame_idx = max(0, min(total_frames - 1, frame_idx))
        return [
            {
                "frame_idx": int(frame_idx),
                "time_sec": float(frame_idx / max(1e-6, fps)),
                "score": float(main_segment.get("score", 0.0)),
            }
        ]
    return []


def _run_asr_on_visual_segment(
    video_path: str,
    query_dict: dict[str, Any],
    visual_result: dict[str, Any],
    cfg: dict[str, Any],
    duration_sec: float,
) -> dict[str, Any]:
    if not bool(visual_result.get("evidence_found")):
        return {"status": "skipped", "reason": "visual_evidence_not_found", "segment": None, "output": None}

    segment = _derive_asr_clip_segment(
        visual_result=visual_result,
        cfg=cfg,
        duration_sec=duration_sec,
    )
    if segment is None:
        return {"status": "skipped", "reason": "invalid_visual_segment", "segment": None, "output": None}

    keep_audio = bool(cfg.get("asr_keep_audio_clip", False))
    sample_rate = int(cfg.get("asr_sample_rate", 16000))
    wav_path: Path | None = None
    try:
        wav_path = _extract_audio_segment_to_wav(
            video_path=video_path,
            start_sec=float(segment["start_sec"]),
            end_sec=float(segment["end_sec"]),
            sample_rate=sample_rate,
        )
        infer_fn = _resolve_asr_infer_fn(cfg)
        output = infer_fn(
            str(wav_path),
            query=query_dict,
            segment=segment,
            config=cfg,
        )
        status = "ok"
        if isinstance(output, dict) and str(output.get("status", "")).strip():
            status = str(output.get("status")).strip()
        result = {
            "status": status,
            "reason": "",
            "segment": segment,
            "audio_clip_path": str(wav_path) if keep_audio else "",
            "output": output,
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": str(exc),
            "segment": segment,
            "audio_clip_path": str(wav_path) if (wav_path and keep_audio) else "",
            "output": None,
        }
    finally:
        if wav_path is not None and wav_path.exists() and not keep_audio:
            wav_path.unlink()
    return result


def _run_ocr_on_visual_frames(
    video: Any,
    query_dict: dict[str, Any],
    visual_result: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if not bool(visual_result.get("evidence_found")):
        return {"status": "skipped", "reason": "visual_evidence_not_found", "keyframes": [], "output": None}

    total_frames = int(len(video))
    fps = float(video.get_avg_fps())
    keyframes = _select_ocr_keyframes(
        visual_result=visual_result,
        cfg=cfg,
        fps=fps,
        total_frames=total_frames,
    )
    if not keyframes:
        return {"status": "skipped", "reason": "no_valid_keyframes", "keyframes": [], "output": None}

    infer_fn = _resolve_ocr_infer_fn(cfg)
    try:
        output = infer_fn(
            video,
            keyframes,
            query=query_dict,
            config=cfg,
        )
        status = "ok"
        if isinstance(output, dict) and str(output.get("status", "")).strip():
            status = str(output.get("status")).strip()
        return {
            "status": status,
            "reason": "",
            "keyframes": keyframes,
            "output": output,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "keyframes": keyframes,
            "output": None,
        }


def _resolve_similarity_fn(config: dict[str, Any]) -> SimilarityFn:
    similarity_fn = config.get("similarity_fn")
    if callable(similarity_fn):
        return similarity_fn

    backend = str(config.get("backend", "hf_blip_itm")).lower()
    if backend not in {"hf_blip_itm", "blip", "transformers_blip"}:
        raise ValueError(
            "Unsupported backend. Set config['backend']='hf_blip_itm' (or 'blip') or provide "
            "a callable config['similarity_fn']."
        )

    model_id = _resolve_hf_blip_model_id(config)
    device = str(config.get("device", ""))
    batch_size = int(config.get("batch_size", 32))
    scorer = _get_hf_blip_scorer(model_id=model_id, device=device, batch_size=batch_size)
    return scorer.score_frames


def _resolve_hf_blip_model_id(config: dict[str, Any]) -> str:
    explicit = str(config.get("hf_blip_model_id", "")).strip()
    if explicit:
        return explicit
    raw = str(config.get("blip_model", "large")).strip()
    mapping = {
        "base": "Salesforce/blip-itm-base-coco",
        "large": "Salesforce/blip-itm-large-coco",
    }
    return mapping.get(raw.lower(), raw)


class _HFBLIPITMScorer:
    def __init__(self, model_id: str, device: str, batch_size: int):
        try:
            import torch
            from PIL import Image
            from transformers import BlipForImageTextRetrieval, BlipProcessor
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Transformers BLIP backend requires torch, pillow, and transformers. "
                "Please install dependencies or provide config['similarity_fn']."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.batch_size = max(1, int(batch_size))
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def _to_numpy_image(self, frame: Any) -> np.ndarray:
        if hasattr(frame, "asnumpy") and callable(frame.asnumpy):
            image = frame.asnumpy()
        elif hasattr(frame, "detach") and callable(frame.detach):
            tensor = frame.detach()
            if hasattr(tensor, "cpu") and callable(tensor.cpu):
                tensor = tensor.cpu()
            image = tensor.numpy() if hasattr(tensor, "numpy") and callable(tensor.numpy) else np.asarray(tensor)
        elif hasattr(frame, "numpy") and callable(frame.numpy):
            image = frame.numpy()
        else:
            image = np.asarray(frame)

        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size > 0:
                if float(np.max(image)) <= 1.0:
                    image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _fetch_batch_frames(self, video: Any, batch_indices: list[int]) -> list[np.ndarray]:
        if not batch_indices:
            return []

        if hasattr(video, "get_batch") and callable(video.get_batch):
            try:
                batch = video.get_batch(batch_indices)
                if hasattr(batch, "asnumpy") and callable(batch.asnumpy):
                    batch_np = batch.asnumpy()
                else:
                    batch_np = np.asarray(batch)
                if isinstance(batch_np, np.ndarray) and batch_np.ndim >= 4:
                    return [self._to_numpy_image(batch_np[j]) for j in range(len(batch_indices))]
            except Exception:
                pass

        frames: list[np.ndarray] = []
        for idx in batch_indices:
            frames.append(self._to_numpy_image(video[idx]))
        return frames

    def score_frames(self, video: Any, query_text: str, frame_indices: list[int]) -> list[float]:
        if not frame_indices:
            return []

        similarities: list[float] = []

        for i in range(0, len(frame_indices), self.batch_size):
            batch_indices = frame_indices[i : i + self.batch_size]
            batch_images = []
            raw_frames = self._fetch_batch_frames(video=video, batch_indices=batch_indices)
            for raw_image in raw_frames:
                pil_img = self._image_cls.fromarray(raw_image)
                batch_images.append(pil_img)

            if not batch_images:
                continue

            text_batch = [query_text for _ in range(len(batch_images))]
            inputs = self.processor(
                images=batch_images,
                text=text_batch,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                try:
                    outputs = self.model(**inputs, use_itm_head=True)
                except TypeError:
                    outputs = self.model(**inputs)
                logits = getattr(outputs, "itm_score", None)
                if logits is None:
                    logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise RuntimeError("BLIP ITM model output does not contain logits/itm_score.")
                probs = self._torch.nn.functional.softmax(logits, dim=-1)
                similarities.extend(float(probs[j, 1].item()) for j in range(len(batch_indices)))

        return similarities


@lru_cache(maxsize=4)
def _get_hf_blip_scorer(model_id: str, device: str, batch_size: int) -> _HFBLIPITMScorer:
    return _HFBLIPITMScorer(model_id=model_id, device=device, batch_size=batch_size)


def _empty_result(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(query.get("id", "")),
        "query_text": str(query.get("query_text", "")),
        "query_type": str(query.get("query_type", "")),
        "time_hint": str(query.get("time_hint", "")),
        "evidence_found": False,
        "main_segment": None,
        "supporting_frames": [],
        "skip_reason": "",
    }


def _normalize_query(query: Any) -> dict[str, Any]:
    return normalize_query_input(query)


def _is_supported_query_type(query_type: str) -> bool:
    return (query_type or "").strip() in _SUPPORTED_QUERY_TYPES


def _unsupported_query_type_reason(query_type: str) -> str:
    current = (query_type or "").strip() or "Unknown"
    supported = ", ".join(sorted(_SUPPORTED_QUERY_TYPES))
    return f"query_type='{current}' is not supported yet; current localizer supports: {supported}."


def _localize_visual_query(
    video: Any,
    query_dict: dict[str, Any],
    cfg: dict[str, Any],
    similarity_fn: SimilarityFn,
) -> dict[str, Any]:
    result = _empty_result(query_dict)
    query_text = str(query_dict.get("query_text", "")).strip()
    if not query_text:
        return result

    total_frames = int(len(video))
    fps = float(video.get_avg_fps())
    duration_sec = float(total_frames / max(1e-6, fps))
    if total_frames <= 0 or duration_sec <= 0:
        return result

    coarse_step = max(1, int(round(fps / max(1e-6, float(cfg["coarse_sample_fps"])))))
    coarse_indices = _sample_indices(0, total_frames - 1, coarse_step, anchors=[0, total_frames - 1])
    coarse_scores = similarity_fn(video, query_text, coarse_indices)
    if not coarse_scores:
        return result

    point_arm_ids, arm_ucb = _compute_arm_ucb(
        frame_indices=coarse_indices,
        frame_scores=coarse_scores,
        total_frames=total_frames,
        fps=fps,
        arm_seconds=float(cfg["arm_seconds"]),
    )
    ucb_min = min(arm_ucb) if arm_ucb else 0.0
    ucb_max = max(arm_ucb) if arm_ucb else 1.0
    denom = max(1e-8, ucb_max - ucb_min)

    coarse_points: list[dict[str, Any]] = []
    for idx, score, arm_id in zip(coarse_indices, coarse_scores, point_arm_ids):
        arm_boost = (arm_ucb[arm_id] - ucb_min) / denom if arm_ucb else 0.0
        fused = 0.7 * float(score) + 0.3 * float(arm_boost)
        coarse_points.append(
            {
                "frame_idx": int(idx),
                "time_sec": float(idx / max(1e-6, fps)),
                "score": float(score),
                "fused_score": float(_clamp(fused, 0.0, 1.0)),
                "arm_id": int(arm_id),
            }
        )

    candidates = _build_candidates(
        coarse_points=coarse_points,
        cfg=cfg,
        duration_sec=duration_sec,
        time_hint=str(query_dict.get("time_hint", "")),
    )
    if not candidates:
        return result

    best = candidates[0]
    if float(best["score"]) < float(cfg["segment_score_threshold"]):
        return result

    start_idx = max(0, int(math.floor(float(best["start_sec"]) * fps)))
    end_idx = min(total_frames - 1, int(math.ceil(float(best["end_sec"]) * fps)))
    fine_step = max(1, int(round(fps / max(1e-6, float(cfg["fine_sample_fps"])))))
    fine_indices = _sample_indices(start_idx, end_idx, fine_step, anchors=[int(best["peak_idx"])])
    fine_scores = similarity_fn(video, query_text, fine_indices)

    if fine_scores:
        fine_mean = float(np.mean(fine_scores))
        fine_peak = float(np.max(fine_scores))
        refined_score = 0.6 * float(best["score"]) + 0.25 * fine_mean + 0.15 * fine_peak
        best_score = float(_clamp(refined_score, 0.0, 1.0))
    else:
        best_score = float(best["score"])

    if best_score < float(cfg["segment_score_threshold"]):
        return result

    support = _select_supporting_frames(
        frame_indices=fine_indices,
        frame_scores=fine_scores,
        fps=fps,
        peak_idx=int(best["peak_idx"]),
        min_gap_sec=float(cfg["min_frame_gap_sec"]),
        max_support_frames=int(cfg["max_support_frames"]),
    )

    result["evidence_found"] = True
    result["main_segment"] = {
        "start_sec": float(best["start_sec"]),
        "end_sec": float(best["end_sec"]),
        "score": float(best_score),
    }
    result["supporting_frames"] = support
    return result


def _localize_asr_query(
    video: Any,
    video_path: str,
    query_dict: dict[str, Any],
    cfg: dict[str, Any],
    similarity_fn: SimilarityFn,
) -> dict[str, Any]:
    visual_result = _localize_visual_query(
        video=video,
        query_dict=query_dict,
        cfg=cfg,
        similarity_fn=similarity_fn,
    )
    if not bool(visual_result.get("evidence_found")):
        if not str(visual_result.get("skip_reason", "")).strip():
            visual_result["skip_reason"] = "Visual localization failed for ASR route."
        return visual_result

    fps = float(video.get_avg_fps())
    total_frames = int(len(video))
    duration_sec = float(total_frames / max(1e-6, fps))
    asr_payload = _run_asr_on_visual_segment(
        video_path=video_path,
        query_dict=query_dict,
        visual_result=visual_result,
        cfg=cfg,
        duration_sec=duration_sec,
    )
    visual_result["asr_segment"] = asr_payload.get("segment")
    visual_result["asr_result"] = asr_payload.get("output")
    visual_result["asr_status"] = str(asr_payload.get("status", "unknown"))
    visual_result["asr_reason"] = str(asr_payload.get("reason", "")).strip()
    audio_clip_path = str(asr_payload.get("audio_clip_path", "")).strip()
    if audio_clip_path:
        visual_result["asr_audio_clip_path"] = audio_clip_path
    return visual_result


def _localize_ocr_query(
    video: Any,
    query_dict: dict[str, Any],
    cfg: dict[str, Any],
    similarity_fn: SimilarityFn,
) -> dict[str, Any]:
    visual_result = _localize_visual_query(
        video=video,
        query_dict=query_dict,
        cfg=cfg,
        similarity_fn=similarity_fn,
    )
    if not bool(visual_result.get("evidence_found")):
        if not str(visual_result.get("skip_reason", "")).strip():
            visual_result["skip_reason"] = "Visual localization failed for OCR route."
        return visual_result

    ocr_payload = _run_ocr_on_visual_frames(
        video=video,
        query_dict=query_dict,
        visual_result=visual_result,
        cfg=cfg,
    )
    visual_result["ocr_keyframes"] = ocr_payload.get("keyframes", [])
    visual_result["ocr_result"] = ocr_payload.get("output")
    visual_result["ocr_status"] = str(ocr_payload.get("status", "unknown"))
    visual_result["ocr_reason"] = str(ocr_payload.get("reason", "")).strip()
    return visual_result


def _localize_single_query(
    video: Any,
    video_path: str,
    query: Any,
    cfg: dict[str, Any],
    similarity_fn: SimilarityFn | None = None,
) -> dict[str, Any]:
    query_dict = _normalize_query(query)
    query_type = str(query_dict.get("query_type", "")).strip()
    if not _is_supported_query_type(query_type):
        result = _empty_result(query_dict)
        result["skip_reason"] = _unsupported_query_type_reason(query_type)
        return result

    if similarity_fn is None:
        result = _empty_result(query_dict)
        result["skip_reason"] = "Visual backend is not initialized."
        return result

    if query_type == "Visual":
        return _localize_visual_query(
            video=video,
            query_dict=query_dict,
            cfg=cfg,
            similarity_fn=similarity_fn,
        )

    if query_type == "ASR":
        return _localize_asr_query(
            video=video,
            video_path=video_path,
            query_dict=query_dict,
            cfg=cfg,
            similarity_fn=similarity_fn,
        )

    if query_type == "OCR":
        return _localize_ocr_query(
            video=video,
            query_dict=query_dict,
            cfg=cfg,
            similarity_fn=similarity_fn,
        )

    result = _empty_result(query_dict)
    result["skip_reason"] = _unsupported_query_type_reason(query_type)
    return result


def _open_video(video_path: str) -> Any:
    if VideoReader is None or cpu is None:
        raise ImportError("decord is required for focus_localizer.py. Please install decord.")
    return VideoReader(video_path, ctx=cpu(0), num_threads=1)


def localize_query_evidence(video_path: str, query: Any, config: dict | None = None) -> dict:
    """
    Localize one query to one main evidence segment plus supporting frames.

    Args:
        video_path: input video path
        query: one query item from query_result.retrieval_queries
        config: localizer config (defaults to short-video default_config)
    """
    cfg = _merge_config(config)
    normalized_query = _normalize_query(query)
    query_type = str(normalized_query.get("query_type", "")).strip()
    if not _is_supported_query_type(query_type):
        result = _empty_result(normalized_query)
        result["skip_reason"] = _unsupported_query_type_reason(query_type)
        return result

    similarity_fn: SimilarityFn | None = _resolve_similarity_fn(cfg)
    video = _open_video(video_path)

    return _localize_single_query(
        video=video,
        video_path=video_path,
        query=normalized_query,
        cfg=cfg,
        similarity_fn=similarity_fn,
    )


def localize_all_queries(
    video_path: str,
    retrieval_queries: list[Any],
    config: dict | None = None,
) -> list[dict]:
    """
    Batch localize all retrieval queries from base_Omni query extraction output.

    Args:
        video_path: input video path
        retrieval_queries: list of query dicts
        config: localizer config (defaults to short-video default_config)
    """
    cfg = _merge_config(config)
    normalized_queries = [_normalize_query(q) for q in retrieval_queries]
    query_types = {str(q.get("query_type", "")).strip() for q in normalized_queries}
    has_supported_query = any(_is_supported_query_type(t) for t in query_types)
    if not has_supported_query:
        results: list[dict] = []
        for q in normalized_queries:
            item = _empty_result(q)
            item["skip_reason"] = _unsupported_query_type_reason(str(q.get("query_type", "")))
            results.append(item)
        return results

    needs_visual = "Visual" in query_types
    needs_asr = "ASR" in query_types
    needs_ocr = "OCR" in query_types

    similarity_fn: SimilarityFn | None = None
    if needs_visual or needs_asr or needs_ocr:
        similarity_fn = _resolve_similarity_fn(cfg)

    video = _open_video(video_path)

    return [
        _localize_single_query(
            video=video,
            video_path=video_path,
            query=q,
            cfg=cfg,
            similarity_fn=similarity_fn,
        )
        for q in normalized_queries
    ]
