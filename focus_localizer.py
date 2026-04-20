"""
Lightweight query-guided local evidence localization built from FOCUS ideas.

This module adapts keyframe-focused FOCUS into a short-video local evidence
localizer with route-aware query handling:
- locate one main evidence segment per query
- return only a few supporting frames inside that segment
- always localize visually first for every supported query route
- currently supports Visual and ASR query routes

Public APIs:
- localize_query_evidence(video_path, query, config)
- localize_all_queries(video_path, retrieval_queries, config)
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import wave
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
_SUPPORTED_QUERY_TYPES = {"Visual", "ASR"}


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
    "asr_backend": "whisper",
    "whisper_model_path": "/sda/yuqifan/HFOCUS/Whisper/large-v3.pt",
    "whisper_device": "",
    "whisper_language": "",
    "whisper_temperature": 0.0,
    "whisper_beam_size": 5,
    "asr_analysis_max_events": 4,
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


def _resolve_whisper_device(device_hint: str) -> str:
    device = (device_hint or "").strip()
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_whisper_model_path(config: dict[str, Any]) -> str:
    explicit = str(config.get("whisper_model_path", "")).strip()
    if explicit:
        return explicit
    return "/sda/yuqifan/HFOCUS/Whisper/large-v3.pt"


class _OpenAIWhisperRuntime:
    def __init__(self, model_path: str, device: str):
        try:
            import whisper
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Whisper ASR requires openai-whisper. Please install it or provide config['asr_infer_fn']."
            ) from exc

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Whisper model not found: {model_path}")

        self.device = _resolve_whisper_device(device)
        self.model_path = str(model_file)
        self.model = whisper.load_model(self.model_path, device=self.device)


@lru_cache(maxsize=2)
def _get_openai_whisper_runtime(model_path: str, device: str) -> _OpenAIWhisperRuntime:
    return _OpenAIWhisperRuntime(model_path=model_path, device=device)


def _read_wav_mono_float32(audio_path: str) -> tuple[int, np.ndarray]:
    with wave.open(audio_path, "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sample_rate = int(wf.getframerate())
        frame_count = int(wf.getnframes())
        raw = wf.readframes(frame_count)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = (
            buf[:, 0].astype(np.int32)
            | (buf[:, 1].astype(np.int32) << 8)
            | (buf[:, 2].astype(np.int32) << 16)
        )
        sign = vals & 0x800000
        vals = vals - (sign << 1)
        data = vals.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1 and data.size >= channels:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        data = np.zeros((1,), dtype=np.float32)
    return sample_rate, np.clip(data, -1.0, 1.0)


def _compute_audio_stats(audio_path: str) -> dict[str, float]:
    sr, samples = _read_wav_mono_float32(audio_path)
    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    rms_db = float(20.0 * np.log10(rms + 1e-8))
    peak = float(np.max(abs_samples))
    crest = float(peak / max(1e-8, rms))
    zcr = float(np.mean(samples[:-1] * samples[1:] < 0.0)) if samples.size > 1 else 0.0

    frame_size = max(1, int(round(0.05 * sr)))
    usable = (samples.size // frame_size) * frame_size
    if usable > 0:
        frames = samples[:usable].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    else:
        frame_rms = np.array([rms], dtype=np.float32)

    median_rms = float(np.median(frame_rms))
    active_threshold = max(0.015, 1.8 * median_rms)
    active_ratio = float(np.mean(frame_rms >= active_threshold))
    burst_ratio = float(np.mean(frame_rms >= max(0.08, 2.5 * median_rms)))
    dynamic_range = float(np.percentile(frame_rms, 90) - np.percentile(frame_rms, 10))

    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / max(1, sr))
    spec_sum = float(np.sum(spectrum)) + 1e-8
    centroid = float(np.sum(freqs * spectrum) / spec_sum)
    high_band = float(np.sum(spectrum[freqs >= 2000.0]) / spec_sum)
    low_band = float(np.sum(spectrum[freqs <= 300.0]) / spec_sum)

    return {
        "sample_rate": float(sr),
        "duration_sec": float(samples.size / max(1, sr)),
        "rms_db": rms_db,
        "peak": peak,
        "crest_factor": crest,
        "zcr": zcr,
        "active_ratio": active_ratio,
        "burst_ratio": burst_ratio,
        "dynamic_range": dynamic_range,
        "spectral_centroid_hz": centroid,
        "high_band_ratio": high_band,
        "low_band_ratio": low_band,
    }


def _normalize_transcript_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_whisper_segments(raw_result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = raw_result.get("segments")
    if not isinstance(segments, list):
        return []
    parsed: list[dict[str, Any]] = []
    for seg in segments[:64]:
        if not isinstance(seg, dict):
            continue
        parsed.append(
            {
                "start_sec": float(seg.get("start", 0.0) or 0.0),
                "end_sec": float(seg.get("end", 0.0) or 0.0),
                "text": _normalize_transcript_text(str(seg.get("text", "") or "")),
                "avg_logprob": float(seg.get("avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0) or 0.0),
            }
        )
    return parsed


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _detect_sound_events(
    transcript: str,
    audio_stats: dict[str, float],
    max_events: int,
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    lower = (transcript or "").lower()

    def add(event: str, confidence: str, reason: str) -> None:
        if any(x.get("event") == event for x in events):
            return
        events.append({"event": event, "confidence": confidence, "reason": reason})

    if _contains_any(lower, ["music", "bgm", "song", "melody", "lyrics", "♪", "背景音乐", "配乐", "音乐"]):
        add("background_music", "high", "transcript contains explicit music-related cues")
    if _contains_any(lower, ["scream", "screaming", "yell", "shout", "尖叫", "惨叫", "嘶吼", "大喊"]):
        add("scream_or_loud_shout", "high", "transcript contains scream/shout related words")
    if _contains_any(lower, ["explosion", "boom", "blast", "爆炸", "轰", "爆破"]):
        add("explosion_like_event", "high", "transcript contains explosion-related cues")
    if _contains_any(lower, ["glass", "shatter", "crash", "碎裂", "破碎", "玻璃", "撞击"]):
        add("shatter_or_impact", "high", "transcript contains shatter/impact related words")

    crest = float(audio_stats.get("crest_factor", 0.0))
    active_ratio = float(audio_stats.get("active_ratio", 0.0))
    high_band = float(audio_stats.get("high_band_ratio", 0.0))
    rms_db = float(audio_stats.get("rms_db", -120.0))
    burst_ratio = float(audio_stats.get("burst_ratio", 0.0))
    zcr = float(audio_stats.get("zcr", 0.0))

    if high_band > 0.38 and crest > 7.0 and active_ratio > 0.18:
        add("high_frequency_sharp_sound", "medium", "high spectral high-band ratio with strong transients")
    if crest > 10.0 and burst_ratio > 0.04 and active_ratio < 0.55:
        add("impact_like_transients", "medium", "high crest factor with burst-like energy")
    if active_ratio > 0.80 and zcr < 0.10 and -28.0 <= rms_db <= -12.0:
        add("possible_background_music", "medium", "sustained activity with relatively stable energy")
    if active_ratio < 0.08 and rms_db < -40.0:
        add("very_low_audio_or_silence", "high", "very low energy and limited active frames")

    if not events and active_ratio > 0.18:
        add("speech_or_foreground_audio", "low", "speech-like activity detected but no specific event cue")
    return events[: max(1, int(max_events))]


def _infer_emotion_label(
    transcript: str,
    audio_stats: dict[str, float],
) -> dict[str, str]:
    lower = (transcript or "").lower()
    rms_db = float(audio_stats.get("rms_db", -120.0))
    crest = float(audio_stats.get("crest_factor", 0.0))
    high_band = float(audio_stats.get("high_band_ratio", 0.0))
    active_ratio = float(audio_stats.get("active_ratio", 0.0))

    if _contains_any(lower, ["angry", "rage", "fight", "idiot", "hate", "怒", "愤怒", "吵", "骂"]):
        return {"label": "tense", "confidence": "medium", "reason": "aggressive wording appears in transcript"}
    if _contains_any(lower, ["happy", "laugh", "fun", "哈哈", "开心", "笑"]):
        return {"label": "positive", "confidence": "medium", "reason": "positive/laughter cues in transcript"}
    if active_ratio < 0.08 and rms_db < -40.0:
        return {"label": "neutral", "confidence": "high", "reason": "audio is mostly quiet"}
    if rms_db > -16.0 or (crest > 8.0 and high_band > 0.30):
        return {"label": "tense", "confidence": "low", "reason": "high intensity and sharp acoustic profile"}
    if rms_db < -24.0 and high_band < 0.20:
        return {"label": "calm", "confidence": "low", "reason": "lower-energy and low-highband acoustic profile"}
    return {"label": "neutral", "confidence": "low", "reason": "no strong emotional cue from transcript/acoustics"}


def _build_asr_analysis_summary(
    emotion: dict[str, str],
    sound_events: list[dict[str, str]],
) -> str:
    emotion_label = str(emotion.get("label", "neutral"))
    if sound_events:
        events = ", ".join(str(x.get("event", "")) for x in sound_events if str(x.get("event", "")).strip())
    else:
        events = "none"
    return f"emotion={emotion_label}; sound_events={events}"


def _run_whisper_infer_with_model(
    whisper_model: Any,
    whisper_device: str,
    whisper_model_path: str,
    audio_path: str,
    *,
    query: dict[str, Any],
    segment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = query, segment
    language = str(config.get("whisper_language", "") or "").strip() or None
    temperature = float(config.get("whisper_temperature", 0.0) or 0.0)
    beam_size = int(config.get("whisper_beam_size", 5) or 5)
    max_events = int(config.get("asr_analysis_max_events", 4) or 4)

    transcribe_kwargs: dict[str, Any] = {
        "task": "transcribe",
        "temperature": temperature,
        "beam_size": max(1, beam_size),
        "fp16": str(whisper_device).startswith("cuda"),
    }
    if language:
        transcribe_kwargs["language"] = language

    try:
        raw_result = whisper_model.transcribe(audio_path, **transcribe_kwargs)
    except TypeError:
        transcribe_kwargs.pop("beam_size", None)
        raw_result = whisper_model.transcribe(audio_path, **transcribe_kwargs)

    raw_result = raw_result if isinstance(raw_result, dict) else {}
    transcript = _normalize_transcript_text(str(raw_result.get("text", "") or ""))
    segments_payload = _extract_whisper_segments(raw_result)
    audio_stats = _compute_audio_stats(audio_path)
    sound_events = _detect_sound_events(transcript=transcript, audio_stats=audio_stats, max_events=max_events)
    emotion = _infer_emotion_label(transcript=transcript, audio_stats=audio_stats)
    summary = _build_asr_analysis_summary(emotion=emotion, sound_events=sound_events)

    return {
        "status": "ok",
        "text": transcript,
        "language": str(raw_result.get("language", "") or ""),
        "segments": segments_payload,
        "emotion": emotion,
        "sound_events": sound_events,
        "analysis_summary": summary,
        "audio_stats": audio_stats,
        "asr_backend": "whisper",
        "whisper_device": str(whisper_device),
        "whisper_model_path": str(whisper_model_path),
    }


def build_openai_whisper_asr_infer_fn(
    whisper_model: Any,
    *,
    whisper_device: str = "",
    whisper_model_path: str = "",
) -> Callable[..., dict[str, Any]]:
    device = _resolve_whisper_device(whisper_device)
    model_path = (whisper_model_path or "").strip() or "preloaded"

    def _infer(
        audio_path: str,
        *,
        query: dict[str, Any],
        segment: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return _run_whisper_infer_with_model(
            whisper_model=whisper_model,
            whisper_device=device,
            whisper_model_path=model_path,
            audio_path=audio_path,
            query=query,
            segment=segment,
            config=config,
        )

    return _infer


def _default_asr_infer_whisper(
    audio_path: str,
    *,
    query: dict[str, Any],
    segment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_path = _resolve_whisper_model_path(config)
    device = _resolve_whisper_device(str(config.get("whisper_device", "")))
    runtime = _get_openai_whisper_runtime(model_path=model_path, device=device)
    return _run_whisper_infer_with_model(
        whisper_model=runtime.model,
        whisper_device=runtime.device,
        whisper_model_path=runtime.model_path,
        audio_path=audio_path,
        query=query,
        segment=segment,
        config=config,
    )


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
        "note": "ASR placeholder: provide config['asr_infer_fn'].",
        "audio_path": audio_path,
    }


def _resolve_asr_infer_fn(config: dict[str, Any]) -> Callable[..., Any]:
    fn = config.get("asr_infer_fn")
    if callable(fn):
        return fn

    backend = str(config.get("asr_backend", "whisper")).strip().lower()
    if backend in {"whisper", "openai-whisper", "openai_whisper"}:
        return _default_asr_infer_whisper
    return _default_asr_infer_placeholder


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
    query_dict = normalize_query_input(query)
    # Backward-compatibility bridge: OCR route is now merged into Visual route.
    if str(query_dict.get("query_type", "")).strip() == "OCR":
        query_dict["query_type"] = "Visual"
    return query_dict


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

    similarity_fn: SimilarityFn | None = None
    if needs_visual or needs_asr:
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
