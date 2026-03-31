import argparse
import re
import subprocess

import torch
from modelscope import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


FOCUSED_OMNI_PROMPT = """You are a multimodal video evidence analyst.

Watch the full video and focus on both visual stream and heard audio stream.
Do NOT output JSON.
Do NOT output extra sections.

Rules:
1. Use both video and audio evidence jointly.
2. Do NOT treat on-screen text as spoken audio unless it is actually heard.
3. If speech is unclear, explicitly say unclear.
4. Keep the output concise but complete for full-timeline understanding.
5. Do NOT force ASR from background music/noise; if no reliable speech, say so.
6. Avoid repetitive transcript loops; do not repeat the same utterance many times.

Write exactly in the following format:

[Scene]
- setting, people, roles, context

[Full Video Narrative]
- one concise paragraph summarizing the full video from start to end
- focus on temporal flow and interaction changes
- explicitly cover beginning, middle, and ending phases

[Full Audio Track Analysis]
- one concise paragraph summarizing the full audio from start to end
- include: speech presence, language, tone changes, background sounds, silence/noise periods
- if speech is unclear, explicitly say unclear
- explicitly cover beginning, middle, and ending phases

[ASR Transcript]
- provide speech transcription from heard audio only (NO timestamps)
- if speaker identity is unclear, label as Speaker-Unknown
- if words are unclear, mark them as [unclear]
- if no reliable speech is detected, output exactly: [No reliable speech detected]
- do NOT use on-screen text unless it is actually spoken in audio
"""


SECTION_HEADERS = {
    "[Scene]",
    "[Full Video Narrative]",
    "[Full Audio Track Analysis]",
    "[ASR Transcript]",
}


BASE_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

MERGE_SYSTEM_PROMPT = (
    "You are a senior multimodal analyst. Integrate multiple overlapping segment-level analyses "
    "into one coherent full-video report. Keep chronology strict, preserve uncertainty markers, "
    "and avoid dropping rare but important details."
)


def format_hhmmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_duration_seconds(video_path: str) -> float:
    """Get video duration in seconds via ffprobe, with OpenCV fallback."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
        value = float(result.stdout.strip())
        if value > 0:
            return value
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0 and frames > 0:
            return frames / fps
    except Exception as exc:
        raise RuntimeError(
            "Cannot infer video duration automatically. "
            "Please provide --video_end manually."
        ) from exc

    raise RuntimeError(
        "Cannot infer video duration automatically. Please provide --video_end manually."
    )


def build_segments(
    video_start: float,
    video_end: float,
    segment_seconds: float,
    segment_overlap: float,
) -> list[tuple[float, float]]:
    if segment_seconds <= 0:
        return [(video_start, video_end)]
    if segment_overlap < 0:
        raise ValueError("--segment_overlap must be >= 0")
    if segment_overlap >= segment_seconds:
        raise ValueError("--segment_overlap must be smaller than --segment_seconds")

    segments: list[tuple[float, float]] = []
    cursor = video_start
    eps = 1e-6

    while cursor < video_end - eps:
        seg_end = min(video_end, cursor + segment_seconds)
        segments.append((cursor, seg_end))
        if seg_end >= video_end - eps:
            break
        next_cursor = seg_end - segment_overlap
        if next_cursor <= cursor + eps:
            next_cursor = cursor + segment_seconds
        cursor = next_cursor

    return segments


def build_video_conversation(video_item: dict, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": BASE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [video_item, {"type": "text", "text": prompt_text}],
        },
    ]


def build_text_only_conversation(prompt_text: str, system_prompt: str = MERGE_SYSTEM_PROMPT) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]


def build_segment_prompt(
    seg_start: float,
    seg_end: float,
    full_start: float,
    full_end: float,
) -> str:
    return (
        f"[Global Video Range] {format_hhmmss(full_start)} - {format_hhmmss(full_end)}\n"
        f"[Current Segment Range] {format_hhmmss(seg_start)} - {format_hhmmss(seg_end)}\n"
        "Analyze only this current segment, but keep global timeline awareness.\n"
        "Use absolute timestamps within the full video when possible.\n"
        "If uncertainty exists, state uncertainty explicitly.\n\n"
        f"{FOCUSED_OMNI_PROMPT}"
    )


def build_merge_prompt(
    segment_reports: list[dict],
    full_start: float,
    full_end: float,
    is_final_pass: bool,
) -> str:
    mode_note = "final full-video pass" if is_final_pass else "intermediate merge pass"
    lines = [
        f"You are performing a {mode_note}.",
        f"Target full timeline: {format_hhmmss(full_start)} - {format_hhmmss(full_end)}.",
        "Input reports come from overlapping segments and may contain partial repeats or conflicts.",
        "Requirements:",
        "1) preserve chronological order strictly.",
        "2) deduplicate overlap repeats but keep unique details.",
        "3) combine both visual and audio evidence.",
        "4) if evidence conflicts, report both and mark uncertainty.",
        "5) produce exactly the required output format below.",
        "",
        "Output format requirement:",
        FOCUSED_OMNI_PROMPT,
        "",
        "Segment reports:",
    ]

    for idx, item in enumerate(segment_reports, start=1):
        lines.append(
            f"[Segment-{idx}] range={format_hhmmss(item['start'])}-{format_hhmmss(item['end'])}, "
            f"input_tokens={item.get('input_tokens', 'n/a')}"
        )
        lines.append(item["analysis"])
        lines.append("")

    return "\n".join(lines)


def build_model_inputs(
    processor: Qwen2_5OmniProcessor,
    conversation: list[dict],
    use_audio_in_video: bool,
    target_device: torch.device,
) -> dict:
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)
    return inputs


def decode_generation(
    processor: Qwen2_5OmniProcessor,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> str:
    if torch.is_tensor(input_ids) and output_ids.shape[1] >= input_ids.shape[1]:
        generated_ids = output_ids[:, input_ids.shape[1] :]
    else:
        generated_ids = output_ids

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if isinstance(decoded, list) and decoded else str(decoded).strip()


def get_model_context_limit(
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    fallback: int = 32768,
) -> int:
    """Best-effort infer max context length from model/tokenizer configs."""
    candidates: list[int] = []

    def _collect(obj: object | None) -> None:
        if obj is None:
            return
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
            "model_max_length",
        ):
            value = getattr(obj, name, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)

    # model configs
    _collect(getattr(model, "config", None))
    _collect(getattr(getattr(model, "config", None), "text_config", None))
    _collect(getattr(getattr(model, "thinker", None), "config", None))
    _collect(getattr(getattr(getattr(model, "thinker", None), "config", None), "text_config", None))

    # tokenizer config
    _collect(getattr(processor, "tokenizer", None))

    return min(candidates) if candidates else fallback


def build_video_item(
    video_path: str,
    video_start: float,
    video_end: float | None,
    sampling: dict,
) -> dict:
    item = {
        "type": "video",
        "video": video_path,
        "video_start": video_start,
    }
    if video_end is not None:
        item["video_end"] = video_end

    if sampling["mode"] == "nframes":
        item["nframes"] = int(sampling["nframes"])
    else:
        item["fps"] = float(sampling["fps"])
        item["min_frames"] = int(sampling["min_frames"])
        item["max_frames"] = int(sampling["max_frames"])
    return item


def sampling_to_text(sampling: dict, video_start: float, video_end: float | None) -> str:
    if sampling["mode"] == "nframes":
        return f"nframes={sampling['nframes']}, video_start={video_start}, video_end={video_end}"
    return (
        f"fps={sampling['fps']}, min_frames={sampling['min_frames']}, max_frames={sampling['max_frames']}, "
        f"video_start={video_start}, video_end={video_end}"
    )


def reduce_sampling(sampling: dict) -> bool:
    """Aggressively reduce visual tokens when context is too long."""
    if sampling["mode"] == "nframes":
        old = int(sampling["nframes"])
        new = max(4, old // 2)
        if new >= old:
            return False
        sampling["nframes"] = new
        return True

    old_max = int(sampling["max_frames"])
    if old_max > 8:
        sampling["max_frames"] = max(8, old_max // 2)
        sampling["min_frames"] = min(int(sampling["min_frames"]), int(sampling["max_frames"]))
        return True

    old_min = int(sampling["min_frames"])
    if old_min > 8:
        sampling["min_frames"] = max(8, old_min // 2)
        return True

    old_fps = float(sampling["fps"])
    if old_fps > 0.25:
        sampling["fps"] = max(0.25, old_fps * 0.7)
        return True

    sampling.clear()
    sampling.update({"mode": "nframes", "nframes": 8})
    return True


def parse_section_header(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    match = re.match(r"^\[([^\]]+)\](.*)$", stripped)
    if not match:
        return None
    header = f"[{match.group(1).strip()}]"
    suffix = match.group(2).strip()
    return header, suffix


def extract_focused_sections(text: str) -> str:
    """仅保留目标三段，便于快速验证 Omni 的音视频理解能力。"""
    lines = text.splitlines()
    out: list[str] = []
    capture = False

    for line in lines:
        header_info = parse_section_header(line)
        if header_info is not None:
            header, _ = header_info
            capture = header in SECTION_HEADERS
        if capture:
            out.append(line)

    result = "\n".join(out).strip()
    return result if result else text.strip()


def split_into_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        header_info = parse_section_header(raw_line)
        if header_info is not None:
            header, suffix = header_info
            if current_header is not None:
                sections.append((current_header, current_lines))
            current_header = header
            current_lines = [header]
            if suffix:
                suffix_line = suffix if suffix.startswith("-") else f"- {suffix}"
                current_lines.append(suffix_line)
        elif current_header is not None:
            current_lines.append(raw_line)

    if current_header is not None:
        sections.append((current_header, current_lines))
    return sections


def normalize_for_dedup(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"^\-\s*", "", line)
    line = re.sub(r"\[?\s*\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?\]?\s*", "", line)
    line = re.sub(r"[\"'`]", "", line)
    line = re.sub(r"[\[\]]", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_asr_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = ["[ASR Transcript]"]
    seen: set[str] = set()
    max_lines = 14

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        candidate = re.sub(r"\[?\s*\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?\]?\s*", "", stripped).strip()
        if not candidate:
            continue
        if re.fullmatch(r"[\[\]\-: ]+", candidate):
            continue
        if not candidate.startswith("-"):
            candidate = f"- {candidate}"

        norm = normalize_for_dedup(candidate)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(candidate)
        if len(cleaned) >= max_lines + 1:
            break

    if len(cleaned) == 1:
        cleaned.append("- [No reliable speech detected]")
    return cleaned


def postprocess_focused_output(text: str) -> str:
    sections = split_into_sections(text)
    if not sections:
        return text.strip()

    out_lines: list[str] = []
    for header, lines in sections:
        if header == "[ASR Transcript]":
            out_lines.extend(clean_asr_lines(lines))
        else:
            out_lines.extend(lines)
        out_lines.append("")

    return "\n".join(out_lines).strip()


def section_map(text: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for header, lines in split_into_sections(text):
        content: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            content.append(stripped if stripped.startswith("-") else f"- {stripped}")
        mapping[header] = content
    return mapping


def heuristic_merge_reports(reports: list[dict]) -> str:
    ordered_headers = [
        "[Scene]",
        "[Full Video Narrative]",
        "[Full Audio Track Analysis]",
        "[ASR Transcript]",
    ]
    max_lines = {
        "[Scene]": 8,
        "[Full Video Narrative]": 10,
        "[Full Audio Track Analysis]": 10,
        "[ASR Transcript]": 14,
    }

    out: list[str] = []
    for header in ordered_headers:
        seen: set[str] = set()
        lines: list[str] = []
        for item in reports:
            mapping = section_map(item.get("analysis", ""))
            for line in mapping.get(header, []):
                norm = normalize_for_dedup(line)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                lines.append(line)
                if len(lines) >= max_lines[header]:
                    break
            if len(lines) >= max_lines[header]:
                break

        if not lines:
            if header == "[ASR Transcript]":
                lines = ["- [No reliable speech detected]"]
            else:
                lines = ["- [No additional evidence extracted]"]

        out.append(header)
        out.extend(lines)
        out.append("")

    return postprocess_focused_output("\n".join(out).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni focused AV capability test")
    parser.add_argument("--video", type=str, default="test/1.mp4", help="Input video path")
    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni", help="Model path")
    parser.add_argument("--device_map", type=str, default="auto", help="HF device_map")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--fps", type=float, default=3.0, help="Video sampling fps for model input")
    parser.add_argument("--min_frames", type=int, default=16, help="Minimum sampled frames")
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum sampled frames")
    parser.add_argument("--nframes", type=int, default=None, help="Fixed sampled frame count (overrides fps)")
    parser.add_argument("--video_start", type=float, default=0.0, help="Video start time in seconds")
    parser.add_argument("--video_end", type=float, default=None, help="Video end time in seconds")
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help="Override context window limit. Default: auto-detect from model/tokenizer",
    )
    parser.add_argument(
        "--context_reserve_tokens",
        type=int,
        default=512,
        help="Reserved tokens to avoid hitting hard context/OOM boundary",
    )
    parser.add_argument(
        "--max_adaptive_attempts",
        type=int,
        default=8,
        help="Maximum retries for auto frame reduction when context is too long",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=2,
        help="Retry count after CUDA OOM (will reduce generation length and/or frames)",
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=0.0,
        help="Enable segmented inference with fixed sampling per segment. 0 disables segmentation.",
    )
    parser.add_argument(
        "--segment_overlap",
        type=float,
        default=2.0,
        help="Overlap seconds between adjacent segments to reduce boundary information loss.",
    )
    parser.add_argument(
        "--segment_min_seconds",
        type=float,
        default=6.0,
        help="Minimum segment duration when recursively splitting a heavy segment.",
    )
    parser.add_argument(
        "--max_segment_split_depth",
        type=int,
        default=3,
        help="Max recursive split depth for too-long or OOM segments in segmented mode.",
    )
    parser.add_argument(
        "--merge_max_new_tokens",
        type=int,
        default=1200,
        help="Max new tokens for report merge stage in segmented mode.",
    )
    parser.add_argument(
        "--merge_batch_size",
        type=int,
        default=6,
        help="How many segment reports to merge per round in segmented mode.",
    )
    parser.add_argument(
        "--segment_report_path",
        type=str,
        default=None,
        help="Optional path to save all segment-level reports before final merge.",
    )
    parser.add_argument(
        "--final_report_path",
        type=str,
        default=None,
        help="Optional path to save final merged report.",
    )
    args = parser.parse_args()

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.disable_talker()  # text-only output for analysis stability

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    use_audio_in_video = True

    target_device = model.device if hasattr(model, "device") else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_limit = args.max_context_tokens or get_model_context_limit(model, processor)

    if args.nframes is not None:
        sampling = {"mode": "nframes", "nframes": max(4, int(args.nframes))}
    else:
        min_frames = max(1, int(args.min_frames))
        max_frames = max(min_frames, int(args.max_frames))
        sampling = {
            "mode": "fps",
            "fps": max(0.1, float(args.fps)),
            "min_frames": min_frames,
            "max_frames": max_frames,
        }

    # ------------------------------------------------------------------
    # Mode A: segmented inference with fixed sampling per segment
    # ------------------------------------------------------------------
    if args.segment_seconds > 0:
        if args.segment_min_seconds <= 0:
            raise ValueError("--segment_min_seconds must be > 0")
        if args.merge_batch_size < 2:
            raise ValueError("--merge_batch_size must be >= 2")

        full_video_start = float(args.video_start)
        full_video_end = float(args.video_end) if args.video_end is not None else get_video_duration_seconds(args.video)
        if full_video_end <= full_video_start:
            raise ValueError("--video_end must be greater than --video_start")

        segment_token_budget = max(1024, context_limit - int(args.max_new_tokens) - int(args.context_reserve_tokens))
        merge_token_budget = max(1024, context_limit - int(args.merge_max_new_tokens) - int(args.context_reserve_tokens))

        initial_segments = build_segments(
            video_start=full_video_start,
            video_end=full_video_end,
            segment_seconds=float(args.segment_seconds),
            segment_overlap=float(args.segment_overlap),
        )

        segment_logs: list[str] = []
        segment_reports: list[dict] = []

        def run_single_segment(seg_start: float, seg_end: float) -> dict:
            video_item = build_video_item(args.video, seg_start, seg_end, sampling)
            seg_prompt = build_segment_prompt(seg_start, seg_end, full_video_start, full_video_end)
            conversation = build_video_conversation(video_item, seg_prompt)
            inputs = build_model_inputs(
                processor=processor,
                conversation=conversation,
                use_audio_in_video=use_audio_in_video,
                target_device=target_device,
            )

            input_ids = inputs.get("input_ids")
            input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else 0
            if input_len > segment_token_budget:
                raise RuntimeError(f"SEGMENT_INPUT_TOO_LONG: input_tokens={input_len}, budget={segment_token_budget}")

            safe_max_new_tokens = context_limit - input_len - int(args.context_reserve_tokens)
            if safe_max_new_tokens < 32:
                raise RuntimeError(
                    f"SEGMENT_INPUT_TOO_LONG: no generation room. input_tokens={input_len}, context_limit={context_limit}"
                )
            segment_max_new_tokens = min(int(args.max_new_tokens), safe_max_new_tokens)

            try:
                output_ids = model.generate(
                    **inputs,
                    use_audio_in_video=use_audio_in_video,
                    return_audio=False,
                    max_new_tokens=segment_max_new_tokens,
                )
            except torch.OutOfMemoryError as exc:
                raise RuntimeError("SEGMENT_OOM") from exc

            raw_text = decode_generation(processor, output_ids, input_ids)
            focused = postprocess_focused_output(extract_focused_sections(raw_text))
            return {
                "start": seg_start,
                "end": seg_end,
                "input_tokens": input_len,
                "max_new_tokens": segment_max_new_tokens,
                "analysis": focused,
            }

        def run_segment_recursive(seg_start: float, seg_end: float, depth: int = 0) -> list[dict]:
            duration = seg_end - seg_start
            try:
                return [run_single_segment(seg_start, seg_end)]
            except RuntimeError as exc:
                reason = str(exc)
                can_split = (
                    depth < int(args.max_segment_split_depth)
                    and duration > max(float(args.segment_min_seconds) * 1.5, 1.0)
                    and ("SEGMENT_INPUT_TOO_LONG" in reason or "SEGMENT_OOM" in reason)
                )
                if not can_split:
                    raise

                half_overlap = min(float(args.segment_overlap) / 2.0, duration / 4.0)
                mid = (seg_start + seg_end) / 2.0
                left_start = seg_start
                left_end = min(seg_end, mid + half_overlap)
                right_start = max(seg_start, mid - half_overlap)
                right_end = seg_end

                if left_end <= left_start + 1e-6 or right_end <= right_start + 1e-6:
                    left_end = mid
                    right_start = mid

                segment_logs.append(
                    f"[SegmentSplit depth={depth + 1}] "
                    f"{format_hhmmss(seg_start)}-{format_hhmmss(seg_end)} -> "
                    f"{format_hhmmss(left_start)}-{format_hhmmss(left_end)} + "
                    f"{format_hhmmss(right_start)}-{format_hhmmss(right_end)}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return run_segment_recursive(left_start, left_end, depth + 1) + run_segment_recursive(
                    right_start, right_end, depth + 1
                )

        for idx, (seg_start, seg_end) in enumerate(initial_segments, start=1):
            segment_logs.append(
                f"[SegmentInit {idx}/{len(initial_segments)}] {format_hhmmss(seg_start)}-{format_hhmmss(seg_end)}"
            )
            sub_reports = run_segment_recursive(seg_start, seg_end, depth=0)
            segment_reports.extend(sub_reports)

        segment_reports.sort(key=lambda x: (x["start"], x["end"]))

        if args.segment_report_path:
            with open(args.segment_report_path, "w", encoding="utf-8") as f:
                for idx, item in enumerate(segment_reports, start=1):
                    f.write(
                        f"[Segment-{idx}] {format_hhmmss(item['start'])}-{format_hhmmss(item['end'])}, "
                        f"input_tokens={item['input_tokens']}, max_new_tokens={item['max_new_tokens']}\n"
                    )
                    f.write(item["analysis"])
                    f.write("\n\n")

        merge_logs: list[str] = []

        def merge_once(reports: list[dict], is_final_pass: bool) -> dict:
            prompt = build_merge_prompt(
                segment_reports=reports,
                full_start=full_video_start,
                full_end=full_video_end,
                is_final_pass=is_final_pass,
            )
            conversation = build_text_only_conversation(prompt, system_prompt=MERGE_SYSTEM_PROMPT)
            inputs = build_model_inputs(
                processor=processor,
                conversation=conversation,
                use_audio_in_video=False,
                target_device=target_device,
            )
            input_ids = inputs.get("input_ids")
            input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else 0
            if input_len > merge_token_budget and len(reports) > 1:
                raise RuntimeError(f"MERGE_INPUT_TOO_LONG: input_tokens={input_len}, budget={merge_token_budget}")

            safe_max_new_tokens = context_limit - input_len - int(args.context_reserve_tokens)
            if safe_max_new_tokens < 32:
                raise RuntimeError(
                    f"MERGE_INPUT_TOO_LONG: no generation room. input_tokens={input_len}, context_limit={context_limit}"
                )
            merge_max_new_tokens = min(int(args.merge_max_new_tokens), safe_max_new_tokens)

            try:
                output_ids = model.generate(
                    **inputs,
                    use_audio_in_video=False,
                    return_audio=False,
                    max_new_tokens=merge_max_new_tokens,
                )
            except torch.OutOfMemoryError as exc:
                raise RuntimeError("MERGE_OOM") from exc
            merged_raw = decode_generation(processor, output_ids, input_ids)
            merged_text = postprocess_focused_output(extract_focused_sections(merged_raw))

            return {
                "start": reports[0]["start"],
                "end": reports[-1]["end"],
                "input_tokens": input_len,
                "max_new_tokens": merge_max_new_tokens,
                "analysis": merged_text,
            }

        current_reports = segment_reports
        merge_round = 1
        dynamic_batch_size = int(args.merge_batch_size)

        while len(current_reports) > 1:
            next_round: list[dict] = []
            idx = 0
            while idx < len(current_reports):
                batch = current_reports[idx : idx + dynamic_batch_size]
                is_final_pass = len(current_reports) <= dynamic_batch_size
                try:
                    merged_item = merge_once(batch, is_final_pass=is_final_pass)
                    merge_logs.append(
                        f"[Merge round={merge_round}] batch={len(batch)} "
                        f"{format_hhmmss(batch[0]['start'])}-{format_hhmmss(batch[-1]['end'])} "
                        f"-> input_tokens={merged_item['input_tokens']}"
                    )
                    next_round.append(merged_item)
                    idx += len(batch)
                except RuntimeError as exc:
                    err_msg = str(exc)
                    if "MERGE_INPUT_TOO_LONG" in err_msg and dynamic_batch_size > 2:
                        dynamic_batch_size = max(2, dynamic_batch_size // 2)
                        merge_logs.append(
                            f"[MergeAdjust] reduce merge_batch_size to {dynamic_batch_size} due to long merge input."
                        )
                        continue
                    if "MERGE_INPUT_TOO_LONG" in err_msg or "MERGE_OOM" in err_msg:
                        fallback_text = heuristic_merge_reports(batch)
                        merge_logs.append(
                            f"[MergeFallback] batch={len(batch)} "
                            f"{format_hhmmss(batch[0]['start'])}-{format_hhmmss(batch[-1]['end'])} "
                            f"used heuristic merge due to: {err_msg}"
                        )
                        next_round.append(
                            {
                                "start": batch[0]["start"],
                                "end": batch[-1]["end"],
                                "input_tokens": -1,
                                "max_new_tokens": 0,
                                "analysis": fallback_text,
                            }
                        )
                        idx += len(batch)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
            current_reports = next_round
            merge_round += 1

        final_report = current_reports[0]["analysis"] if current_reports else ""
        final_report = postprocess_focused_output(final_report)
        if not final_report.strip():
            final_report = heuristic_merge_reports(segment_reports)
            merge_logs.append("[MergeFallback] final report was empty; used heuristic merge on all segments.")

        final_report_path = args.final_report_path
        if final_report_path is None and args.segment_report_path:
            if args.segment_report_path.endswith(".txt"):
                final_report_path = args.segment_report_path[:-4] + ".final.txt"
            else:
                final_report_path = args.segment_report_path + ".final.txt"
        if final_report_path:
            with open(final_report_path, "w", encoding="utf-8") as f:
                f.write(final_report.strip() + "\n")

        print("=" * 60)
        print("Segmented Sampling Config (Fixed)")
        print("=" * 60)
        print(sampling_to_text(sampling, args.video_start, args.video_end))
        print(
            f"segment_seconds={args.segment_seconds}, segment_overlap={args.segment_overlap}, "
            f"initial_segments={len(initial_segments)}, actual_segments={len(segment_reports)}"
        )
        print(
            f"context_limit={context_limit}, per_segment_budget={segment_token_budget}, "
            f"merge_budget={merge_token_budget}"
        )
        if segment_logs:
            print("Segment logs:")
            for line in segment_logs:
                print(line)
        if merge_logs:
            print("Merge logs:")
            for line in merge_logs:
                print(line)
        if final_report_path:
            print(f"Final report saved to: {final_report_path}")

        print("=" * 60)
        print("Focused Output")
        print("=" * 60)
        print(final_report)
        return

    # ------------------------------------------------------------------
    # Mode B: single-pass inference with adaptive fallback
    # ------------------------------------------------------------------
    input_token_budget = max(1024, context_limit - int(args.max_new_tokens) - int(args.context_reserve_tokens))
    adaptive_logs: list[str] = []
    inputs = None

    for attempt in range(1, args.max_adaptive_attempts + 1):
        video_item = build_video_item(args.video, args.video_start, args.video_end, sampling)
        conversation = build_video_conversation(video_item, FOCUSED_OMNI_PROMPT)

        candidate_inputs = build_model_inputs(
            processor=processor,
            conversation=conversation,
            use_audio_in_video=use_audio_in_video,
            target_device=target_device,
        )

        input_len = None
        if torch.is_tensor(candidate_inputs.get("input_ids")):
            input_len = int(candidate_inputs["input_ids"].shape[1])

        if input_len is None or input_len <= input_token_budget:
            inputs = candidate_inputs
            break

        adaptive_logs.append(
            f"[Adaptive {attempt}] input_tokens={input_len} exceeds budget={input_token_budget}; "
            f"reduce sampling from ({sampling_to_text(sampling, args.video_start, args.video_end)})"
        )

        del candidate_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not reduce_sampling(sampling):
            raise RuntimeError(
                f"Unable to reduce input length under budget. "
                f"input_tokens={input_len}, budget={input_token_budget}."
            )
    else:
        raise RuntimeError(
            f"Adaptive reduction exhausted ({args.max_adaptive_attempts} attempts) and input is still too long."
        )

    if inputs is None:
        raise RuntimeError("Failed to prepare model inputs.")

    input_ids = inputs.get("input_ids")
    input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else 0
    safe_max_new_tokens = context_limit - input_len - int(args.context_reserve_tokens)
    if safe_max_new_tokens < 32:
        raise RuntimeError(
            f"Not enough context left for generation: context_limit={context_limit}, "
            f"input_tokens={input_len}, reserve={args.context_reserve_tokens}."
        )
    generation_max_new_tokens = min(int(args.max_new_tokens), safe_max_new_tokens)

    output_ids = None
    oom_logs: list[str] = []
    oom_retry_budget = args.max_oom_retries

    while True:
        try:
            output_ids = model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
                max_new_tokens=generation_max_new_tokens,
            )
            break
        except torch.OutOfMemoryError as exc:
            if oom_retry_budget <= 0:
                raise RuntimeError(
                    "CUDA OOM after retries. Try lower --max_new_tokens, lower --max_frames "
                    "or use --nframes (e.g. 8~16)."
                ) from exc

            oom_retry_budget -= 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            old_tokens = generation_max_new_tokens
            generation_max_new_tokens = max(32, generation_max_new_tokens // 2)

            action = (
                f"reduce max_new_tokens from {old_tokens} to {generation_max_new_tokens}"
                if generation_max_new_tokens < old_tokens
                else "reduce frames and rebuild inputs"
            )
            oom_logs.append(f"[OOM Retry] {action}")

            if generation_max_new_tokens < old_tokens:
                continue

            if not reduce_sampling(sampling):
                raise RuntimeError(
                    "CUDA OOM and cannot further reduce sampling."
                ) from exc

            video_item = build_video_item(args.video, args.video_start, args.video_end, sampling)
            conversation = build_video_conversation(video_item, FOCUSED_OMNI_PROMPT)
            inputs = build_model_inputs(
                processor=processor,
                conversation=conversation,
                use_audio_in_video=use_audio_in_video,
                target_device=target_device,
            )
            input_ids = inputs.get("input_ids")
            input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else 0
            safe_max_new_tokens = context_limit - input_len - int(args.context_reserve_tokens)
            if safe_max_new_tokens < 32:
                raise RuntimeError(
                    f"Not enough context left for generation after OOM retry: context_limit={context_limit}, "
                    f"input_tokens={input_len}, reserve={args.context_reserve_tokens}."
                ) from exc
            generation_max_new_tokens = min(generation_max_new_tokens, safe_max_new_tokens)

    input_ids = inputs.get("input_ids")
    input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else input_len
    raw_text = decode_generation(processor, output_ids, input_ids)
    focused = postprocess_focused_output(extract_focused_sections(raw_text))

    print("=" * 60)
    print("Sampling Config")
    print("=" * 60)
    print(sampling_to_text(sampling, args.video_start, args.video_end))
    print(f"context_limit={context_limit}, input_tokens={input_len}, max_new_tokens={generation_max_new_tokens}")
    if adaptive_logs:
        print("Adaptive logs:")
        for line in adaptive_logs:
            print(line)
    if oom_logs:
        print("OOM logs:")
        for line in oom_logs:
            print(line)

    print("=" * 60)
    print("Focused Output")
    print("=" * 60)
    print(focused)


if __name__ == "__main__":
    main()
