"""
Qwen2.5-Omni 模块 A：全局视频理解（原生视频 + 音频）
用途：输入原生视频，输出自然语言全局理解报告（非 JSON）
并可选衔接模块 C：Query 提炼（文本小模型）+ ASR 证据补充
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    pipeline,
)
from qwen_omni_utils import process_mm_info


MAX_RETRIEVAL_QUERIES = 5


GLOBAL_UNDERSTANDING_PROMPT = """You are a multimodal video evidence analyst.

Watch the video and produce a structured global understanding report
optimized for downstream evidence retrieval and verification.

Do NOT output JSON. Follow the exact format below strictly, as it will be parsed by later modules.

===================== 
Key Requirements: 
=====================

1. Use both visual and audio evidence jointly.
2. Separate observation and interpretation strictly:
   - Use [Observed] for direct evidence
   - Use [Inferred] for interpretation
3. Assign each key event an Event ID: (E1, E2, E3...)
4. Unclear points MUST explicitly reference related Event IDs.
5. Audio must be analyzed across the full timeline, not only sparse moments.
6. Do NOT treat on-screen text as spoken audio unless it is actually heard. 
7. If speech is unclear, explicitly mark [unclear]. 
8. Do NOT force ASR from background music/noise.
9. Do NOT use exact timestamps.
10. Use only coarse temporal phase hints such as:
   - beginning
   - early-middle
   - middle
   - late-middle
   - ending
   - transition
11. Each key visual event must have an Event ID: E1, E2, E3...
12. Focus on making the output useful for: 
   - later frame retrieval 
   - evidence verification


===================== 
Output Format: 
=====================

[Scene]
- setting: ...
- people: ...
- roles: ...
- context: ...

[Full Video Narrative]
- one concise paragraph describing the full video flow from beginning to ending
- focus on interaction changes and key transitions

[Observed Visual Events]

E1 [phase]:
- [Observed] ...

E2 [phase]:
- [Observed] ...

E3 [phase]:
- [Observed] ...

[Full Audio Track Analysis]
- one concise paragraph summarizing speech presence, tone/emotion, background sound changes, and silence/noise periods
- cover beginning, middle, and ending
- do not fabricate exact wording if unclear

[ASR Transcript] 
- provide speech transcription from heard audio only (NO timestamps) 
- if speaker identity is unclear, label as Speaker-Unknown - if words are unclear, mark them as [unclear] 
- do NOT use on-screen text unless it is actually spoken in audio

[Main Interpretation]
- [Inferred] the most likely explanation of what is happening

[Alternative Interpretation]
- [Inferred] provide a second plausible explanation from another angle，this is mandatory 

[Key Unclear Points]

U1 (related to E2):
- what is unclear:
- why it is unclear:
- what needs verification:

U2 (related to E3):
- what is unclear:
- why it is unclear:
- what needs verification:

[Preliminary Conclusion]
- cautious summary
- confidence: low / medium / high
"""


QUERY_EXTRACTION_PROMPT = """You are a query extraction assistant for multimodal video evidence verification.

Input:
A natural-language global video understanding report written by a multimodal model.

Your task:
Extract 1 to 5 high-value queries for downstream evidence verification.

These queries will be routed to downstream video-internal retrieval modules according to query_type:
- ASR: retrieve spoken words, narration, tone, or sound cues from the video's audio track
- OCR: retrieve on-screen text from the video
- Visual: retrieve visual evidence such as people, objects, symbols, actions, gestures, clothing, patches, flags, or scenes
- Mixed: retrieve evidence that clearly requires multiple video-internal sources together

For each query, you must also decide whether the retrieved evidence may require external background checking.
This is not a separate query type.
Instead, use:
- needs_external_check: true / false
- external_check_focus
- external_check_hint

External checking is for cases where a retrieved person, symbol, slogan, phrase, gesture, organization, historical reference, or other marker may have important meaning beyond what is directly observable in the video.

Do NOT invent any new fact.
Only use information explicitly present in the report.

=====================
Core Rules
=====================

1. Prioritize evidence gaps explicitly listed in [Key Unclear Points].
2. If [Key Unclear Points] is missing or weak, use ambiguity from [Alternative Interpretation], [Main Interpretation], [Observed Visual Events], [Full Audio Track Analysis], or [ASR Transcript].
3. Prefer fewer but more important queries.
4. If the report is already clear enough, output an empty list.
5. Each query must be concrete, short, and retrieval-friendly.
6. Do NOT output broad video-level interpretation queries.
7. Do NOT ask abstract questions about the whole video's ideology, message, motive, or intent.
8. Do NOT invent identities, symbols, or meanings that are not explicitly supported by the report.
9. query_type only describes the main video-internal retrieval route. It does NOT describe whether external checking is needed.
10. needs_external_check can be true for any query_type: ASR, OCR, Visual, or Mixed.

=====================
Output Format
=====================

Output valid JSON only in exactly this format:

{
  "retrieval_queries": [
    {
      "id": "Q1",
      "time_hint": "",
      "query_type": "",
      "query_text": "",
      "why_this_query": "",
      "needs_external_check": false,
      "external_check_focus": "",
      "external_check_hint": ""
    }
  ]
}

=====================
Field Rules
=====================

- id:
  Use Q1, Q2, Q3...

- time_hint:
  A coarse evidence locator only, not an exact timestamp.
  Prefer one of:
  beginning, early-middle, middle, late-middle, ending, transition
  If useful, you may append an unclear-point label such as:
  "middle | U1"

- query_type:
  Must be one of:
  ASR, OCR, Visual, Mixed

  Choose:
  - ASR when the evidence is mainly spoken words, dialogue, narration, shouted phrases, tone, crying, laughter, chanting, or other audible cues
  - OCR when the evidence is mainly subtitles, captions, titles, banners, headlines, labels, signs, or readable on-screen text
  - Visual when the evidence is mainly people, appearance, clothing, gestures, movement, object interaction, symbols, flags, patches, weapons, or scene content
  - Mixed when verification clearly requires more than one video-internal source together

- query_text:
  A short, keyword-rich, retrieval-friendly query.
  Use one sentence only.
  Keep it concrete and specific.
  Focus on observable evidence only.
  Prefer short phrases involving:
  - spoken wording
  - visible text
  - people
  - clothing
  - objects
  - symbols
  - actions
  - gestures
  - sound cues
  - scene elements

- why_this_query:
  One short sentence explaining why this query matters for later verification or judgment.
  Keep it practical and evidence-oriented.

- needs_external_check:
  true or false

  Set it to true when the retrieved evidence may also require external background investigation.

  Typical triggers include:
  - a potentially recognizable public figure whose identity is uncertain
  - a named person, group, organization, country marker, or historical event
  - a distinctive symbol, flag, patch, logo, emblem, salute, or gesture
  - a slogan, phrase, nickname, chant, or headline that may carry political, religious, ideological, extremist, or historical meaning
  - any word, text, or visual marker whose significance could materially change the final interpretation once identified

  Do NOT set it to true for generic objects, ordinary actions, or generic appearance descriptions unless they likely carry specific external symbolic or identity significance.

- external_check_focus:
  If needs_external_check is true, choose one of:
  identity, symbol, slogan, phrase, historical_reference, organization, gesture, ideology_marker, unknown

  If needs_external_check is false, output an empty string.

- external_check_hint:
  If needs_external_check is true, write one short sentence explaining what may need external investigation.
  Keep it concise and specific.
  If needs_external_check is false, output an empty string.

=====================
Important Precision Rules
=====================

1. Never fabricate exact timestamps.
2. Never convert speculation into fact.
3. Never infer religion, ethnicity, nationality, ideology, or political alignment from appearance alone.
4. Only name a person, group, religion, country, ideology, organization, or symbol if it is explicitly stated in the report or directly indicated by visible text/symbols described in the report.
5. If identity is uncertain, use neutral observable descriptions such as:
   - person in black robe
   - man in suit and tie
   - six-pointed star symbol
   - red armband with black symbol
6. needs_external_check should be used carefully, not over-triggered.
7. Do not output duplicate or near-duplicate queries unless they clearly serve different evidence targets.
"""

QUERY_PRECISION_ADDON = """\
Additional precision constraints for this run:

- Use time_hint only as a soft retrieval prior, never as an exact timestamp.
- Prefer phase hints already present in the report:
  beginning / early-middle / middle / late-middle / ending / transition
- query_text must stay short, concrete, and easy to route downstream.
- For ASR queries, prefer spoken wording, speaker tone, narration, chants, or sound cues.
- For OCR queries, prefer exact text phrases if explicitly present in the report.
- For Visual queries, prefer distinctive visible anchors such as clothing, symbol, flag, patch, object, gesture, motion, or scene composition.
- For Mixed queries, use them only when the same verification target clearly needs multiple video-internal sources together.
- needs_external_check may be true for any query_type.
- Set needs_external_check = true only when the retrieved evidence may reasonably benefit from external background lookup.
- Strong triggers for external checking include:
  named people,
  uncertain but potentially recognizable figures,
  flags,
  symbols,
  patches,
  logos,
  slogans,
  nicknames,
  chants,
  historical references,
  organization names,
  ideology-loaded phrases,
  or distinctive gestures/salutes.
- Do not set needs_external_check = true for generic actions or generic people unless there is likely external symbolic or identity significance.
- external_check_focus must be one of:
  identity / symbol / slogan / phrase / historical_reference / organization / gesture / ideology_marker / unknown
- external_check_hint should be short, practical, and investigative.
- If needs_external_check is false, external_check_focus and external_check_hint must both be empty strings.
- Keep why_this_query short and evidence-oriented.
- Output JSON only.
- The first non-space character must be "{"
- The last non-space character must be "}"
"""


SEGMENT_MERGE_PROMPT = """You are a senior video-analysis synthesis assistant.

You will receive multiple overlapping segment reports generated from the same video.
Your task is to merge them into ONE coherent full-video report.

Rules:
1. Preserve chronology strictly from beginning to ending.
2. Deduplicate repeated content caused by segment overlap.
3. Keep unique details; do not drop rare but important evidence.
4. If segment reports conflict, keep both possibilities and mark uncertainty.
5. Use both visual and audio evidence.
6. Do NOT output timestamps.
7. Do NOT output JSON.

===================== 
Output Format: 
=====================

[Scene]
- setting, people, roles, context

[Full Video Narrative]
- one concise paragraph describing the full video flow from beginning to ending
- focus on interaction changes and key transitions

[Observed Visual Events]

E1 [phase]:
- [Observed] ...

E2 [phase]:
- [Observed] ...

E3 [phase]:
- [Observed] ...

[Full Audio Track Analysis]
- one concise paragraph summarizing speech presence, tone/emotion, background sound changes, and silence/noise periods
- cover beginning, middle, and ending
- do not fabricate exact wording if unclear

[ASR Transcript] 
- provide speech transcription from heard audio only (NO timestamps) 
- if speaker identity is unclear, label as Speaker-Unknown - if words are unclear, mark them as [unclear] 
- do NOT use on-screen text unless it is actually spoken in audio

[Main Interpretation]
- [Inferred] the most likely explanation of what is happening

[Alternative Interpretation]
- [Inferred] provide a second plausible explanation from another angle，this is mandatory 

[Key Unclear Points]

U1 (related to E2):
- what is unclear:
- why it is unclear:
- what needs verification:

U2 (related to E3):
- what is unclear:
- why it is unclear:
- what needs verification:

[Preliminary Conclusion]
- cautious summary
- confidence: low / medium / high
"""

MERGE_REQUIRED_HEADERS = [
    "[Scene]",
    "[Full Video Narrative]",
    "[Observed Visual Events]",
    "[Full Audio Track Analysis]",
    "[ASR Transcript]",
    "[Main Interpretation]",
    "[Alternative Interpretation]",
    "[Key Unclear Points]",
    "[Preliminary Conclusion]",
]


_EVENT_TIME_RE = re.compile(r"^\s*([EA]\d+)\s*\[([^\]]+)\]\s*:", re.MULTILINE)
_EVENT_ID_RE = re.compile(r"\b([EA]\d+)\b", re.IGNORECASE)
_TIME_TOKEN_RE = re.compile(r"(\d{1,2}):(\d{2})")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class RetrievalQuery:
    id: str
    time_hint: str
    query_type: str
    query_text: str
    why_this_query: str
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "time_hint": self.time_hint,
            "query_type": self.query_type,
            "query_text": self.query_text,
            "why_this_query": self.why_this_query,
        }
        payload.update(self.extra_fields)
        return payload


@dataclass
class QueryEvidence:
    id: str
    time_hint: str
    query_text: str
    asr_text: str
    evidence_note: str
    audio_segment_path: str


@dataclass
class QueryExtractionResult:
    raw_text: str
    parsed_json: dict[str, Any] | None
    retrieval_queries: list[RetrievalQuery]
    query_evidence: list[QueryEvidence]
    parse_error: bool
    parse_error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "retrieval_queries": [_query_to_output_dict(q) for q in self.retrieval_queries],
            "query_evidence": [asdict(e) for e in self.query_evidence],
            "parse_error": self.parse_error,
            "parse_error_message": self.parse_error_message,
        }


@dataclass
class GlobalPipelineResult:
    report_text: str
    query_result: QueryExtractionResult
    evidence_results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_text": self.report_text,
            "query_result": self.query_result.to_dict(),
            "evidence_results": self.evidence_results,
        }


@dataclass
class OmniRuntime:
    model: Qwen2_5OmniForConditionalGeneration
    processor: Qwen2_5OmniProcessor
    input_device: torch.device
    model_dtype: torch.dtype
    use_audio_in_video: bool


@dataclass
class QueryRuntime:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


@dataclass
class ASRRuntime:
    asr_pipe: Any


@dataclass
class ModelRegistry:
    omni: OmniRuntime | None = None
    query: QueryRuntime | None = None
    asr: ASRRuntime | None = None


def build_global_understanding_prompt(user_focus: str = "") -> str:
    """构建模块 A 的用户提示词。"""
    prompt = GLOBAL_UNDERSTANDING_PROMPT
    if user_focus.strip():
        prompt = f"{prompt}\n\nAdditional focus: {user_focus.strip()}"
    return prompt


def _format_hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_video_duration_seconds(video_path: str) -> float:
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        try:
            value = float((proc.stdout or "").strip())
            if value > 0:
                return value
        except Exception:
            pass
    raise RuntimeError("Failed to read video duration by ffprobe.")


def _build_segments(
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


def _strip_timestamp_tokens(text: str) -> str:
    cleaned = re.sub(r"\[?\s*\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?\]?\s*", "", text or "")
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    return cleaned.strip()


def _postprocess_report_text(text: str) -> str:
    text = _strip_timestamp_tokens(text)
    lines = text.splitlines()
    out: list[str] = []
    in_asr = False
    seen_asr: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\[ASR Transcript\]$", stripped):
            in_asr = True
            out.append("[ASR Transcript]")
            continue
        if re.match(r"^\[[^\]]+\]$", stripped) and stripped != "[ASR Transcript]":
            in_asr = False
            out.append(line)
            continue
        if not in_asr:
            out.append(line)
            continue

        if not stripped:
            continue
        content = stripped
        if not content.startswith("-"):
            content = f"- {content}"
        norm = re.sub(r"^\-\s*", "", content).strip().lower()
        norm = re.sub(r"[\"'`]", "", norm)
        norm = re.sub(r"\s+", " ", norm)
        if not norm:
            continue
        if norm in seen_asr:
            continue
        seen_asr.add(norm)
        out.append(content)

    merged = "\n".join(out).strip()
    if "[ASR Transcript]" in merged:
        asr_tail = merged.split("[ASR Transcript]", 1)[1].strip()
        if not asr_tail:
            merged = merged + "\n- [No reliable speech detected]"
    return merged


def _build_segment_merge_prompt_text(
    segment_reports: list[dict[str, Any]],
    video_start: float,
    video_end: float,
) -> str:
    lines = [
        SEGMENT_MERGE_PROMPT,
        "",
        f"Full video range: {_format_hhmmss(video_start)} - {_format_hhmmss(video_end)}",
        "",
        "Segment reports:",
    ]
    for idx, item in enumerate(segment_reports, start=1):
        lines.append(
            f"[Segment-{idx}] range={_format_hhmmss(float(item['start']))}-{_format_hhmmss(float(item['end']))}"
        )
        lines.append(str(item["report"]).strip())
        lines.append("")
    return "\n".join(lines).strip()


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _extract_headers_from_text(text: str) -> set[str]:
    headers: set[str] = set()
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^\[[^\]]+\]$", stripped):
            headers.add(stripped)
    return headers


def _normalize_line_for_overlap(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    return line


def _append_non_overlapping_text(base: str, addition: str) -> str:
    if not addition.strip():
        return base
    if addition.strip() in base:
        return base

    base_lines = base.splitlines()
    add_lines = addition.splitlines()
    max_overlap = min(len(base_lines), len(add_lines), 20)
    overlap = 0
    for n in range(max_overlap, 0, -1):
        base_tail = [_normalize_line_for_overlap(x) for x in base_lines[-n:]]
        add_head = [_normalize_line_for_overlap(x) for x in add_lines[:n]]
        if base_tail == add_head:
            overlap = n
            break
    merged_lines = base_lines + add_lines[overlap:]
    return "\n".join(merged_lines).strip()


def _generate_query_text_with_stats(
    runtime: QueryRuntime,
    prompt: str,
    max_new_tokens: int,
    system_message: str,
) -> tuple[str, int]:
    inputs = _build_query_inputs(
        runtime=runtime,
        prompt=prompt,
        system_message=system_message,
    )
    inputs = {k: v.to(runtime.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = runtime.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=runtime.tokenizer.pad_token_id,
            eos_token_id=runtime.tokenizer.eos_token_id,
        )

    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids) and outputs.shape[1] >= input_ids.shape[1]:
        generated_ids = outputs[:, input_ids.shape[1] :]
    else:
        generated_ids = outputs

    generated_tokens = int(generated_ids.shape[1]) if torch.is_tensor(generated_ids) else 0
    decoded = runtime.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = decoded[0].strip() if isinstance(decoded, list) and decoded else str(decoded).strip()
    return text, generated_tokens


def _generate_query_text(
    runtime: QueryRuntime,
    prompt: str,
    max_new_tokens: int,
    system_message: str,
) -> str:
    text, _ = _generate_query_text_with_stats(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        system_message=system_message,
    )
    return text


def _generate_merge_report_with_auto_continue(
    runtime: QueryRuntime,
    prompt: str,
    max_new_tokens: int,
    system_message: str,
    max_continuations: int = 3,
) -> tuple[str, int]:
    merged_text, generated_tokens = _generate_query_text_with_stats(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        system_message=system_message,
    )
    merged_text = merged_text.strip()
    continuation_count = 0

    while continuation_count < max_continuations:
        headers = _extract_headers_from_text(merged_text)
        missing_headers = [h for h in MERGE_REQUIRED_HEADERS if h not in headers]
        likely_hit_limit = generated_tokens >= max(1, max_new_tokens - 4)
        if not missing_headers and not likely_hit_limit:
            break

        followup_prompt = (
            "Continue the same merged report from exactly where it stopped.\n"
            "Do not restart from [Scene]. Do not repeat existing text.\n"
            "If already complete, output exactly: <DONE>\n"
            f"Required section headers: {', '.join(MERGE_REQUIRED_HEADERS)}\n"
            f"Missing headers currently: {', '.join(missing_headers) if missing_headers else '(none)'}\n\n"
            "Current partial report:\n"
            f"{merged_text}\n\n"
            "Continue now:"
        )
        extra_text, generated_tokens = _generate_query_text_with_stats(
            runtime=runtime,
            prompt=followup_prompt,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
        )
        extra_text = extra_text.strip()
        if not extra_text or extra_text == "<DONE>":
            break
        merged_text = _append_non_overlapping_text(merged_text, extra_text)
        continuation_count += 1

    return merged_text, continuation_count


def _looks_like_incomplete_json_output(
    text: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if generated_tokens >= max(1, int(max_new_tokens) - 4):
        return True
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    if stripped.count('"') % 2 == 1:
        return True
    if ("retrieval_queries" in stripped or '"queries"' in stripped) and not stripped.endswith("}"):
        return True
    return False


def _repair_query_json_output(
    runtime: QueryRuntime,
    original_prompt: str,
    partial_output: str,
    max_new_tokens: int,
    max_attempts: int = 2,
) -> str:
    current = (partial_output or "").strip()
    for _ in range(max_attempts):
        repair_prompt = (
            "Your previous output was truncated or malformed.\n"
            "Regenerate ONE complete valid JSON object only.\n"
            "Schema MUST be:\n"
            "{\n"
            '  "retrieval_queries": [\n'
            "    {\n"
            '      "id": "Q1",\n'
            '      "time_hint": "",\n'
            '      "query_type": "",\n'
            '      "query_text": "",\n'
            '      "why_this_query": "",\n'
            '      "needs_external_check": false,\n'
            '      "external_check_focus": "",\n'
            '      "external_check_hint": ""\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Hard rules:\n"
            f"- At most {MAX_RETRIEVAL_QUERIES} queries.\n"
            "- Do NOT use key `queries`; must use `retrieval_queries`.\n"
            "- query_type MUST be one of: ASR, OCR, Visual, Mixed.\n"
            "- Output JSON only; no markdown/code fences/explanations.\n\n"
            "Original task prompt:\n"
            f"{original_prompt}\n\n"
            "Previous partial output:\n"
            f"{current}\n"
        )
        repaired = _generate_query_text(
            runtime=runtime,
            prompt=repair_prompt,
            max_new_tokens=max_new_tokens,
            system_message=(
                "You are a strict JSON fixer. "
                "Return one complete JSON object only."
            ),
        ).strip()
        repaired = _clean_query_model_output(repaired)
        if repaired.startswith("{") and repaired.endswith("}"):
            return repaired
        if repaired:
            current = _append_non_overlapping_text(current, repaired)
    return current


def _extract_event_time_map(report_text: str) -> dict[str, str]:
    """从模块 A 报告中提取事件 ID 到时间段的映射，如 E2 -> [00:03–00:06]。"""
    mapping: dict[str, str] = {}
    for match in _EVENT_TIME_RE.finditer(report_text):
        event_id = match.group(1).upper()
        time_range = match.group(2).strip()
        mapping[event_id] = f"[{time_range}]"
    return mapping


def _resolve_time_hint_from_event_ids(time_hint: str, event_time_map: dict[str, str]) -> str:
    """若 time_hint 为 E2/A1 等事件 ID，则自动映射为时间段。"""
    hint = (time_hint or "").strip()
    if not hint:
        return ""

    # 已含时间戳则直接保留
    if _TIME_TOKEN_RE.search(hint):
        return hint

    ids = [m.upper() for m in _EVENT_ID_RE.findall(hint)]
    resolved = [event_time_map[i] for i in ids if i in event_time_map]
    if resolved:
        return "; ".join(resolved)
    return hint


def build_query_extraction_prompt(report_text: str) -> str:
    """构建模块 C 的提示词（Prompt 2 + 精度约束 + 全局报告）。"""
    return (
        f"{QUERY_EXTRACTION_PROMPT}"
        f"{QUERY_PRECISION_ADDON}"
        f"\nGlobal report:\n{report_text.strip()}"
    )


def _build_query_inputs(
    runtime: QueryRuntime,
    prompt: str,
    system_message: str | None = None,
) -> dict[str, torch.Tensor]:
    """优先使用 chat template 组装 Query 输入，提高 Instruct 模型遵循性。"""
    tokenizer = runtime.tokenizer
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        if system_message is None:
            system_message = (
                "You are a query extraction assistant. "
                "Output JSON only. Never output <think> tags or analysis text."
            )
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer(text, return_tensors="pt")
    return tokenizer(prompt, return_tensors="pt")


def _clean_query_model_output(text: str) -> str:
    """清洗 Query 模型输出：去除 <think> 块与代码围栏，保留 JSON 主体。"""
    cleaned = _THINK_BLOCK_RE.sub("", text or "").strip()
    cleaned = cleaned.replace("```json", "```").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
    return cleaned


def _get_module_device(module: torch.nn.Module) -> torch.device | None:
    """返回模块中第一个非-meta 参数所在设备。"""
    try:
        for param in module.parameters():
            if torch.is_tensor(param) and param.device.type != "meta":
                return param.device
    except Exception:
        pass
    return None


def _get_query_dtype(device: str) -> torch.dtype:
    """根据设备为 Query 文本模型选择 dtype。"""
    if str(device).startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def extract_last_json_object(text: str) -> dict[str, Any]:
    """从文本中提取最后一个合法 JSON 对象，优先返回包含 retrieval_queries 的对象。"""
    decoder = json.JSONDecoder()
    last_obj: dict[str, Any] | None = None
    last_with_queries: dict[str, Any] | None = None

    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            last_obj = obj
            if "retrieval_queries" in obj:
                last_with_queries = obj

    if last_with_queries is not None:
        return last_with_queries
    if last_obj is not None:
        return last_obj
    raise ValueError("No valid JSON object found in query model output.")


def _coalesce_query_value(item: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _is_query_dict(item: Any) -> bool:
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


def _extract_query_list(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    preferred_keys = ["retrieval_queries", "queries", "query_list", "query_items"]
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list) and all(_is_query_dict(x) for x in value):
            return value

    for value in payload.values():
        if isinstance(value, dict):
            nested = _extract_query_list(value)
            if nested is not None:
                return nested

    for value in payload.values():
        if isinstance(value, list) and value and all(_is_query_dict(x) for x in value):
            return value
    return None


def _normalize_query_type(raw_query_type: str) -> str:
    allowed = {"ASR", "OCR", "Visual", "Mixed", "RAG"}
    mapping = {
        "ASR": "ASR",
        "AUDIO": "ASR",
        "OCR": "OCR",
        "TEXT": "OCR",
        "VISUAL": "Visual",
        "VISION": "Visual",
        "MIXED": "Mixed",
        "MULTIMODAL": "Mixed",
        "RAG": "RAG",
    }
    query_type_upper = (raw_query_type or "").strip().upper()
    query_type = mapping.get(query_type_upper, "Visual")
    if query_type not in allowed:
        return "Visual"
    return query_type


def _normalize_single_query_item(
    item: dict[str, Any],
    index: int,
    event_time_map: dict[str, str],
) -> tuple[RetrievalQuery | None, str | None]:
    qid = str(_coalesce_query_value(item, ["id", "query_id"], default=f"Q{index + 1}")).strip() or f"Q{index + 1}"
    raw_time_hint = str(_coalesce_query_value(item, ["time_hint", "time", "phase_hint"], default="")).strip()
    resolved_time_hint = _resolve_time_hint_from_event_ids(raw_time_hint, event_time_map)
    query_text = str(
        _coalesce_query_value(
            item,
            ["query_text", "query", "text", "retrieval_query", "query_content"],
            default="",
        )
    ).strip()
    why_this_query = str(
        _coalesce_query_value(
            item,
            ["why_this_query", "reason", "rationale", "why"],
            default="",
        )
    ).strip()
    query_type = _normalize_query_type(
        str(_coalesce_query_value(item, ["query_type", "type", "route"], default="Visual"))
    )

    if not query_text:
        return None, f"retrieval_queries[{index}] missing usable query_text."

    known_core_keys = {
        "id",
        "query_id",
        "time_hint",
        "time",
        "phase_hint",
        "query_type",
        "type",
        "route",
        "query_text",
        "query",
        "text",
        "retrieval_query",
        "query_content",
        "why_this_query",
        "reason",
        "rationale",
        "why",
    }
    extra_fields = {k: v for k, v in item.items() if k not in known_core_keys}

    extra_fields["needs_external_check"] = _to_bool(item.get("needs_external_check"), default=False)
    extra_fields["external_check_focus"] = str(item.get("external_check_focus", "") or "").strip()
    extra_fields["external_check_hint"] = str(item.get("external_check_hint", "") or "").strip()

    return (
        RetrievalQuery(
            id=qid,
            time_hint=resolved_time_hint,
            query_type=query_type,
            query_text=query_text,
            why_this_query=why_this_query,
            extra_fields=extra_fields,
        ),
        None,
    )


def _query_to_output_dict(query: RetrievalQuery) -> dict[str, Any]:
    return query.to_output_dict()


def _validate_and_normalize_queries(
    payload: dict[str, Any],
    event_time_map: dict[str, str],
) -> tuple[list[RetrievalQuery], list[str]]:
    """兼容不同 Query Schema，仅强约束核心字段并保留额外字段。"""
    errors: list[str] = []
    normalized: list[RetrievalQuery] = []

    queries = _extract_query_list(payload)
    if not isinstance(queries, list):
        return [], ["No valid query list found. Expected key retrieval_queries/queries."]

    for idx, item in enumerate(queries):
        if len(normalized) >= MAX_RETRIEVAL_QUERIES:
            break
        if not isinstance(item, dict):
            errors.append(f"retrieval_queries[{idx}] must be an object.")
            continue

        query, err = _normalize_single_query_item(item=item, index=idx, event_time_map=event_time_map)
        if err:
            errors.append(err)
            continue
        if query is not None:
            normalized.append(query)

    return normalized, errors


def _time_token_to_seconds(token: tuple[str, str]) -> float:
    minutes = int(token[0])
    seconds = int(token[1])
    return minutes * 60 + seconds


def _parse_time_hint_range_seconds(time_hint: str) -> tuple[float, float] | None:
    matches = _TIME_TOKEN_RE.findall(time_hint or "")
    if len(matches) >= 2:
        start = _time_token_to_seconds(matches[0])
        end = _time_token_to_seconds(matches[1])
        if end <= start:
            end = start + 4.0
        return max(0.0, start), end
    if len(matches) == 1:
        start = _time_token_to_seconds(matches[0])
        return max(0.0, start), start + 4.0
    return None


def _extract_audio_segment(video_path: str, start_sec: float, end_sec: float, out_wav: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.2f}",
        "-to",
        f"{end_sec:.2f}",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-300:]
        raise RuntimeError(f"ffmpeg failed: {stderr_tail}")


def _load_omni_runtime(
    model_path: str,
    device: str | None = None,
    device_map: str | None = "auto",
) -> OmniRuntime:
    print("=" * 60)
    print("模块 A：加载模型")
    print("=" * 60)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device is None and n_gpu > 0:
        device = "cuda:0"
    elif device is None:
        device = "cpu"

    model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    effective_device_map = device_map
    model_load_kwargs = {
        "torch_dtype": model_dtype,
        "attn_implementation": "sdpa",
    }
    if effective_device_map not in (None, "none"):
        model_load_kwargs["device_map"] = effective_device_map

    print(f"模型路径: {model_path}")
    print(f"device: {device}")
    print(f"device_map: {effective_device_map}")
    print(f"可见 GPU 数: {n_gpu}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        **model_load_kwargs,
    )

    if effective_device_map in (None, "none"):
        model = model.to(device)
        input_device = torch.device(device)
    else:
        thinker_embed = model.thinker.get_input_embeddings()
        input_device = _get_module_device(thinker_embed) or torch.device(device)
        print(f"[提示] 当前为分片推理，输入张量将放置到 {input_device}")

    model.disable_talker()
    print("已禁用 Talker（仅文本报告输出）")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("模型加载完成")

    return OmniRuntime(
        model=model,
        processor=processor,
        input_device=input_device,
        model_dtype=model_dtype,
        use_audio_in_video=True,
    )


def _load_query_runtime(
    query_model_path: str,
    device: str | None = None,
) -> QueryRuntime:
    print("\n" + "=" * 60)
    print("模块 C：开始加载 Query 模型")
    print("=" * 60)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device is None and n_gpu > 0:
        device = "cuda:0"
    elif device is None:
        device = "cpu"

    query_dtype = _get_query_dtype(device)

    print(f"query_model_path: {query_model_path}")
    print(f"query_device: {device}")
    print(f"query_dtype: {query_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(query_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        query_model_path,
        torch_dtype=query_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 固定为贪心解码，避免采样参数干扰并减少无关“思维过程”输出
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    if hasattr(gen_cfg, "temperature"):
        gen_cfg.temperature = None
    if hasattr(gen_cfg, "top_p"):
        gen_cfg.top_p = None
    if hasattr(gen_cfg, "top_k"):
        gen_cfg.top_k = None

    if "instruct" not in str(query_model_path).lower():
        print("[提示] 当前 query model 可能不是 Instruct 版本，JSON 遵循性可能下降。")

    print("query model 加载完成")
    return QueryRuntime(model=model, tokenizer=tokenizer, device=device)


def _load_asr_runtime(asr_model_path: str, asr_device: str | None = None) -> ASRRuntime:
    print("\n" + "=" * 60)
    print("ASR：开始加载模型")
    print("=" * 60)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if asr_device is None and n_gpu > 0:
        asr_device = "cuda:0"
    elif asr_device is None:
        asr_device = "cpu"

    if str(asr_device).startswith("cuda"):
        device_index = int(str(asr_device).split(":")[1]) if ":" in str(asr_device) else 0
        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=asr_model_path,
            device=device_index,
            torch_dtype=_get_query_dtype(asr_device),
        )
    else:
        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=asr_model_path,
            device=-1,
        )

    print("ASR 模型加载完成")
    return ASRRuntime(asr_pipe=asr_pipe)


def initialize_model_registry(
    omni_model_path: str,
    omni_device: str | None = None,
    omni_device_map: str | None = "auto",
    load_query: bool = False,
    query_model_path: str = "/sda/yuqifan/HFOCUS/Qwen3-4B",
    query_device: str | None = None,
    load_asr: bool = False,
    asr_model_path: str = "/sda/yuqifan/HFOCUS/Qwen-audio",
    asr_device: str | None = None,
) -> ModelRegistry:
    """统一预加载模型：默认可在一个阶段将 Omni / Query / ASR 一起加载，后续可扩展新模型。"""
    print("\n" + "=" * 60)
    print("统一模型预加载开始")
    print("=" * 60)

    registry = ModelRegistry()
    registry.omni = _load_omni_runtime(
        model_path=omni_model_path,
        device=omni_device,
        device_map=omni_device_map,
    )

    if load_query:
        registry.query = _load_query_runtime(
            query_model_path=query_model_path,
            device=query_device,
        )

    if load_asr:
        registry.asr = _load_asr_runtime(
            asr_model_path=asr_model_path,
            asr_device=asr_device,
        )

    print("\n统一模型预加载完成")
    return registry


def run_global_video_understanding(
    video_path: str,
    user_focus: str = "",
    model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    device: str | None = None,
    device_map: str | None = "auto",
    max_new_tokens: int = 1024,
    video_fps: float = 4.0,
    video_min_frames: int = 32,
    video_max_frames: int = 384,
    video_nframes: int | None = None,
    video_start: float = 0.0,
    video_end: float | None = None,
    segment_seconds: float = 20.0,
    segment_overlap: float = 2.0,
    merge_batch_size: int = 6,
    merge_max_new_tokens: int = 1024,
    merge_max_continuations: int = 3,
    merge_model_path: str = "/sda/yuqifan/HFOCUS/Qwen3-4B",
    merge_device: str | None = None,
    query_runtime: QueryRuntime | None = None,
    omni_runtime: OmniRuntime | None = None,
) -> str:
    """模块 A：输入原生视频，返回自然语言全局理解报告。"""

    runtime = omni_runtime
    if runtime is None:
        runtime = _load_omni_runtime(model_path=model_path, device=device, device_map=device_map)
    else:
        print("\n[模块 A] 使用预加载 Omni 模型")

    print("\n" + "=" * 60)
    print("模块 A：构造输入并开始推理")
    print("=" * 60)

    system_prompt = (
        "You are a powerful multimodal assistant specialized in video understanding. "
        "Jointly use visual and audio evidence from the same video. "
        "Pay attention to actions, scene context, temporal order, spoken content, tone, "
        "background sounds, and meaningful acoustic cues. "
        "Do not rely on visual stream alone when audio is informative. "
        "Analyze audio across the full timeline, not only sparse moments. "
        "Do not confuse on-screen text with spoken audio unless actually heard. "
        "Ground conclusions in evidence and clearly distinguish observation from interpretation."
    )

    effective_min_frames = max(1, int(video_min_frames))
    effective_max_frames = max(effective_min_frames, int(video_max_frames))
    effective_fps = max(0.1, float(video_fps))

    def _run_omni_for_range(start_sec: float, end_sec: float | None, is_segment: bool) -> str:
        video_item: dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "video_start": float(start_sec),
        }
        if end_sec is not None:
            video_item["video_end"] = float(end_sec)
        if video_nframes is not None:
            video_item["nframes"] = max(4, int(video_nframes))
        else:
            video_item["fps"] = effective_fps
            video_item["min_frames"] = effective_min_frames
            video_item["max_frames"] = effective_max_frames

        user_prompt = build_global_understanding_prompt(user_focus)
        if is_segment and end_sec is not None:
            user_prompt = (
                f"[Global video range] {_format_hhmmss(full_start)} - {_format_hhmmss(full_end)}\n"
                f"[Current segment] {_format_hhmmss(start_sec)} - {_format_hhmmss(end_sec)}\n"
                "Analyze only current segment but keep chronology awareness.\n\n"
                f"{user_prompt}"
            )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = runtime.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=runtime.use_audio_in_video
        )

        inputs = runtime.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=runtime.use_audio_in_video,
        )
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            value = value.to(runtime.input_device)
            if value.is_floating_point():
                value = value.to(runtime.model_dtype)
            inputs[key] = value

        output_ids = runtime.model.generate(
            **inputs,
            use_audio_in_video=runtime.use_audio_in_video,
            return_audio=False,
            max_new_tokens=max_new_tokens,
        )
        input_ids = inputs.get("input_ids")
        if torch.is_tensor(input_ids) and output_ids.shape[1] >= input_ids.shape[1]:
            generated_ids = output_ids[:, input_ids.shape[1] :]
        else:
            generated_ids = output_ids

        decoded = runtime.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_text = decoded[0].strip() if isinstance(decoded, list) and decoded else str(decoded).strip()
        return _postprocess_report_text(raw_text)

    full_start = float(video_start)
    full_end = float(video_end) if video_end is not None else _get_video_duration_seconds(video_path)
    if full_end <= full_start:
        raise ValueError("video_end must be greater than video_start.")

    print(f"输入视频: {video_path}")
    print(f"音频轨道: 启用 (use_audio_in_video={runtime.use_audio_in_video})")
    print(f"user_focus: {user_focus if user_focus.strip() else '(none)'}")
    print(f"max_new_tokens: {max_new_tokens}")
    if video_nframes is not None:
        print(f"采样配置: nframes={video_nframes}")
    else:
        print(
            f"采样配置: fps={effective_fps}, min_frames={effective_min_frames}, "
            f"max_frames={effective_max_frames}"
        )

    if segment_seconds <= 0:
        print("模式: 单段推理")
        print("开始推理...")
        report_text = _run_omni_for_range(full_start, full_end, is_segment=False)
    else:
        print(
            f"模式: 分段推理+汇总 (segment_seconds={segment_seconds}, "
            f"segment_overlap={segment_overlap})"
        )
        segments = _build_segments(
            video_start=full_start,
            video_end=full_end,
            segment_seconds=segment_seconds,
            segment_overlap=segment_overlap,
        )
        print(f"分段数量: {len(segments)}")

        segment_reports: list[dict[str, Any]] = []
        for idx, (seg_start, seg_end) in enumerate(segments, start=1):
            print(
                f"[Segment {idx}/{len(segments)}] "
                f"{_format_hhmmss(seg_start)}-{_format_hhmmss(seg_end)}"
            )
            seg_report = _run_omni_for_range(seg_start, seg_end, is_segment=True)
            segment_reports.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "report": seg_report,
                }
            )

        if len(segment_reports) == 1:
            report_text = segment_reports[0]["report"]
        else:
            merge_runtime = query_runtime
            if merge_runtime is None:
                merge_runtime = _load_query_runtime(
                    query_model_path=merge_model_path,
                    device=merge_device,
                )
            current = segment_reports
            round_idx = 1
            effective_merge_batch = max(2, int(merge_batch_size))
            while len(current) > 1:
                next_round: list[dict[str, Any]] = []
                for batch in _chunk_list(current, effective_merge_batch):
                    merge_prompt = _build_segment_merge_prompt_text(
                        segment_reports=batch,
                        video_start=full_start,
                        video_end=full_end,
                    )
                    merged, continuation_count = _generate_merge_report_with_auto_continue(
                        runtime=merge_runtime,
                        prompt=merge_prompt,
                        max_new_tokens=merge_max_new_tokens,
                        system_message=(
                            "You are a precise report-merging assistant. "
                            "Output plain text only, no JSON, no <think> tags."
                        ),
                        max_continuations=max(0, int(merge_max_continuations)),
                    )
                    merged = _postprocess_report_text(merged)
                    next_round.append(
                        {
                            "start": batch[0]["start"],
                            "end": batch[-1]["end"],
                            "report": merged,
                        }
                    )
                    print(
                        f"[Merge round {round_idx}] "
                        f"{_format_hhmmss(batch[0]['start'])}-{_format_hhmmss(batch[-1]['end'])}"
                    )
                    if continuation_count > 0:
                        print(f"[Merge continuation] appended {continuation_count} extra generation round(s)")
                current = next_round
                round_idx += 1
            report_text = current[0]["report"]

    print("\n" + "=" * 60)
    print("模块 A：全局理解报告")
    print("=" * 60)
    print(report_text)

    return report_text


def _build_asr_evidence_for_queries(
    queries: list[RetrievalQuery],
    video_path: str,
    asr_runtime: ASRRuntime,
    keep_audio_segments: bool = False,
) -> list[QueryEvidence]:
    evidences: list[QueryEvidence] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="query_asr_"))
    try:
        for query in queries:
            time_range = _parse_time_hint_range_seconds(query.time_hint)
            if time_range is None:
                evidences.append(
                    QueryEvidence(
                        id=query.id,
                        time_hint=query.time_hint,
                        query_text=query.query_text,
                        asr_text="",
                        evidence_note="No valid timestamp found in time_hint; skipped ASR.",
                        audio_segment_path="",
                    )
                )
                continue

            start_sec, end_sec = time_range
            wav_path = tmp_dir / f"{query.id}.wav"

            try:
                _extract_audio_segment(video_path, start_sec, end_sec, wav_path)
                asr_result = asr_runtime.asr_pipe(str(wav_path))
                if isinstance(asr_result, dict):
                    asr_text = str(asr_result.get("text", "")).strip()
                else:
                    asr_text = str(asr_result).strip()

                evidences.append(
                    QueryEvidence(
                        id=query.id,
                        time_hint=query.time_hint,
                        query_text=query.query_text,
                        asr_text=asr_text,
                        evidence_note="ASR extracted from evidence segment.",
                        audio_segment_path=str(wav_path) if keep_audio_segments else "",
                    )
                )
            except Exception as exc:
                evidences.append(
                    QueryEvidence(
                        id=query.id,
                        time_hint=query.time_hint,
                        query_text=query.query_text,
                        asr_text="",
                        evidence_note=f"ASR extraction failed: {exc}",
                        audio_segment_path="",
                    )
                )

            if not keep_audio_segments and wav_path.exists():
                wav_path.unlink()
    finally:
        if not keep_audio_segments and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return evidences


def run_query_extraction(
    report_text: str,
    query_model_path: str,
    device: str | None = None,
    max_new_tokens: int = 1024,
    query_runtime: QueryRuntime | None = None,
    video_path: str | None = None,
    asr_runtime: ASRRuntime | None = None,
    keep_audio_segments: bool = False,
) -> QueryExtractionResult:
    """模块 C：从全局理解报告中提炼检索 Query（JSON 输出）。"""

    runtime = query_runtime
    if runtime is None:
        runtime = _load_query_runtime(query_model_path=query_model_path, device=device)
    else:
        print("\n[模块 C] 使用预加载 Query 模型")

    print("\n" + "=" * 60)
    print("模块 C：开始 Query 提炼")
    print("=" * 60)

    prompt = build_query_extraction_prompt(report_text)
    raw_text, generated_tokens = _generate_query_text_with_stats(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        system_message=(
            "You are a query extraction assistant. "
            "Output one complete valid JSON object only. "
            f"Use key retrieval_queries only and at most {MAX_RETRIEVAL_QUERIES} items."
        ),
    )

    if _looks_like_incomplete_json_output(raw_text, generated_tokens, max_new_tokens):
        print("\n[模块 C] 检测到 query 输出疑似截断/不完整，尝试自动修复...")
        repaired = _repair_query_json_output(
            runtime=runtime,
            original_prompt=prompt,
            partial_output=raw_text,
            max_new_tokens=max_new_tokens,
            max_attempts=2,
        )
        if repaired and repaired.strip() != raw_text.strip():
            raw_text = repaired.strip()

    print("原始 query model 输出:")
    print(raw_text)
    cleaned_text = _clean_query_model_output(raw_text)
    if cleaned_text != raw_text:
        print("\n清洗后的 query model 输出（用于解析）:")
        print(cleaned_text)

    parsed_json: dict[str, Any] | None = None
    retrieval_queries: list[RetrievalQuery] = []
    query_evidence: list[QueryEvidence] = []
    parse_error = False
    parse_error_message: str | None = None

    event_time_map = _extract_event_time_map(report_text)

    try:
        parsed_json = extract_last_json_object(cleaned_text)
        retrieval_queries, validation_errors = _validate_and_normalize_queries(
            parsed_json,
            event_time_map=event_time_map,
        )
        if validation_errors:
            parse_error = True
            parse_error_message = "; ".join(validation_errors)
    except Exception as exc:
        try:
            parsed_json = extract_last_json_object(raw_text)
            retrieval_queries, validation_errors = _validate_and_normalize_queries(
                parsed_json,
                event_time_map=event_time_map,
            )
            if validation_errors:
                parse_error = True
                parse_error_message = "; ".join(validation_errors)
        except Exception:
            parse_error = True
            parse_error_message = str(exc)

    if parse_error and parse_error_message:
        print(f"[模块 C][解析告警] {parse_error_message}")

    # 可选 ASR 证据增强
    if asr_runtime is not None and video_path:
        print("\n模块 C：开始 ASR 证据片段抽取")
        query_evidence = _build_asr_evidence_for_queries(
            queries=retrieval_queries,
            video_path=video_path,
            asr_runtime=asr_runtime,
            keep_audio_segments=keep_audio_segments,
        )

    final_query_json = {"retrieval_queries": [_query_to_output_dict(q) for q in retrieval_queries]}

    print(f"解析后的 query 数量: {len(retrieval_queries)}")
    print("最终 query JSON:")
    print(json.dumps(final_query_json, ensure_ascii=False, indent=2))

    return QueryExtractionResult(
        raw_text=raw_text,
        parsed_json=parsed_json,
        retrieval_queries=retrieval_queries,
        query_evidence=query_evidence,
        parse_error=parse_error,
        parse_error_message=parse_error_message,
    )


def run_global_understanding_and_query_extraction(
    video_path: str,
    user_focus: str = "",
    omni_model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    query_model_path: str = "/sda/yuqifan/HFOCUS/Qwen3-4B",
    omni_device: str | None = None,
    omni_device_map: str | None = "auto",
    query_device: str | None = None,
    omni_max_new_tokens: int = 1024,
    query_max_new_tokens: int = 1024,
    video_fps: float = 4.0,
    video_min_frames: int = 32,
    video_max_frames: int = 384,
    video_nframes: int | None = None,
    video_start: float = 0.0,
    video_end: float | None = None,
    segment_seconds: float = 20.0,
    segment_overlap: float = 2.0,
    merge_batch_size: int = 6,
    merge_max_new_tokens: int = 1024,
    merge_max_continuations: int = 3,
    run_asr_evidence: bool = False,
    asr_model_path: str = "/sda/yuqifan/HFOCUS/Qwen-audio",
    asr_device: str | None = None,
    keep_audio_segments: bool = False,
    run_localizer: bool = False,
    localizer_config: dict[str, Any] | None = None,
    model_registry: ModelRegistry | None = None,
) -> GlobalPipelineResult:
    """总流程：模块 A 生成报告，再由模块 C 提炼 Query。"""

    registry = model_registry
    if registry is None:
        registry = initialize_model_registry(
            omni_model_path=omni_model_path,
            omni_device=omni_device,
            omni_device_map=omni_device_map,
            load_query=True,
            query_model_path=query_model_path,
            query_device=query_device,
            load_asr=run_asr_evidence,
            asr_model_path=asr_model_path,
            asr_device=asr_device,
        )

    report_text = run_global_video_understanding(
        video_path=video_path,
        user_focus=user_focus,
        model_path=omni_model_path,
        device=omni_device,
        device_map=omni_device_map,
        max_new_tokens=omni_max_new_tokens,
        video_fps=video_fps,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
        video_nframes=video_nframes,
        video_start=video_start,
        video_end=video_end,
        segment_seconds=segment_seconds,
        segment_overlap=segment_overlap,
        merge_batch_size=merge_batch_size,
        merge_max_new_tokens=merge_max_new_tokens,
        merge_max_continuations=merge_max_continuations,
        merge_model_path=query_model_path,
        merge_device=query_device,
        query_runtime=registry.query,
        omni_runtime=registry.omni,
    )

    query_result = run_query_extraction(
        report_text=report_text,
        query_model_path=query_model_path,
        device=query_device,
        max_new_tokens=query_max_new_tokens,
        query_runtime=registry.query,
        video_path=video_path,
        asr_runtime=registry.asr,
        keep_audio_segments=keep_audio_segments,
    )

    evidence_results: list[dict[str, Any]] | None = None
    if run_localizer:
        localizer_cfg = localizer_config if localizer_config is not None else {}
        evidence_results = run_query_evidence_localizer(
            video_path=video_path,
            retrieval_queries=query_result.retrieval_queries,
            config=localizer_cfg,
        )

    return GlobalPipelineResult(
        report_text=report_text,
        query_result=query_result,
        evidence_results=evidence_results,
    )


def _resolve_save_path(video_path: str, save_txt: str) -> Path:
    if save_txt == "auto":
        video = Path(video_path)
        return video.with_suffix(".txt")
    return Path(save_txt)


def _resolve_query_json_save_path(video_path: str, save_query_json: str) -> Path:
    if save_query_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".queries.json")
    return Path(save_query_json)


def _resolve_evidence_json_save_path(video_path: str, save_evidence_json: str) -> Path:
    if save_evidence_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".evidence.json")
    return Path(save_evidence_json)


def _resolve_localization_json_save_path(video_path: str, save_localization_json: str) -> Path:
    if save_localization_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".localization.json")
    return Path(save_localization_json)


def _resolve_localizer_frames_dir(video_path: str, localizer_frames_dir: str | None) -> Path:
    if localizer_frames_dir and localizer_frames_dir.strip():
        return Path(localizer_frames_dir)
    video = Path(video_path)
    return video.with_name(f"{video.stem}.localizer_frames")


def _sanitize_token(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _to_uint8_image(frame: Any) -> np.ndarray:
    if hasattr(frame, "asnumpy") and callable(frame.asnumpy):
        image = frame.asnumpy()
    elif hasattr(frame, "numpy") and callable(frame.numpy):
        image = frame.numpy()
    else:
        image = np.asarray(frame)

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size > 0 and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def export_localizer_supporting_frames(
    video_path: str,
    evidence_results: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("导出 localizer 帧需要 decord 和 pillow。") from exc

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    video = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = int(len(video))
    fps = float(video.get_avg_fps()) if total_frames > 0 else 0.0

    manifest_queries: list[dict[str, Any]] = []
    total_exported = 0

    for query_rank, item in enumerate(evidence_results, start=1):
        query_id = str(item.get("query_id", "")).strip() or f"Q{query_rank}"
        query_dir = output_root / f"{query_rank:02d}_{_sanitize_token(query_id, f'Q{query_rank}')}"
        query_dir.mkdir(parents=True, exist_ok=True)

        supporting_frames = item.get("supporting_frames")
        supporting = supporting_frames if isinstance(supporting_frames, list) else []

        exported_frames: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for frame_rank, frame_info in enumerate(supporting, start=1):
            if not isinstance(frame_info, dict):
                continue
            try:
                frame_idx = int(frame_info.get("frame_idx"))
            except (TypeError, ValueError):
                continue
            if frame_idx < 0 or frame_idx >= total_frames or frame_idx in seen_indices:
                continue
            seen_indices.add(frame_idx)

            time_sec = float(frame_info.get("time_sec", frame_idx / max(1e-6, fps)))
            score = float(frame_info.get("score", 0.0))
            file_name = (
                f"{frame_rank:02d}_idx{frame_idx:06d}_"
                f"t{time_sec:07.3f}_s{score:0.4f}.jpg"
            )
            save_path = query_dir / file_name

            image = _to_uint8_image(video[frame_idx])
            Image.fromarray(image).save(save_path, format="JPEG", quality=95)

            exported_frames.append(
                {
                    "frame_rank": frame_rank,
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "score": score,
                    "image_path": str(save_path),
                }
            )

        query_payload = dict(item)
        query_payload["exported_frames"] = exported_frames
        (query_dir / "query_localization.json").write_text(
            json.dumps(query_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_exported += len(exported_frames)
        manifest_queries.append(
            {
                "query_rank": query_rank,
                "query_id": query_id,
                "query_dir": str(query_dir),
                "exported_frame_count": len(exported_frames),
            }
        )

    manifest = {
        "video_path": video_path,
        "output_dir": str(output_root),
        "query_count": len(evidence_results),
        "exported_frame_count": total_exported,
        "queries": manifest_queries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_localizer_config(
    blip_model: str,
    device: str | None,
    batch_size: int,
    config_json: str | None,
) -> dict[str, Any]:
    from focus_localizer import default_config as focus_localizer_default_config

    cfg: dict[str, Any] = dict(focus_localizer_default_config)
    cfg["blip_model"] = blip_model
    cfg["batch_size"] = int(batch_size)
    if device:
        cfg["device"] = device

    if config_json:
        parsed = json.loads(config_json)
        if not isinstance(parsed, dict):
            raise ValueError("--localizer_config_json must be a JSON object")
        cfg.update(parsed)

    return cfg


def run_query_evidence_localizer(
    video_path: str,
    retrieval_queries: list[RetrievalQuery] | list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from focus_localizer import localize_all_queries

    normalized_queries: list[dict[str, Any]] = []
    for q in retrieval_queries:
        if isinstance(q, RetrievalQuery):
            normalized_queries.append(_query_to_output_dict(q))
        elif isinstance(q, dict):
            normalized_queries.append(q)

    return localize_all_queries(
        video_path=video_path,
        retrieval_queries=normalized_queries,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 模块 A/C 推理脚本")

    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--focus", type=str, default="", help="可选用户关注点")

    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni", help="Omni 模型路径")
    parser.add_argument("--device", type=str, default=None, help="Omni 运行设备，如 cuda:0 / cpu")
    parser.add_argument("--device_map", type=str, default="auto", help="Omni HF device_map，默认 auto")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="模块 A 最大生成 token 数")
    parser.add_argument("--fps", type=float, default=4.0, help="视频采样 fps（提升采样默认 4.0）")
    parser.add_argument("--min_frames", type=int, default=32, help="最小采样帧数")
    parser.add_argument("--max_frames", type=int, default=384, help="最大采样帧数")
    parser.add_argument("--nframes", type=int, default=None, help="固定采样帧数（优先于 fps/min/max）")
    parser.add_argument("--video_start", type=float, default=0.0, help="视频起始秒")
    parser.add_argument("--video_end", type=float, default=None, help="视频结束秒，不填则自动探测")
    parser.add_argument("--segment_seconds", type=float, default=20.0, help="分段秒数；<=0 表示不分段")
    parser.add_argument("--segment_overlap", type=float, default=2.0, help="分段重叠秒数")
    parser.add_argument("--merge_batch_size", type=int, default=6, help="分段汇总每轮批大小")
    parser.add_argument("--merge_max_new_tokens", type=int, default=1024, help="分段汇总模型最大生成 token 数")
    parser.add_argument("--merge_max_continuations", type=int, default=3, help="分段汇总自动续写最大轮数")

    parser.add_argument("--run_query_extraction", action="store_true", help="启用模块 C：Query 提炼")
    parser.add_argument("--query_model", type=str, default="/sda/yuqifan/HFOCUS/Qwen3-4B", help="Query 文本模型路径")
    parser.add_argument("--query_device", type=str, default=None, help="Query 模型运行设备，如 cuda:0 / cpu")
    parser.add_argument("--query_max_new_tokens", type=int, default=1024, help="模块 C 最大生成 token 数")

    parser.add_argument("--run_asr_evidence", action="store_true", help="为每条 query 抽取时间片段并做 ASR")
    parser.add_argument("--asr_model", type=str, default="/sda/yuqifan/HFOCUS/Qwen-audio", help="ASR 模型路径")
    parser.add_argument("--asr_device", type=str, default=None, help="ASR 运行设备，如 cuda:0 / cpu")
    parser.add_argument("--keep_audio_segments", action="store_true", help="保留抽取的音频片段文件")
    parser.add_argument("--run_localizer", action="store_true", help="启用 query-guided 局部证据定位（FOCUS-localizer）")
    parser.add_argument("--localizer_blip_model", type=str, default="large", help="localizer 使用的 BLIP ITM 模型（base/large）")
    parser.add_argument("--localizer_device", type=str, default=None, help="localizer 推理设备，如 cuda:0 / cpu")
    parser.add_argument("--localizer_batch_size", type=int, default=32, help="localizer BLIP 批大小")
    parser.add_argument(
        "--localizer_frames_dir",
        type=str,
        default=None,
        help="localizer 选中帧导出目录。默认自动保存为同目录 <视频名>.localizer_frames",
    )
    parser.add_argument(
        "--localizer_config_json",
        type=str,
        default=None,
        help="localizer 额外配置（JSON 对象字符串），用于覆盖默认短视频参数",
    )

    parser.add_argument("--disable_preload", action="store_true", help="禁用统一模型预加载（默认会在 A+C 模式预加载）")

    parser.add_argument(
        "--save_txt",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存模块 A 报告文本。仅写 --save_txt 时自动保存为同名 .txt；也可指定输出路径",
    )
    parser.add_argument(
        "--save_query_json",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存模块 C Query JSON。仅写 --save_query_json 时自动保存为同名 .queries.json；也可指定输出路径",
    )
    parser.add_argument(
        "--save_evidence_json",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存 ASR 证据 JSON。仅写 --save_evidence_json 时自动保存为同名 .evidence.json；也可指定输出路径",
    )
    parser.add_argument(
        "--save_localization_json",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存 localizer 结果 JSON。仅写 --save_localization_json 时自动保存为同名 .localization.json；也可指定输出路径",
    )

    args = parser.parse_args()

    model_registry: ModelRegistry | None = None
    if args.run_query_extraction and not args.disable_preload:
        model_registry = initialize_model_registry(
            omni_model_path=args.model,
            omni_device=args.device,
            omni_device_map=args.device_map,
            load_query=True,
            query_model_path=args.query_model,
            query_device=args.query_device,
            load_asr=args.run_asr_evidence,
            asr_model_path=args.asr_model,
            asr_device=args.asr_device,
        )

    report = run_global_video_understanding(
        video_path=args.video,
        user_focus=args.focus,
        model_path=args.model,
        device=args.device,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        video_fps=args.fps,
        video_min_frames=args.min_frames,
        video_max_frames=args.max_frames,
        video_nframes=args.nframes,
        video_start=args.video_start,
        video_end=args.video_end,
        segment_seconds=args.segment_seconds,
        segment_overlap=args.segment_overlap,
        merge_batch_size=args.merge_batch_size,
        merge_max_new_tokens=args.merge_max_new_tokens,
        merge_max_continuations=args.merge_max_continuations,
        merge_model_path=args.query_model,
        merge_device=args.query_device,
        query_runtime=model_registry.query if model_registry else None,
        omni_runtime=model_registry.omni if model_registry else None,
    )

    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)
    print(report)

    if args.save_txt is not None:
        save_path = _resolve_save_path(args.video, args.save_txt)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report, encoding="utf-8")
        print(f"\n报告已保存: {save_path}")

    if args.run_query_extraction:
        query_result = run_query_extraction(
            report_text=report,
            query_model_path=args.query_model,
            device=args.query_device,
            max_new_tokens=args.query_max_new_tokens,
            query_runtime=model_registry.query if model_registry else None,
            video_path=args.video,
            asr_runtime=(model_registry.asr if model_registry else (_load_asr_runtime(args.asr_model, args.asr_device) if args.run_asr_evidence else None)),
            keep_audio_segments=args.keep_audio_segments,
        )

        query_json_payload = {"retrieval_queries": [_query_to_output_dict(q) for q in query_result.retrieval_queries]}
        print("\n" + "=" * 60)
        print("最终 Query 提炼结果")
        print("=" * 60)
        print(json.dumps(query_json_payload, ensure_ascii=False, indent=2))

        if args.save_query_json is not None:
            query_save_path = _resolve_query_json_save_path(args.video, args.save_query_json)
            query_save_path.parent.mkdir(parents=True, exist_ok=True)
            query_save_path.write_text(
                json.dumps(query_json_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nQuery JSON 已保存: {query_save_path}")

        if args.run_localizer:
            localizer_cfg = _build_localizer_config(
                blip_model=args.localizer_blip_model,
                device=args.localizer_device,
                batch_size=args.localizer_batch_size,
                config_json=args.localizer_config_json,
            )
            evidence_results = run_query_evidence_localizer(
                video_path=args.video,
                retrieval_queries=query_result.retrieval_queries,
                config=localizer_cfg,
            )
            localization_payload = {"evidence_results": evidence_results}

            print("\n" + "=" * 60)
            print("局部证据定位结果（FOCUS-localizer）")
            print("=" * 60)
            print(json.dumps(localization_payload, ensure_ascii=False, indent=2))

            if args.save_localization_json is not None:
                localization_save_path = _resolve_localization_json_save_path(args.video, args.save_localization_json)
                localization_save_path.parent.mkdir(parents=True, exist_ok=True)
                localization_save_path.write_text(
                    json.dumps(localization_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"\n局部证据定位 JSON 已保存: {localization_save_path}")

            localizer_frames_dir = _resolve_localizer_frames_dir(args.video, args.localizer_frames_dir)
            try:
                frame_manifest = export_localizer_supporting_frames(
                    video_path=args.video,
                    evidence_results=evidence_results,
                    output_dir=localizer_frames_dir,
                )
                print(f"\n局部证据帧已导出: {localizer_frames_dir}")
                print(f"导出帧总数: {frame_manifest['exported_frame_count']}")
            except Exception as exc:
                print(f"\n[提示] 局部证据帧导出失败: {exc}")
        elif args.save_localization_json is not None:
            print("\n[提示] 已指定 --save_localization_json，但未启用 --run_localizer，跳过保存。")

        if args.run_asr_evidence:
            evidence_payload = {"query_evidence": [asdict(e) for e in query_result.query_evidence]}
            print("\n" + "=" * 60)
            print("ASR 证据结果")
            print("=" * 60)
            print(json.dumps(evidence_payload, ensure_ascii=False, indent=2))

            if args.save_evidence_json is not None:
                evidence_save_path = _resolve_evidence_json_save_path(args.video, args.save_evidence_json)
                evidence_save_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_save_path.write_text(
                    json.dumps(evidence_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"\nASR 证据 JSON 已保存: {evidence_save_path}")
    else:
        if args.save_query_json is not None:
            print("\n[提示] 已指定 --save_query_json，但未启用 --run_query_extraction，跳过保存。")
        if args.run_localizer:
            print("\n[提示] 已启用 --run_localizer，但未启用 --run_query_extraction，localizer 不会执行。")
        if args.save_localization_json is not None:
            print("\n[提示] 已指定 --save_localization_json，但未启用 --run_query_extraction，跳过保存。")
        if args.run_asr_evidence:
            print("\n[提示] 已启用 --run_asr_evidence，但未启用 --run_query_extraction，ASR 不会执行。")
        if args.save_evidence_json is not None:
            print("\n[提示] 已指定 --save_evidence_json，但未启用 --run_query_extraction，跳过保存。")
