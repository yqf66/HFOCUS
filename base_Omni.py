"""
Qwen2.5-Omni 模块 A：全局视频理解（原生视频 + 音频）
用途：输入原生视频，输出自然语言全局理解报告（非 JSON）
并可选衔接模块 C：Query 提炼（文本小模型）
"""

import argparse
import gc
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info
from query_utils import coalesce_query_value, extract_query_list, normalize_query_type, to_bool


MAX_RETRIEVAL_QUERIES = 5
DEFAULT_LOCALIZER_WHISPER_MODEL_PATH = "/sda/yuqifan/HFOCUS/Whisper/large-v3.pt"
DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH = "/sda/yuqifan/HFOCUS/Qwen3-VL"
DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH = "/sda/yuqifan/HFOCUS/Qwen3-4B"
SMART_OMNI_MEMORY_UTILIZATION = 0.90
SMART_OMNI_MEMORY_RESERVE_GIB = 2.0
SMART_OMNI_MIN_GPU_BUDGET_GIB = 6.0
OMNI_OOM_MAX_RETRIES = 2
OMNI_OOM_FRAME_SCALE = 0.75
OMNI_OOM_FPS_SCALE = 0.85
OMNI_OOM_TOKEN_SCALE = 0.8


GLOBAL_UNDERSTANDING_PROMPT = """
You are a multimodal video evidence analyst.

Watch the video and produce a structured global understanding report
optimized for downstream evidence retrieval and verification.

Do NOT output JSON.
Follow the exact format below strictly, as it will be parsed by later modules.

=====================
Core Goal:
=====================

Your job is NOT to make a final judgment.
Your job is to build a careful, retrieval-oriented global understanding report that:
1. summarizes the full video coherently,
2. preserves important uncertainty,
3. surfaces decision-relevant competing interpretations,
4. identifies the highest-value points that need later verification.

=====================
Key Requirements:
=====================

1. Use both visual and audio evidence jointly.

2. Separate observation and interpretation strictly:
   - Use [Observed] only for directly visible / audible evidence.
   - Use [Inferred] only for interpretation, explanation, or hypothesis.
   - Do NOT mix them in the same bullet.

3. Assign each key visual event an Event ID: E1, E2, E3...
   - Use a small number of salient events rather than many trivial ones.
   - Prefer events that matter for later verification, ambiguity resolution, or final judgment.

4. Do not overcommit to a single interpretation when important context is missing.
   - If the same observed event could support multiple meaningful readings, keep the main interpretation cautious.
   - Explicitly surface a genuinely competing explanation when appropriate.

5. [CRITICAL AUDIO-VISUAL FIREWALL]:
   - Do NOT treat on-screen text as spoken audio.
   - If text appears visually but is not audibly spoken by a real voice, it MUST NOT appear in [Speech Transcript].
   - Do NOT use visual context to fill in missing spoken words.
   - Do NOT convert subtitles, captions, slogans, OCR text, or memes into speech.
   
6. [ON-SCREEN TEXT EXTRACTION]:
   - Actively scan for any visible on-screen text across the whole video.
   - This includes subtitles, captions, overlays, memes, UI text, signs, posters, banners, labels, logos with readable words, printed text on clothing/objects, and other readable visual text.
   - Report text that may help later retrieval, verification, interpretation, or final judgment.
   - Quote the exact visible wording when readable.
   - If partially readable, quote only the reliable part and mark the rest as [unclear].
   - Do NOT treat on-screen text as spoken audio unless it is actually heard.
   - If no visible text is observed, write: No On-Screen Text Observed.

7. [MANDATORY ENTITY EXTRACTION]:
   You must actively scan for and explicitly document high-value visible entities across domains.
   Pay special attention to:
   - National, regional, or international flags and emblems.
   - Religious, ideological, political, military, or organizational symbols / insignias.
   - Protest signs, slogans, campaign merchandise, distinct patches, uniforms, badges, armbands, salutes, portraits, or banners.
   - Recognizable public figures, leader lookalikes, impersonations, parodies, or heavy imitation makeup / costume.
   - If a person appears to imitate or resemble a real public figure, BUT you cannot identify them confidently, label them as [Unidentified Key Figure] and describe the visible cues neutrally.
   - If identity or symbol significance is uncertain but potentially important, still report it neutrally rather than omitting it.

8. Very important for speech transcription:
   - Transcribe as much spoken wording as you can across the whole video.
   - Prioritize fuller transcription over short fragments when speech is reasonably audible.
   - Do NOT paraphrase.
   - Do NOT summarize speech inside the transcript section.
   - Do NOT use speaker labels or speaker attribution.
   - If a short part is uncertain, omit only that small part and keep the surrounding audible words.
   - Use `none reliable` only if there is truly no usable spoken content anywhere in the video.

9. For [Scene] and [Full Video Narrative], stay cautious and evidence-grounded.
   - Do NOT assign strong moral, legal, or political roles unless directly supported.
   - Do NOT smooth over major ambiguities just to make the video story more coherent.

10. [Alternative Interpretation] must be genuinely decision-relevant.
   - It must present a competing explanation that could materially change downstream verification or final judgment.
   - Do NOT merely restate the main interpretation in weaker or vaguer wording.

11. [Key Unclear Points] must prioritize high-impact uncertainty.
   - Focus on uncertainties that most affect later retrieval, verification, classification, or answer quality.
   - Prefer unclear points that distinguish the main interpretation from the alternative interpretation.
   - Each unclear point should identify what is missing and what kind of evidence would help resolve it.
   - Unclear points may concern a single event, multiple events, cross-modal relations, or broader scene-level ambiguity.
   - When helpful, you may mention which observed evidence the uncertainty is most related to, but do not force every unclear point to attach to a single event.
   

=====================
Output Format:
=====================

[Scene]
- setting: describe the visible environment or setting neutrally
- people: describe the visible people or characters neutrally. For each important person / character, include stable visual attributes useful for later retrieval, such as clothing, accessories, hairstyle, facial hair, apparent age group, apparent skin tone / perceived ethnicity when visually inferable, and other distinctive visible cues. Use cautious wording when uncertain.
- roles: cautiously describe likely social or interaction roles based on visible evidence; avoid strong moral, legal, or political role assignments unless directly supported
- high-value entities: [List any visible flags, logos, patches, symbols, slogans, salutes, portraits, uniforms, banners, or recognizable / possibly recognizable figures based on Requirement 8. For each item, include a short category tag such as [symbol], [identity-candidate], [organization], [slogan], [uniform-marker], [unknown], plus a neutral visible description. Write "None" if none are observed.]
- context: a cautious high-level situational context based only on supported evidence

[Full Video Narrative]
- one concise paragraph describing the overall video flow in a coherent but cautious way
- focus on the most important interactions, transitions, and evidence-relevant developments
- do not smooth over major ambiguities just to make the narrative more complete

[Observed Visual Events]

E1:
- [Observed] ...

E2:
- [Observed] ...

E3:
- [Observed] ...

[Speech Transcript]
- "..."
- "..."
- include spoken wording only
- do NOT paraphrase
- if a short part is uncertain, omit only that small part and keep the rest
- if no reliable speech is audible anywhere, write: none reliable

[On-screen Text]
- ...
- ...
- If no visible on-screen text is observed, write: No On-Screen Text Observed

[Main Interpretation]
- [Inferred] the most likely explanation of what is happening, written cautiously and grounded in the observed evidence

[Alternative Interpretation]
- [Inferred] a genuinely competing explanation that could change downstream verification or final judgment
- [Inferred] when possible, indicate what kind of missing evidence would help distinguish it from the main interpretation

[Key Unclear Points]

U1:
- what is unclear:
- why it matters:
- what needs verification:

U2:
- what is unclear:
- why it matters:
- what needs verification:

[Preliminary Conclusion]
- [Inferred] a cautious summary of the current best understanding without making a final categorical judgment
- confidence: low / medium / high
"""


QUERY_EXTRACTION_PROMPT = """You are a query extraction assistant for multimodal video evidence verification.

Input:
A natural-language global video understanding report written by a multimodal model.

Your task:
Extract 1 to 5 high-value queries for downstream evidence verification. Output JSON ONLY.
Each query should be easy for a retrieval system to route to the correct segment.

=====================
[CRITICAL] Downstream Architecture Constraint & Doubt Translation
=====================
The downstream retrieval module uses Text-to-Image similarity FIRST to find the relevant video segment, then applies ASR if needed.
THEREFORE, EVERY QUERY (including ASR and Visual) MUST CONTAIN STRONG VISUAL ANCHORS.

Furthermore, you must TRANSLATE ABSTRACT DOUBTS INTO VISIBLE TESTS. Do not query invisible motives; query the visible evidence that proves them.
- Bad Doubt Query: "is the interaction hostile or friendly" (Abstract, invisible).
- Good Doubt Query: "facial expressions and body posture during the close interaction" (Visible test for hostility).
- Bad ASR Query: "what is being said" (No visual anchor).
- Good ASR Query: "spoken words during podium scene with person in bright jacket" (Visual anchor allows segment retrieval).

When converting doubts into queries, think in this order:
1. What is the exact doubt?
2. What visible/audio evidence would resolve it?
3. What short retrieval phrase would most reliably find that evidence?

=====================
[MANDATORY] Prioritization: Targeting the "Critical Unknowns"
=====================
1. RESOLVE THE AMBIGUITY (Top Priority): Your primary objective is to verify the video's core discrepancies. You MUST prioritize generating queries that target:
   - Specific gaps listed in the report's `[Key Unclear Points]`.
   - The exact moments where the `[Alternative Interpretation]` forks from the `[Main Interpretation]`.
   - Contradictions (e.g., aggressive gestures but calm audio).
2. HIGH-VALUE TARGETS: You MUST create queries targeting any items listed in the report's "high-value entities" (e.g., specific flags, patches, symbols, parodies of leaders). Do not ignore them.
3. ASR QUERIES SHOULD FOLLOW THE TRANSCRIPT: If the report contains spoken wording, generate ASR queries that help retrieve the scene for that wording. Do not depend on speaker attribution.
4. EXTERNAL-CHECK CANDIDATES: If the report describes a potentially meaningful identity, impersonation, parody, symbol, slogan, patch, logo, organization, salute, or historical reference, prefer a neutral retrieval query plus `needs_external_check=true`.
5. MERGE OVERLAPS: Do not output multiple queries targeting the same visual event, text, or scene. Combine them into one comprehensive query.

=====================
[CRITICAL] External Check Hard Threshold (DEFAULT: FALSE)
=====================
By default, `needs_external_check` MUST BE `false`. 
Do NOT abuse this flag. It is an expensive operation reserved ONLY for real-world ideological, political, or historical entities.

[DO NOT TRIGGER] (Keep false):
- Generic speech, dialogue, or arguments.
- Generic people, jobs, or situational roles .
- Generic clothing, masks, or weapons .
- Clarifying the immediate plot or events in the video.

[ONLY TRIGGER] (Set true) IF explicitly mentioned in the report:
1. Recognizable public figures or their parodies .
2. National, regional, or religious flags/symbols.
3. Political logos, slogans, or patches.
4. Specific named organizations or historical references.
5. [Unidentified Key Figure], leader lookalikes, impersonations, or other clearly salient but not yet identified real-world figure candidates.
6. Distinctive gestures/salutes/portraits/chants that may carry ideological or historical meaning.

If `needs_external_check` is true, `external_check_focus` MUST be exactly ONE of the following (Do NOT invent new categories like "role"): 
identity, symbol, slogan, phrase, historical_reference, organization, gesture, ideology_marker, unknown.

If `needs_external_check` is false, `external_check_focus` and `external_check_hint` MUST be exactly "".

=====================
Query Design Rules (Low-Assumption & Retrieval-Friendly)
=====================
1. query_text is for EVIDENCE RETRIEVAL, not for asking questions. Do NOT use QA wording ("what is he saying").
2. LOW-ASSUMPTION: Minimize interpretive assumptions. Use observable anchors (e.g., "person in dark jacket raising an object") over inferred roles or motives.
3. DO NOT hard-code uncertain interpretations into query_text. Keep them neutral.
4. query_type MUST be one of: ASR, Visual. (Do not use Mixed).
   - ASR: For spoken words. Must describe the visible scene, speaker appearance, or interaction context, but do not rely on speaker identity.
   - Visual: For people, clothing, impersonations, symbols, actions, objects, and text-like visual cues (titles, subtitles, slogans, logos, fonts, overlays).
5. query_text should usually contain:
   main target + strongest visible anchor.
6. If a query resolves a `[Key Unclear Points]` item, make the query target the observable test, not the final interpretation.
7. Prefer queries about imitation makeup, face resemblance, costume resemblance, portraits, symbols, emblems, and patches when they may change the interpretation.

=====================
Output Format
=====================
Output valid JSON only. The first non-space character must be "{" and the last must be "}".

{
  "retrieval_queries": [
    {
      "id": "Q1",
      "time_hint": "",
      "query_type": "ASR | Visual",
      "query_text": "Short, keyword-rich, retrieval-friendly phrase packed with visual anchors",
      "why_this_query": "One short sentence explaining what needs verification or interpretation.",
      "needs_external_check": true,
      "external_check_focus": "symbol",
      "external_check_hint": "Verify the meaning or identity significance of the visible symbol, figure, or slogan."
    }
  ]
}
"""

QUERY_PRECISION_ADDON = """\
Additional precision constraints for this run:

- Set time_hint to "".
- query_text must stay short, concrete, and easy to route downstream.
- query_type must be one of:
  ASR / Visual
- Do not use Mixed.

=====================
Retrieval-first principle
=====================

- query_text is for evidence retrieval, not for final interpretation.
- The main purpose of query_text is to help downstream modules find the correct candidate segment.
- Do not write query_text as a broad question about meaning, motive, ideology, or final judgment.
- If interpretation is needed, place it in why_this_query, not in query_text.
- Prefer wording that still works even if the upstream global interpretation is partially wrong.
- Prefer noun/action phrases over full sentences.

=====================
Low-assumption rule
=====================

- Minimize interpretive assumptions in query_text.
- Prefer observable anchors over inferred roles, motives, or event labels.
- If two phrasings are possible, choose the one with fewer assumptions.
- Do not hard-code uncertain interpretations into query_text.
- Prefer observable wording such as:
  person in dark jacket,
  person wearing face covering,
  person at podium,
  uniformed person,
  raising object,
  entering scene,
  removing head covering,
  speaking near stage,
  exchange between two people

=====================
ASR query rule
=====================

- If the report contains [Speech Transcript], use its wording to decide what spoken content needs retrieval.
- ASR query_text should usually look like:
  "spoken words during [visible scene anchor]"
  not:
  "what does he say about ..."
- Do not rely on speaker names or speaker attribution.

=====================
ASR query constraints
=====================

- For ASR queries:
  - the downstream pipeline will first retrieve the relevant visual segment and return a coarse time span,
    then extract the corresponding audio clip for recognition
  - therefore ASR queries must be segment-localization-friendly, not just answer-seeking
  - include where the speech happens or what visible interaction anchors the speech segment
  - ASR queries should help locate a candidate clip before transcription
  - prefer concise forms such as:
    "spoken words during podium scene with person in bright jacket"
    "spoken words during close interaction between two people"
    "spoken words during vehicle-side confrontation"
    "spoken words during crowd scene with central figure"
  - avoid vague QA-style wording such as:
    "what is he saying"
    "what do they mean"
  - avoid putting disputed interpretation directly into query_text

=====================
Text-in-visual constraints
=====================

- For text content visible on screen (title, subtitle, caption, slogan, banner text, logo text, stylized font):
  - use query_type=Visual
  - prefer exact text phrases if explicitly present in the report
  - if exact text is uncertain, use scene, object, or region anchors to localize the right frame
  - prefer retrieval-friendly forms such as:
    "headline text on screen during podium scene"
    "banner text above stage with crowd below"
    "caption text in lower third during close interaction"

=====================
Visual query constraints
=====================

- For Visual queries:
  - prefer distinctive visible anchors such as clothing, face resemblance, makeup, symbol, flag, patch, object, gesture, motion, or scene composition
  - prefer directly observable actions over inferred event labels
  - use compact descriptions tied to visible evidence
  - strongly prioritize figure-identification cues, imitation makeup/costume, portraits, emblems, flags, insignia, and patches when present

=====================
External-check constraints
=====================

- needs_external_check may be true for any query_type.
- Set needs_external_check = true only when the retrieved evidence may reasonably benefit from external background lookup.
- Strong triggers for external checking include:
  named people,
  uncertain but potentially recognizable figures,
  impersonations or lookalikes of real figures,
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
- external_check_hint should explain exactly what to verify externally, for example:
  "Identify the person or public-figure lookalike shown here."
  "Verify the meaning of the visible flag, emblem, patch, or slogan."
- If needs_external_check is false, external_check_focus and external_check_hint must both be empty strings.

=====================
Style constraints
=====================

- Keep why_this_query short and evidence-oriented.
- query_text should usually be a short retrieval phrase, not a natural-language full question.
- Prefer one concrete target per query.
- Do not output duplicate or near-duplicate queries unless they clearly serve different evidence targets.
- Prefer fewer, higher-value queries over many overlapping ones.
- Output JSON only.
- The first non-space character must be "{"
- The last non-space character must be "}"
"""


SEGMENT_MERGE_PROMPT = """You are a multimodal video evidence synthesis assistant.

You will receive multiple overlapping segment reports generated from the same video.
Your task is to merge them into ONE structured full-video report optimized for downstream evidence retrieval and verification.

Do NOT output JSON.
Follow the exact format below strictly, as it will be parsed by later modules.

=====================
Core Goal:
=====================

Your job is NOT to make a final judgment.
Your job is to build a careful, retrieval-oriented merged report that:
1. preserves the important evidence from all segment reports,
2. deduplicates overlap without dropping rare but important details,
3. keeps important uncertainty explicit,
4. surfaces decision-relevant competing interpretations,
5. retains the highest-value points for later verification.

=====================
Key Requirements:
=====================

1. Use both visual and audio evidence jointly.

2. Separate observation and interpretation strictly:
   - Use [Observed] only for directly visible / audible evidence.
   - Use [Inferred] only for interpretation, explanation, or hypothesis.
   - Do NOT mix them in the same bullet.

3. Assign each key merged visual event an Event ID: E1, E2, E3...
   - Do NOT omit an event if it carries important uncertainty, ambiguity, or competing interpretation.
   - Prefer salient events that matter for later verification, uncertainty identification, or competing interpretations.
   - When overlapping segment reports describe the same event, merge them carefully.
   - When reports may describe different events, do NOT collapse them prematurely.

4. Do not overcommit to a single interpretation when important context is missing.
   - If the same observed evidence could support multiple meaningful readings, keep the main interpretation cautious.
   - Explicitly surface a genuinely competing explanation when appropriate.

5. [MANDATORY ENTITY PRESERVATION]:
   You must actively preserve and merge high-value visible entities across segment reports.
   Pay special attention to:
   - National, regional, or international flags and emblems.
   - Religious, ideological, political, military, or organizational symbols / insignias.
   - Protest signs, slogans, campaign merchandise, distinct patches, uniforms, badges, armbands, salutes, portraits, or banners.
   - Recognizable public figures, leader lookalikes, impersonations, parodies, or heavy imitation makeup / costume.
   - If a person appears to imitate or resemble a real public figure, BUT identification is uncertain, label them as [Unidentified Key Figure] and describe the visible cues neutrally.
   - If identity or symbol significance is uncertain but potentially important, still preserve it neutrally rather than omitting it.

6. Very important for speech transcription:
   - Preserve as much spoken wording as possible across all segments.
   - Prioritize fuller transcription over short fragments when speech is reasonably audible.
   - Do NOT paraphrase.
   - Do NOT summarize speech inside the transcript section.
   - Do NOT use speaker labels or speaker attribution.
   - If a short part is uncertain, omit only that small part and keep the surrounding audible words.
   - Use `none reliable` only if there is truly no usable spoken content anywhere in the full video.

7. [SPECIAL MERGE RULE: Speech Transcript]
   - Treat [Speech Transcript] as evidence-preservation-first.
   - Prefer direct concatenation / retention of transcript lines from segment reports rather than semantic rewriting.
   - Remove only clear duplicates caused by overlap.
   - If two lines are similar but not clearly identical, keep both rather than risk losing information.
   - Do NOT compress multiple transcript lines into a summary sentence.

8. For [Scene] and [Full Video Narrative], stay cautious and evidence-grounded.
   - Do NOT assign strong moral, legal, or political roles unless directly supported.
   - Do NOT smooth over major ambiguities just to make the video story more coherent.

9. [Alternative Interpretation] must be genuinely decision-relevant.
   - It must present a competing explanation that could materially change downstream verification or final judgment.
   - Do NOT merely restate the main interpretation in weaker or vaguer wording.

10. [SPECIAL MERGE RULE: Key Unclear Points]
   - Treat [Key Unclear Points] as uncertainty-preservation-first.
   - Prefer direct retention / light consolidation of unclear points from segment reports rather than aggressive semantic merging.
   - If two unclear points appear related but target different events, evidence types, or verification needs, keep both.
   - Do NOT remove an unclear point just because another point sounds more general.
   - Only merge items when they are clearly the same uncertainty.

11. When segment reports conflict:
   - keep both supported possibilities if the conflict cannot be resolved from the available reports,
   - mark uncertainty clearly,
   - do NOT force a single clean story at the cost of losing evidence.

=====================
Output Format:
=====================

[Scene]
- setting: describe the visible environment or setting neutrally
- people: describe the visible people or characters neutrally
- roles: cautiously describe likely social or interaction roles based on visible evidence; avoid strong moral, legal, or political role assignments unless directly supported
- high-value entities: [List any visible flags, logos, patches, symbols, slogans, salutes, portraits, uniforms, banners, or recognizable / possibly recognizable figures based on Requirement 8. For each item, include a short category tag such as [symbol], [identity-candidate], [organization], [slogan], [uniform-marker], [unknown], plus a neutral visible description. Write "None" if none are observed.]
- context: a cautious high-level situational context based only on supported evidence

[Full Video Narrative]
- one concise paragraph describing the overall video flow in a coherent but cautious way
- focus on the most important interactions, transitions, and evidence-relevant developments
- do not smooth over major ambiguities just to make the narrative more complete

[Observed Visual Events]

E1:
- [Observed] ...

E2:
- [Observed] ...

E3:
- [Observed] ...

[Speech Transcript]
- "..."
- "..."
- include spoken wording only
- do NOT paraphrase
- do NOT add speaker labels
- do NOT add timestamps
- if a short part is uncertain, omit only that small part and keep the rest
- preserve overlapping segment evidence conservatively
- if no reliable speech is audible anywhere, write: none reliable

[Main Interpretation]
- [Inferred] the most likely explanation of what is happening, written cautiously and grounded in the observed evidence

[Alternative Interpretation]
- [Inferred] a genuinely competing explanation that could change downstream verification or final judgment
- [Inferred] when possible, indicate what kind of missing evidence would help distinguish it from the main interpretation

[Key Unclear Points]

U1:
- what is unclear:
- why it matters:
- what needs verification:

U2:
- what is unclear:
- why it matters:
- what needs verification:

[Preliminary Conclusion]
- [Inferred] a cautious summary of the current best understanding without making a final categorical judgment
- confidence: low / medium / high
"""

MERGE_REQUIRED_HEADERS = [
    "[Scene]",
    "[Full Video Narrative]",
    "[Observed Visual Events]",
    "[Speech Transcript]",
    "[Main Interpretation]",
    "[Alternative Interpretation]",
    "[Key Unclear Points]",
    "[Preliminary Conclusion]",
]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class RetrievalQuery:
    id: str
    query_type: str
    query_text: str
    why_this_query: str
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "query_type": self.query_type,
            "query_text": self.query_text,
            "why_this_query": self.why_this_query,
        }
        payload.update(self.extra_fields)
        return payload


@dataclass
class QueryExtractionResult:
    raw_text: str
    parsed_json: dict[str, Any] | None
    retrieval_queries: list[RetrievalQuery]
    parse_error: bool
    parse_error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "retrieval_queries": [_query_to_output_dict(q) for q in self.retrieval_queries],
            "parse_error": self.parse_error,
            "parse_error_message": self.parse_error_message,
        }


@dataclass
class GlobalPipelineResult:
    report_text: str
    query_result: QueryExtractionResult
    evidence_results: list[dict[str, Any]] | None = None
    evidence_judge_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_text": self.report_text,
            "query_result": self.query_result.to_dict(),
            "evidence_results": self.evidence_results,
            "evidence_judge_result": self.evidence_judge_result,
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
class WhisperRuntime:
    model: Any
    device: str
    model_path: str


@dataclass
class ModelRegistry:
    omni: OmniRuntime | None = None
    query: QueryRuntime | None = None
    whisper: WhisperRuntime | None = None
    evidence_judge_vl: Any | None = None


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
    in_speech = False
    seen_speech: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\[(Speech Transcript|Heard Speech Fragments)\]$", stripped):
            in_speech = True
            out.append("[Speech Transcript]")
            continue
        if re.match(r"^\[[^\]]+\]$", stripped) and stripped not in {"[Speech Transcript]", "[Heard Speech Fragments]"}:
            in_speech = False
            out.append(line)
            continue
        if not in_speech:
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
        if norm in seen_speech:
            continue
        seen_speech.add(norm)
        out.append(content)

    merged = "\n".join(out).strip()
    if "[Speech Transcript]" in merged:
        speech_tail = merged.split("[Speech Transcript]", 1)[1].strip()
        if not speech_tail:
            merged = merged + "\n- none reliable"
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


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _format_gib(num_gib: float) -> str:
    return f"{num_gib:.2f} GiB"


def _collect_cuda_memory_stats() -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return stats

    n_gpu = torch.cuda.device_count()
    for idx in range(n_gpu):
        props = torch.cuda.get_device_properties(idx)
        total_bytes = int(props.total_memory)
        try:
            with torch.cuda.device(idx):
                free_bytes, _ = torch.cuda.mem_get_info()
            free_bytes = int(free_bytes)
        except Exception:
            reserved = int(torch.cuda.memory_reserved(idx))
            free_bytes = max(0, total_bytes - reserved)
        allocated_bytes = int(torch.cuda.memory_allocated(idx))
        reserved_bytes = int(torch.cuda.memory_reserved(idx))
        stats.append(
            {
                "index": idx,
                "name": props.name,
                "total_bytes": total_bytes,
                "free_bytes": free_bytes,
                "allocated_bytes": allocated_bytes,
                "reserved_bytes": reserved_bytes,
            }
        )
    return stats


def _print_cuda_memory_snapshot(title: str = "当前 CUDA 显存快照") -> list[dict[str, Any]]:
    stats = _collect_cuda_memory_stats()
    if not stats:
        return stats

    print(f"[显存] {title}")
    for item in stats:
        idx = int(item["index"])
        print(
            f"  - cuda:{idx} ({item['name']}): "
            f"free={_format_gib(_bytes_to_gib(int(item['free_bytes'])))} / "
            f"total={_format_gib(_bytes_to_gib(int(item['total_bytes'])))} | "
            f"allocated={_format_gib(_bytes_to_gib(int(item['allocated_bytes'])))} | "
            f"reserved={_format_gib(_bytes_to_gib(int(item['reserved_bytes'])))}"
        )
    return stats


def _pick_most_free_cuda_device(min_free_gib: float = 0.0) -> str | None:
    stats = _collect_cuda_memory_stats()
    if not stats:
        return None
    best = max(stats, key=lambda x: int(x["free_bytes"]))
    best_free_gib = _bytes_to_gib(int(best["free_bytes"]))
    if best_free_gib < float(min_free_gib):
        return None
    return f"cuda:{int(best['index'])}"


def _build_smart_max_memory(
    memory_stats: list[dict[str, Any]],
    utilization: float = SMART_OMNI_MEMORY_UTILIZATION,
    reserve_gib: float = SMART_OMNI_MEMORY_RESERVE_GIB,
    min_budget_gib: float = SMART_OMNI_MIN_GPU_BUDGET_GIB,
) -> dict[int | str, str]:
    if not memory_stats:
        return {}

    raw_budgets: list[tuple[int, float]] = []
    for item in memory_stats:
        idx = int(item["index"])
        free_gib = _bytes_to_gib(int(item["free_bytes"]))
        total_gib = _bytes_to_gib(int(item["total_bytes"]))
        budget_gib = max(0.0, free_gib * float(utilization) - float(reserve_gib))
        budget_gib = min(budget_gib, max(0.0, total_gib - 1.0))
        if budget_gib >= float(min_budget_gib):
            raw_budgets.append((idx, budget_gib))

    if not raw_budgets:
        return {}

    avg_budget = sum(b for _, b in raw_budgets) / float(len(raw_budgets))
    max_per_gpu = max(4.0, avg_budget * 1.10)

    max_memory: dict[int | str, str] = {}
    for idx, raw_budget in raw_budgets:
        balanced_budget = min(raw_budget, max_per_gpu)
        rounded_budget = max(4, int(balanced_budget))
        max_memory[idx] = f"{rounded_budget}GiB"

    max_memory["cpu"] = "64GiB"
    return max_memory


def _resolve_best_cuda_device_or_default(default: str = "cuda:0") -> str:
    best = _pick_most_free_cuda_device()
    if best is not None:
        return best
    return default


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


def _normalize_single_query_item(
    item: dict[str, Any],
    index: int,
) -> tuple[RetrievalQuery | None, str | None]:
    qid = str(coalesce_query_value(item, ["id", "query_id"], default=f"Q{index + 1}")).strip() or f"Q{index + 1}"
    query_text = str(
        coalesce_query_value(
            item,
            ["query_text", "query", "text", "retrieval_query", "query_content"],
            default="",
        )
    ).strip()
    why_this_query = str(
        coalesce_query_value(
            item,
            ["why_this_query", "reason", "rationale", "why"],
            default="",
        )
    ).strip()
    query_type = normalize_query_type(
        str(coalesce_query_value(item, ["query_type", "type", "route"], default="Visual"))
    )
    if query_type == "OCR":
        query_type = "Visual"

    if not query_text:
        return None, f"retrieval_queries[{index}] missing usable query_text."

    known_core_keys = {
        "id",
        "query_id",
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
    for legacy_key in list(extra_fields.keys()):
        lowered = legacy_key.strip().lower()
        if lowered == "time":
            extra_fields.pop(legacy_key, None)
            continue
        if "time" in lowered and "hint" in lowered:
            extra_fields.pop(legacy_key, None)
            continue
        if lowered.startswith("phase") and "hint" in lowered:
            extra_fields.pop(legacy_key, None)

    extra_fields["needs_external_check"] = to_bool(item.get("needs_external_check"), default=False)
    extra_fields["external_check_focus"] = str(item.get("external_check_focus", "") or "").strip()
    extra_fields["external_check_hint"] = str(item.get("external_check_hint", "") or "").strip()

    return (
        RetrievalQuery(
            id=qid,
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
) -> tuple[list[RetrievalQuery], list[str]]:
    """兼容不同 Query Schema，仅强约束核心字段并保留额外字段。"""
    errors: list[str] = []
    normalized: list[RetrievalQuery] = []

    queries = extract_query_list(payload)
    if not isinstance(queries, list):
        return [], ["No valid query list found. Expected key retrieval_queries/queries."]

    for idx, item in enumerate(queries):
        if len(normalized) >= MAX_RETRIEVAL_QUERIES:
            break
        if not isinstance(item, dict):
            errors.append(f"retrieval_queries[{idx}] must be an object.")
            continue

        query, err = _normalize_single_query_item(item=item, index=idx)
        if err:
            errors.append(err)
            continue
        if query is not None:
            normalized.append(query)

    return normalized, errors

def _load_omni_runtime(
    model_path: str,
    device: str | None = None,
    device_map: str | None = "auto",
) -> OmniRuntime:
    print("=" * 60)
    print("加载模型")
    print("=" * 60)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device is None and n_gpu > 0:
        device = _resolve_best_cuda_device_or_default("cuda:0")
    elif device is None:
        device = "cpu"

    if str(device).startswith("cuda") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("[提示] 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    effective_device_map = device_map
    if n_gpu > 1 and str(device_map).lower() == "auto":
        # 多卡推理时优先给首卡留出更多激活空间，降低 generate 阶段 OOM 风险。
        effective_device_map = "balanced_low_0"
    model_load_kwargs = {
        "torch_dtype": model_dtype,
        "attn_implementation": "sdpa",
    }
    if effective_device_map not in (None, "none"):
        model_load_kwargs["device_map"] = effective_device_map

    smart_max_memory: dict[int | str, str] | None = None
    if n_gpu > 1 and effective_device_map not in (None, "none"):
        memory_stats = _print_cuda_memory_snapshot("Omni 加载前")
        smart_max_memory = _build_smart_max_memory(memory_stats)
        if smart_max_memory:
            model_load_kwargs["max_memory"] = smart_max_memory

    print(f"模型路径: {model_path}")
    print(f"device: {device}")
    print(f"device_map: {effective_device_map}")
    print(f"可见 GPU 数: {n_gpu}")
    if smart_max_memory:
        print(f"智能 max_memory: {smart_max_memory}")

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
    print("开始加载 Query 模型")
    print("=" * 60)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device is None and n_gpu > 0:
        device = _resolve_best_cuda_device_or_default("cuda:0")
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


def _resolve_whisper_device(device: str | None = None) -> str:
    if device and str(device).strip():
        return str(device).strip()
    if torch.cuda.is_available():
        return _resolve_best_cuda_device_or_default("cuda:0")
    return "cpu"


def _load_whisper_runtime(
    whisper_model_path: str = DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
    device: str | None = None,
) -> WhisperRuntime:
    print("\n" + "=" * 60)
    print("开始加载 Whisper 模型")
    print("=" * 60)

    resolved_device = _resolve_whisper_device(device)
    model_path = str(Path(whisper_model_path))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Whisper model not found: {model_path}")

    try:
        import whisper
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Whisper runtime requires openai-whisper. Please install it before running localizer ASR."
        ) from exc

    print(f"whisper_model_path: {model_path}")
    print(f"whisper_device: {resolved_device}")
    model = whisper.load_model(model_path, device=resolved_device)
    print("Whisper 模型加载完成")
    return WhisperRuntime(model=model, device=resolved_device, model_path=model_path)


def initialize_model_registry(
    omni_model_path: str,
    omni_device: str | None = None,
    omni_device_map: str | None = "auto",
    load_query: bool = False,
    query_model_path: str = "/sda/yuqifan/HFOCUS/Qwen3-4B",
    query_device: str | None = None,
    load_whisper: bool = False,
    whisper_model_path: str = DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
    whisper_device: str | None = None,
) -> ModelRegistry:
    """统一预加载模型：默认可在一个阶段将 Omni / Query 一起加载。"""
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

    if load_whisper:
        registry.whisper = _load_whisper_runtime(
            whisper_model_path=whisper_model_path,
            device=whisper_device,
        )

    print("\n统一模型预加载完成")
    return registry


def _release_omni_runtime(registry: ModelRegistry | None) -> None:
    if registry is None or registry.omni is None:
        return

    print("\n" + "=" * 60)
    print("进入证据分析阶段：释放 Omni 模型以回收显存")
    print("=" * 60)
    _print_cuda_memory_snapshot("释放 Omni 前")

    runtime = registry.omni
    registry.omni = None
    try:
        model = runtime.model
        processor = runtime.processor
        del model
        del processor
    except Exception:
        pass
    del runtime

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    _print_cuda_memory_snapshot("释放 Omni 后")


def _get_or_load_evidence_judge_vl_runtime(
    registry: ModelRegistry | None,
    model_path: str,
    device: str | None,
) -> Any:
    if registry is not None and registry.evidence_judge_vl is not None:
        print("[提示] 使用已缓存 evidence_judge Qwen3-VL runtime")
        return registry.evidence_judge_vl

    from evidence_judge_pipeline import _load_vl_runtime

    print("\n" + "=" * 60)
    print("加载 evidence_judge Qwen3-VL 模型")
    print("=" * 60)
    runtime = _load_vl_runtime(model_path=model_path, device=device)
    if registry is not None:
        registry.evidence_judge_vl = runtime
    return runtime


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _shrink_sampling_after_oom(
    fps: float,
    min_frames: int,
    max_frames: int,
    nframes: int | None,
    generation_tokens: int,
) -> tuple[float, int, int, int | None, int, bool]:
    changed = False

    if nframes is not None:
        next_nframes = max(4, int(round(float(nframes) * OMNI_OOM_FRAME_SCALE)))
        if next_nframes < int(nframes):
            nframes = next_nframes
            changed = True
    else:
        next_max_frames = max(16, int(round(float(max_frames) * OMNI_OOM_FRAME_SCALE)))
        if next_max_frames < int(max_frames):
            max_frames = next_max_frames
            changed = True

        next_min_frames = max(4, min(int(min_frames), max_frames))
        if next_min_frames > max_frames:
            next_min_frames = max_frames
        if next_min_frames < int(min_frames):
            min_frames = next_min_frames
            changed = True

        next_fps = max(0.2, float(fps) * OMNI_OOM_FPS_SCALE)
        if next_fps < float(fps):
            fps = next_fps
            changed = True

    next_generation_tokens = max(128, int(round(float(generation_tokens) * OMNI_OOM_TOKEN_SCALE)))
    if next_generation_tokens < int(generation_tokens):
        generation_tokens = next_generation_tokens
        changed = True

    return fps, min_frames, max_frames, nframes, generation_tokens, changed


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
        "Pay attention to actions, scene context, spoken content, and meaningful visual/audio cues. "
        "Do not rely on visual stream alone when audio is informative. "
        "Analyze audio across the full timeline. "
        "Do not confuse on-screen text with spoken audio unless actually heard. "
        "Focus on fuller speech transcription rather than speaker attribution. "
        "Be sensitive to important figures, impersonations, leader lookalikes, symbols, slogans, patches, logos, and gestures. "
        "Do not output timestamps or phase labels. "
        "Ground conclusions in evidence and clearly distinguish observation from interpretation."
    )

    effective_min_frames = max(1, int(video_min_frames))
    effective_max_frames = max(effective_min_frames, int(video_max_frames))
    effective_fps = max(0.1, float(video_fps))

    def _run_omni_for_range(start_sec: float, end_sec: float | None, is_segment: bool) -> str:
        attempt_fps = effective_fps
        attempt_min_frames = effective_min_frames
        attempt_max_frames = effective_max_frames
        attempt_nframes = max(4, int(video_nframes)) if video_nframes is not None else None
        attempt_max_new_tokens = max_new_tokens

        for attempt in range(1, OMNI_OOM_MAX_RETRIES + 2):
            video_item: dict[str, Any] = {
                "type": "video",
                "video": video_path,
                "video_start": float(start_sec),
            }
            if end_sec is not None:
                video_item["video_end"] = float(end_sec)
            if attempt_nframes is not None:
                video_item["nframes"] = max(4, int(attempt_nframes))
            else:
                video_item["fps"] = float(attempt_fps)
                video_item["min_frames"] = max(1, int(attempt_min_frames))
                video_item["max_frames"] = max(int(attempt_min_frames), int(attempt_max_frames))

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

            try:
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
                    max_new_tokens=int(attempt_max_new_tokens),
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
            except Exception as exc:
                if not _is_cuda_oom_error(exc) or attempt > OMNI_OOM_MAX_RETRIES:
                    raise

                print(
                    f"[OOM保护] 段推理触发显存不足，准备重试 {attempt + 1}/{OMNI_OOM_MAX_RETRIES + 1}。"
                )
                _print_cuda_memory_snapshot("OOM 触发时")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                (
                    attempt_fps,
                    attempt_min_frames,
                    attempt_max_frames,
                    attempt_nframes,
                    attempt_max_new_tokens,
                    changed,
                ) = _shrink_sampling_after_oom(
                    fps=attempt_fps,
                    min_frames=attempt_min_frames,
                    max_frames=attempt_max_frames,
                    nframes=attempt_nframes,
                    generation_tokens=attempt_max_new_tokens,
                )
                if not changed:
                    raise

                if attempt_nframes is not None:
                    print(
                        f"[OOM保护] 降采样: nframes={attempt_nframes}, max_new_tokens={attempt_max_new_tokens}"
                    )
                else:
                    print(
                        "[OOM保护] 降采样: "
                        f"fps={attempt_fps:.3f}, min_frames={attempt_min_frames}, "
                        f"max_frames={attempt_max_frames}, max_new_tokens={attempt_max_new_tokens}"
                    )

        raise RuntimeError("Unexpected OOM retry flow termination.")

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


def run_query_extraction(
    report_text: str,
    query_model_path: str,
    device: str | None = None,
    max_new_tokens: int = 1024,
    query_runtime: QueryRuntime | None = None,
) -> QueryExtractionResult:
    """模块 C：从全局理解报告中提炼检索 Query（JSON 输出）。"""

    runtime = query_runtime
    if runtime is None:
        runtime = _load_query_runtime(query_model_path=query_model_path, device=device)

    prompt = build_query_extraction_prompt(report_text)
    raw_text, _ = _generate_query_text_with_stats(
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        system_message=(
            "You are a query extraction assistant. "
            "Output one complete valid JSON object only. "
            "Use retrieval-friendly wording with strong visual anchors. "
            "Do not rely on speaker attribution or time hints. "
            "Use query_type only from ASR, Visual. "
            f"Use key retrieval_queries only and at most {MAX_RETRIEVAL_QUERIES} items."
        ),
    )

    cleaned_text = _clean_query_model_output(raw_text)

    parsed_json: dict[str, Any] | None = None
    retrieval_queries: list[RetrievalQuery] = []
    parse_error = False
    parse_error_message: str | None = None

    try:
        parsed_json = extract_last_json_object(cleaned_text)
        retrieval_queries, validation_errors = _validate_and_normalize_queries(
            parsed_json,
        )
        if validation_errors:
            parse_error = True
            parse_error_message = "; ".join(validation_errors)
    except Exception as exc:
        parse_error = True
        parse_error_message = str(exc)

    return QueryExtractionResult(
        raw_text=raw_text,
        parsed_json=parsed_json,
        retrieval_queries=retrieval_queries,
        parse_error=parse_error,
        parse_error_message=parse_error_message,
    )


def _resolve_harm_rules_text(
    *,
    harm_rules_text: str = "",
    harm_rules_txt_path: str | None = None,
) -> str:
    text = str(harm_rules_text or "").strip()
    if text:
        return text

    txt_path = str(harm_rules_txt_path or "").strip()
    if txt_path:
        return Path(txt_path).expanduser().read_text(encoding="utf-8").strip()

    from evidence_judge_pipeline import BUILTIN_HARM_RULES

    return str(BUILTIN_HARM_RULES).strip()


def _is_same_model_path(a: str, b: str) -> bool:
    pa = str(a or "").strip()
    pb = str(b or "").strip()
    if not pa or not pb:
        return False
    try:
        return Path(pa).expanduser().resolve() == Path(pb).expanduser().resolve()
    except Exception:
        return pa == pb


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
    run_localizer: bool = False,
    run_evidence_judge: bool = False,
    harm_rules_text: str = "",
    evidence_judge_vl_model_path: str = DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH,
    evidence_judge_vl_device: str | None = None,
    evidence_judge_vl_max_new_tokens: int = 768,
    evidence_judge_max_visual_frames: int = 6,
    evidence_judge_llm_model_path: str = DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH,
    evidence_judge_llm_device: str | None = None,
    evidence_judge_llm_max_new_tokens: int = 1024,
    localizer_config: dict[str, Any] | None = None,
    localizer_whisper_model_path: str = DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
    localizer_whisper_device: str | None = None,
    model_registry: ModelRegistry | None = None,
) -> GlobalPipelineResult:
    """总流程：模块 A 生成报告，再由模块 C 提炼 Query。"""

    effective_run_localizer = bool(run_localizer or run_evidence_judge)

    registry = model_registry
    if registry is None:
        registry = initialize_model_registry(
            omni_model_path=omni_model_path,
            omni_device=omni_device,
            omni_device_map=omni_device_map,
            load_query=True,
            query_model_path=query_model_path,
            query_device=query_device,
            load_whisper=effective_run_localizer,
            whisper_model_path=localizer_whisper_model_path,
            whisper_device=localizer_whisper_device,
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
    )

    evidence_results: list[dict[str, Any]] | None = None
    if effective_run_localizer:
        localizer_cfg = localizer_config if localizer_config is not None else {}
        evidence_results = run_query_evidence_localizer(
            video_path=video_path,
            retrieval_queries=query_result.retrieval_queries,
            config=localizer_cfg,
            whisper_runtime=registry.whisper,
        )

    evidence_judge_result: dict[str, Any] | None = None
    if run_evidence_judge:
        resolved_harm_rules_text = _resolve_harm_rules_text(harm_rules_text=harm_rules_text)
        if evidence_results is None:
            raise ValueError("run_evidence_judge 依赖 localizer 结果，但 evidence_results 为空。")

        _release_omni_runtime(registry)
        evidence_judge_vl_runtime = _get_or_load_evidence_judge_vl_runtime(
            registry=registry,
            model_path=evidence_judge_vl_model_path,
            device=evidence_judge_vl_device,
        )
        judge_runtime_for_evidence = (
            registry.query if _is_same_model_path(query_model_path, evidence_judge_llm_model_path) else None
        )
        evidence_judge_result = run_query_evidence_judge(
            video_path=video_path,
            retrieval_queries=query_result.retrieval_queries,
            evidence_results=evidence_results,
            report_text=report_text,
            harm_rules_text=resolved_harm_rules_text,
            qwen3_vl_model_path=evidence_judge_vl_model_path,
            qwen3_vl_device=evidence_judge_vl_device,
            qwen3_vl_max_new_tokens=evidence_judge_vl_max_new_tokens,
            max_visual_frames=evidence_judge_max_visual_frames,
            judge_model_path=evidence_judge_llm_model_path,
            judge_device=evidence_judge_llm_device,
            judge_max_new_tokens=evidence_judge_llm_max_new_tokens,
            vl_runtime=evidence_judge_vl_runtime,
            judge_runtime=judge_runtime_for_evidence,
        )

    return GlobalPipelineResult(
        report_text=report_text,
        query_result=query_result,
        evidence_results=evidence_results,
        evidence_judge_result=evidence_judge_result,
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


def _resolve_localization_json_save_path(video_path: str, save_localization_json: str) -> Path:
    if save_localization_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".localization.json")
    return Path(save_localization_json)


def _resolve_evidence_card_json_save_path(video_path: str, save_evidence_card_json: str) -> Path:
    if save_evidence_card_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".evidence_card.json")
    return Path(save_evidence_card_json)


def _resolve_judge_input_txt_save_path(video_path: str, save_judge_input_txt: str) -> Path:
    if save_judge_input_txt == "auto":
        video = Path(video_path)
        return video.with_suffix(".judge_input.txt")
    return Path(save_judge_input_txt)


def _resolve_judge_final_txt_save_path(video_path: str, save_judge_final_txt: str) -> Path:
    if save_judge_final_txt == "auto":
        video = Path(video_path)
        return video.with_suffix(".judge_final.txt")
    return Path(save_judge_final_txt)


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
    whisper_model_path: str = DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
    whisper_device: str | None = None,
) -> dict[str, Any]:
    from focus_localizer import default_config as focus_localizer_default_config

    cfg: dict[str, Any] = dict(focus_localizer_default_config)
    cfg["blip_model"] = blip_model
    cfg["batch_size"] = int(batch_size)
    cfg["asr_backend"] = "whisper"
    cfg["whisper_model_path"] = str(whisper_model_path)
    if device:
        cfg["device"] = device
    if whisper_device:
        cfg["whisper_device"] = str(whisper_device)

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
    whisper_runtime: WhisperRuntime | None = None,
) -> list[dict[str, Any]]:
    from focus_localizer import build_openai_whisper_asr_infer_fn, localize_all_queries

    normalized_queries: list[dict[str, Any]] = []
    for q in retrieval_queries:
        if isinstance(q, RetrievalQuery):
            normalized_queries.append(_query_to_output_dict(q))
        elif isinstance(q, dict):
            normalized_queries.append(q)

    cfg = dict(config or {})
    cfg.setdefault("asr_backend", "whisper")
    cfg.setdefault("whisper_model_path", DEFAULT_LOCALIZER_WHISPER_MODEL_PATH)
    if whisper_runtime is not None and not callable(cfg.get("asr_infer_fn")):
        cfg["asr_infer_fn"] = build_openai_whisper_asr_infer_fn(
            whisper_model=whisper_runtime.model,
            whisper_device=whisper_runtime.device,
            whisper_model_path=whisper_runtime.model_path,
        )
        cfg.setdefault("whisper_device", whisper_runtime.device)

    return localize_all_queries(
        video_path=video_path,
        retrieval_queries=normalized_queries,
        config=cfg,
    )


def run_query_evidence_judge(
    video_path: str,
    retrieval_queries: list[RetrievalQuery] | list[dict[str, Any]],
    evidence_results: list[dict[str, Any]],
    report_text: str,
    harm_rules_text: str = "",
    harm_rules_txt_path: str | None = None,
    *,
    qwen3_vl_model_path: str = DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH,
    qwen3_vl_device: str | None = None,
    qwen3_vl_max_new_tokens: int = 768,
    max_visual_frames: int = 6,
    judge_model_path: str = DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH,
    judge_device: str | None = None,
    judge_max_new_tokens: int = 1024,
    vl_runtime: Any | None = None,
    judge_runtime: Any | None = None,
) -> dict[str, Any]:
    from evidence_judge_pipeline import run_evidence_judge_pipeline

    normalized_queries: list[dict[str, Any]] = []
    for q in retrieval_queries:
        if isinstance(q, RetrievalQuery):
            normalized_queries.append(_query_to_output_dict(q))
        elif isinstance(q, dict):
            normalized_queries.append(q)

    resolved_harm_rules_text = _resolve_harm_rules_text(
        harm_rules_text=harm_rules_text,
        harm_rules_txt_path=harm_rules_txt_path,
    )

    return run_evidence_judge_pipeline(
        queries_payload={"retrieval_queries": normalized_queries},
        localization_payload={"evidence_results": evidence_results},
        global_report=report_text,
        harm_rules=resolved_harm_rules_text,
        video_path=video_path,
        qwen3_vl_model=qwen3_vl_model_path,
        qwen3_vl_device=qwen3_vl_device,
        qwen3_vl_max_new_tokens=qwen3_vl_max_new_tokens,
        max_visual_frames=max_visual_frames,
        judge_model=judge_model_path,
        judge_device=judge_device,
        judge_max_new_tokens=judge_max_new_tokens,
        vl_runtime=vl_runtime,
        judge_runtime=judge_runtime,
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
    parser.add_argument(
        "--localizer_whisper_model",
        type=str,
        default=DEFAULT_LOCALIZER_WHISPER_MODEL_PATH,
        help="localizer ASR 使用的 Whisper 模型路径（默认仓库内 Whisper/large-v3.pt）",
    )
    parser.add_argument(
        "--localizer_whisper_device",
        type=str,
        default=None,
        help="localizer Whisper 运行设备，如 cuda / cuda:0 / cpu",
    )

    parser.add_argument("--run_evidence_judge", action="store_true", help="启用证据卡分析 + 最终审判（Qwen3-VL + Qwen3-4B）")
    parser.add_argument(
        "--harm_rules_txt",
        type=str,
        default=None,
        help="有害内容分类规则文本路径；不填则自动使用 evidence_judge_pipeline 内置规则",
    )
    parser.add_argument(
        "--evidence_judge_vl_model",
        type=str,
        default=DEFAULT_EVIDENCE_JUDGE_VL_MODEL_PATH,
        help="evidence_judge 的 Qwen3-VL 模型路径",
    )
    parser.add_argument(
        "--evidence_judge_vl_device",
        type=str,
        default=None,
        help="evidence_judge 的 Qwen3-VL 运行设备，如 cuda:0 / cpu",
    )
    parser.add_argument(
        "--evidence_judge_vl_max_new_tokens",
        type=int,
        default=768,
        help="evidence_judge 每个 query 的 Qwen3-VL 最大生成 token",
    )
    parser.add_argument(
        "--evidence_judge_max_visual_frames",
        type=int,
        default=6,
        help="evidence_judge 的 Visual query 输入帧上限",
    )
    parser.add_argument(
        "--evidence_judge_model",
        type=str,
        default=DEFAULT_EVIDENCE_JUDGE_LLM_MODEL_PATH,
        help="evidence_judge 的最终审判模型路径（Qwen3-4B）",
    )
    parser.add_argument(
        "--evidence_judge_device",
        type=str,
        default=None,
        help="evidence_judge 最终审判模型设备，如 cuda:0 / cpu",
    )
    parser.add_argument(
        "--evidence_judge_max_new_tokens",
        type=int,
        default=1024,
        help="evidence_judge 最终审判模型最大生成 token",
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
        "--save_localization_json",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存 localizer 结果 JSON。仅写 --save_localization_json 时自动保存为同名 .localization.json；也可指定输出路径",
    )
    parser.add_argument(
        "--save_evidence_card_json",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存 evidence_card JSON。仅写 --save_evidence_card_json 时自动保存为同名 .evidence_card.json；也可指定输出路径",
    )
    parser.add_argument(
        "--save_judge_input_txt",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存审判输入文本。仅写 --save_judge_input_txt 时自动保存为同名 .judge_input.txt；也可指定输出路径",
    )
    parser.add_argument(
        "--save_judge_final_txt",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存最终审判结果。仅写 --save_judge_final_txt 时自动保存为同名 .judge_final.txt；也可指定输出路径",
    )

    args = parser.parse_args()

    effective_run_localizer = bool(args.run_localizer or args.run_evidence_judge)

    model_registry: ModelRegistry | None = None
    if args.run_query_extraction and not args.disable_preload:
        model_registry = initialize_model_registry(
            omni_model_path=args.model,
            omni_device=args.device,
            omni_device_map=args.device_map,
            load_query=True,
            query_model_path=args.query_model,
            query_device=args.query_device,
            load_whisper=effective_run_localizer,
            whisper_model_path=args.localizer_whisper_model,
            whisper_device=args.localizer_whisper_device,
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

        if effective_run_localizer:
            localizer_cfg = _build_localizer_config(
                blip_model=args.localizer_blip_model,
                device=args.localizer_device,
                batch_size=args.localizer_batch_size,
                config_json=args.localizer_config_json,
                whisper_model_path=args.localizer_whisper_model,
                whisper_device=args.localizer_whisper_device,
            )
            evidence_results = run_query_evidence_localizer(
                video_path=args.video,
                retrieval_queries=query_result.retrieval_queries,
                config=localizer_cfg,
                whisper_runtime=model_registry.whisper if model_registry else None,
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

            if args.run_evidence_judge:
                _release_omni_runtime(model_registry)
                evidence_judge_vl_runtime = _get_or_load_evidence_judge_vl_runtime(
                    registry=model_registry,
                    model_path=args.evidence_judge_vl_model,
                    device=args.evidence_judge_vl_device,
                )
                judge_runtime_for_evidence = (
                    model_registry.query if (model_registry and _is_same_model_path(args.query_model, args.evidence_judge_model)) else None
                )
                evidence_judge_result = run_query_evidence_judge(
                    video_path=args.video,
                    retrieval_queries=query_result.retrieval_queries,
                    evidence_results=evidence_results,
                    report_text=report,
                    harm_rules_txt_path=args.harm_rules_txt,
                    qwen3_vl_model_path=args.evidence_judge_vl_model,
                    qwen3_vl_device=args.evidence_judge_vl_device,
                    qwen3_vl_max_new_tokens=args.evidence_judge_vl_max_new_tokens,
                    max_visual_frames=args.evidence_judge_max_visual_frames,
                    judge_model_path=args.evidence_judge_model,
                    judge_device=args.evidence_judge_device,
                    judge_max_new_tokens=args.evidence_judge_max_new_tokens,
                    vl_runtime=evidence_judge_vl_runtime,
                    judge_runtime=judge_runtime_for_evidence,
                )

                print("\n" + "=" * 60)
                print("最终审判结果（Qwen3-4B）")
                print("=" * 60)
                print(str(evidence_judge_result.get("judge_normalized", "")).strip())

                if args.save_evidence_card_json is not None:
                    evidence_card_save_path = _resolve_evidence_card_json_save_path(args.video, args.save_evidence_card_json)
                    evidence_card_save_path.parent.mkdir(parents=True, exist_ok=True)
                    evidence_card_save_path.write_text(
                        str(evidence_judge_result.get("evidence_card_json", "")),
                        encoding="utf-8",
                    )
                    print(f"\nEvidence Card JSON 已保存: {evidence_card_save_path}")

                if args.save_judge_input_txt is not None:
                    judge_input_save_path = _resolve_judge_input_txt_save_path(args.video, args.save_judge_input_txt)
                    judge_input_save_path.parent.mkdir(parents=True, exist_ok=True)
                    judge_input_save_path.write_text(
                        str(evidence_judge_result.get("judge_prompt", "")),
                        encoding="utf-8",
                    )
                    print(f"审判输入文本已保存: {judge_input_save_path}")

                if args.save_judge_final_txt is not None:
                    judge_final_save_path = _resolve_judge_final_txt_save_path(args.video, args.save_judge_final_txt)
                    judge_final_save_path.parent.mkdir(parents=True, exist_ok=True)
                    judge_final_save_path.write_text(
                        "\n".join(
                            [
                                "[raw_output]",
                                str(evidence_judge_result.get("judge_raw", "")),
                                "",
                                "[normalized_output]",
                                str(evidence_judge_result.get("judge_normalized", "")),
                            ]
                        ),
                        encoding="utf-8",
                    )
                    print(f"审判结果已保存: {judge_final_save_path}")
        elif args.save_localization_json is not None:
            print("\n[提示] 已指定 --save_localization_json，但未启用 --run_localizer/--run_evidence_judge，跳过保存。")

        if not args.run_evidence_judge:
            if args.save_evidence_card_json is not None:
                print("\n[提示] 已指定 --save_evidence_card_json，但未启用 --run_evidence_judge，跳过保存。")
            if args.save_judge_input_txt is not None:
                print("\n[提示] 已指定 --save_judge_input_txt，但未启用 --run_evidence_judge，跳过保存。")
            if args.save_judge_final_txt is not None:
                print("\n[提示] 已指定 --save_judge_final_txt，但未启用 --run_evidence_judge，跳过保存。")

    else:
        if args.save_query_json is not None:
            print("\n[提示] 已指定 --save_query_json，但未启用 --run_query_extraction，跳过保存。")
        if args.run_localizer:
            print("\n[提示] 已启用 --run_localizer，但未启用 --run_query_extraction，localizer 不会执行。")
        if args.run_evidence_judge:
            print("\n[提示] 已启用 --run_evidence_judge，但未启用 --run_query_extraction，evidence_judge 不会执行。")
        if args.save_localization_json is not None:
            print("\n[提示] 已指定 --save_localization_json，但未启用 --run_query_extraction，跳过保存。")
        if args.save_evidence_card_json is not None:
            print("\n[提示] 已指定 --save_evidence_card_json，但未启用 --run_query_extraction，跳过保存。")
        if args.save_judge_input_txt is not None:
            print("\n[提示] 已指定 --save_judge_input_txt，但未启用 --run_query_extraction，跳过保存。")
        if args.save_judge_final_txt is not None:
            print("\n[提示] 已指定 --save_judge_final_txt，但未启用 --run_query_extraction，跳过保存。")
