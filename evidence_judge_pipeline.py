from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from query_utils import normalize_query_input

DEFAULT_QWEN3_VL_MODEL_PATH = "/sda/yuqifan/HFOCUS/Qwen3-VL"
DEFAULT_QWEN3_4B_MODEL_PATH = "/sda/yuqifan/HFOCUS/Qwen3-4B"

BUILTIN_HARM_RULES = """
I. Maliciously Inciting Division and Intensifying Social Conflict

The core task is to identify any content that intends to create hatred, confrontation, or discrimination between groups.

The definition and judgment criteria for Category 1 harmful content are as follows and must be followed strictly:

-1. Creating conflict between groups or ideologies: must be judged as harmful.
This includes provoking hostility, suspicion, or antagonism between groups defined by gender, ethnicity, region, social class, religion, or similar attributes. It is harmful whether the incitement is explicit or indirect, including subtle suggestion, labeling, or the use of oppositional narrative frameworks such as "us vs. them" that encourage viewers to infer group conflict.

-2. Creating national or political confrontation: must be judged as harmful.
This includes provoking hostility between different countries, such as China vs. the United States or Russia vs. Ukraine, or intensifying political factional struggle within a country.

-3. Discriminating against or humiliating specific groups: must be judged as harmful.
This includes mockery, ridicule, humiliation, or the reinforcement of stereotypes based on the group attributes described above. The key distinction from confrontation is that discrimination is centered on contempt and degradation rather than direct opposition.

Important principle:
Judgment should focus on whether the content is creating division. Even if no vulgar language is used, the content must be directly judged as harmful if its core logic, context, or implied effect is to undermine social cohesion and incite antagonistic emotions. For content expressed through stories, metaphors, satire, or other indirect means, its deeper social impact must be carefully understood.

II. False Information and Superstitious Content

The core task is to identify and judge any video content that spreads false information, pseudoscience, or feudal superstition.

The definition and judgment criteria for Category 2 harmful content are as follows and must be followed strictly:

-1. Pseudoscientific content: must be judged as harmful.
This refers to theories proposed by self-styled "independent scientists" or individuals that contradict scientific consensus. Examples include self-invented formulas, concepts, or models, such as fabricated "formulas" or alleged products like "energy particle water," as well as any unverified anti-scientific or anti-intellectual claims disguised as scientific knowledge. The key criterion is the anti-scientific nature of the content itself. Regardless of whether it is presented as academic discussion or personal opinion, the act of spreading it is harmful.

-2. Feudal superstition and fortune-telling content: must be judged as harmful.
This includes, but is not limited to, fortune-telling, divination, physiognomy, geomancy/feng shui for changing fate, witchcraft, magic rituals, or other claims that human destiny is determined by supernatural forces or can be predicted or altered through such means.

-3. Rumors and conspiracy theories: must be judged as harmful.
Examples include pseudo-history, flat-earth claims, or any other unverified false information or conspiracy theories.

Important principle:
Judgment must be based on the subject matter and essence of the content rather than its presentation style. As long as the core content, theme, or main demonstrative segment of the video involves any of the above, it must be directly judged as harmful. It must not be classified as harmless on the grounds that it appears to be "personal theoretical discussion," contains "no inflammatory language," or is presented in a "calm" or "rational" tone.

III. Undermining Social Safety and Order, or Harming Mental Well-Being

The core task is to identify any content that directly depicts or promotes violence, pornography, terrorism, or behavior that disrupts social order.

The definition and judgment criteria for Category 3 harmful content are as follows and must be followed strictly:

-1. Violent and abusive content: must be judged as harmful.
This includes directly showing or describing in detail cruel scenes such as fighting, abuse, murder, or self-harm. Non-educational content that glorifies or sensationalizes violence also falls into this category.

-2. Terror-inducing and psychologically disturbing content: must be judged as harmful.
This includes bloody, horrifying, or tragic scenes or sound effects intended to create intense mental shock, fear, or distress.

-3. Pornographic and sexually suggestive content: must be judged as harmful.
This includes direct displays of sexual acts or sexual organs, as well as implicit nudity, accidental exposure, sexual teasing, and prostitution-related content.

-4. Behavior that disrupts social order: must be judged as harmful.
This includes showing serious acts that undermine public order, such as smashing, looting, arson, violent attacks on police, or mob violence that severely disrupts social stability.

Important principle:
Judgment should be based on the degree of direct presentation and the potential harm of the content. Regardless of whether the content is framed as "documentary," "art," or "entertainment," if its core includes the concrete harmful elements described above and poses a direct threat to viewers' sense of safety or mental well-being, or causes severe discomfort, it must be judged as harmful. Any claimed "background" or "purpose" need not be considered.

IV. Violating Mainstream Values and Moral Norms

The core task is to identify any content that challenges public morality and social norms, distorts right and wrong, or spreads misleading value orientations.

The definition and judgment criteria for Category 4 harmful content are as follows and must be followed strictly:

-1. Reversing right and wrong: must be judged as harmful.
This includes glorifying criminals, organized crime groups, or historical wrongdoers; vilifying heroes, martyrs, or national symbols; or portraying war, violence, major disasters, or historical tragedies in an entertaining, frivolous, or trivializing way.

-2. Promoting distorted values and harmful behavior: must be judged as harmful.
This includes promoting decadent or nihilistic attitudes, suicide, or self-harm; displaying and encouraging profanity, verbal abuse, smoking, gambling, or other harmful behaviors; and spreading ideas or customs such as incestuous marriage or mocking and bullying the weak.

-3. Damaging nature and social harmony: must be judged as harmful.
This includes promoting or showing pollution, waste of resources, or destruction of ecological balance in a careless or approving manner.

Important principle:
Judgment should be based on the value orientation of the content. Even if the content appears in the form of "humor," "satire," or "personal opinion," it must still be directly judged as harmful if its ultimate effect is to blur moral boundaries, trivialize serious matters, encourage harmful behavior, or spread pessimistic and destructive worldviews that challenge mainstream social values and moral bottom lines. Particular attention should be paid to its likely social impact.
"""


VL_SYSTEM_PROMPT = (
    "You are a local video evidence verification assistant. "
    "You will receive a retrieval query, the local evidence associated with that query, "
    "and a global understanding report of the video. "
    "Your task is to answer the verification question implied by why_this_query, "
    "which targets an unclear point about the video, based on the current local evidence "
    "and in light of the video's global understanding report, "
    "and to assess how fully the current evidence can answer that question."
)

VL_USER_PROMPT_TEMPLATE = """Please generate an evidence card based on the following local evidence, and output strictly in JSON only (do not output any other text).

Task definition:
- query_text is only a retrieval phrase used to localize the video segment; it is not the real question to answer in this step.
- why_this_query is the actual verification target that this evidence card must address.
- Your core task is to answer the question that why_this_query is trying to verify, based primarily on [Evidence For Analysis], and to explain the strength of the evidence and the remaining uncertainty.

Rules you must follow:
1. Local evidence comes first:
   - Base your main judgment on [Evidence For Analysis].
   - [Omni Global Understanding Report] is background context that may help interpretation, but it must not replace the current local evidence.
   - If the global report conflicts with the current local evidence, trust the current local evidence first and explicitly note the inconsistency.

2. query_text is not the final verification target:
   - Do not make a superficial judgment about whether query_text itself is "true" or "supported."
   - Instead, answer the verification intent expressed by why_this_query.
   - In other words, you must answer whether the current evidence can clarify the issue that why_this_query is trying to verify.

3. Strictly separate observation from inference:
   - Only information directly supported by images, ASR transcripts, emotion analysis, sound event detection, main segments, or supporting frames counts as observed evidence.
   - Motives, stances, identities, deeper meanings, and value judgments may only be inferred cautiously when the evidence allows.
   - Do not present guesses as facts.

4. Handling different query types:
   - If query_type = Visual:
     Focus on the extracted supporting frames, main segment, and visual cues, and explain what the current visual evidence supports with respect to why_this_query.
   - If query_type = ASR:
     Focus on the highest-relevance visual anchor frame, the ASR transcript for the corresponding segment, emotion analysis, and sound event detection, and explain what the current audio-related evidence supports with respect to why_this_query.
   - Do not treat ASR emotion labels as definitive semantic meaning, and do not use a single visual frame to over-complete the meaning of spoken content.

5. The output should serve the downstream final judge:
   - answer_to_why must directly answer the question that why_this_query is trying to verify.
   - support_level indicates how strongly the current evidence supports that answer; it is not a judgment about the whole video.
   - remaining_uncertainty should include only the most important unresolved gaps that still matter for the final judgment.
   - confidence reflects the reliability and stability of this evidence card itself.

Output format (must be valid JSON):
{{
  "query_id": "...",
  "query_type": "...",
  "verification_focus": "...",
  "evidence_summary": "...",
  "answer_to_why": "...",
  "support_level": "strongly_supported | supported | weakly_supported | inconclusive | contradicted",
  "remaining_uncertainty": "...",
  "confidence": "low | medium | high"
}}

support_level rules:
- strongly_supported: the evidence directly and clearly answers why_this_query in the intended direction, with little ambiguity
- supported: the evidence generally answers why_this_query in the intended direction, but some uncertainty remains
- weakly_supported: the evidence gives only limited support for answering why_this_query in the intended direction
- inconclusive: the evidence is not sufficient to answer why_this_query reliably
- contradicted: the evidence tends to answer why_this_query against the intended direction

Field requirements:
- verification_focus:
  State in one sentence the core question that why_this_query is actually trying to verify. Do not copy it too literally if it is overly long.
- evidence_summary:
  Briefly summarize only the parts of the current local evidence that are truly relevant to this verification target. Do not broadly restate all input content.
- answer_to_why:
  Directly answer the question that why_this_query is trying to clarify. If the current evidence only answers part of it, explicitly state what can and cannot currently be confirmed.
- remaining_uncertainty:
  Include only the most important unresolved issue(s), prioritizing those that still matter for the downstream final judge.
- confidence:
  Reflect your confidence in this evidence card, considering evidence quality, directness, consistency, and ambiguity.

[Query]
- id: {query_id}
- type: {query_type}
- text: {query_text}
- why_this_query: {why_this_query}

[Evidence For Analysis]
{evidence_for_analysis_json}

[Omni Global Understanding Report]
{global_report}
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a final harmful-content adjudication model. "
    "You will receive multiple evidence cards, an Omni global understanding report, "
    "and harmful-content classification rules. "
    "Your task is to make the final judgment based on the rules, not to improvise freely. "
    "You must treat the verified local evidence as the core basis of judgment, "
    "and use inferential content from the global report cautiously. "
    "When evidence is insufficient, conflicting, or not enough to map cleanly to the rules, "
    "you must state that clearly in your reasoning and must not present guesses as facts."
)

JUDGE_USER_PROMPT_TEMPLATE = """Please perform the final adjudication.

Task goal:
Based on the provided evidence cards, the global understanding report, and the harmful-content classification rules,
determine whether the final label for this video should be <Harmful> or <Harmless>.

Adjudication principles (must follow):
1. Rules first:
   - The final judgment must align as closely as possible with [Harmful Content Classification Rules].
   - Do not make the final decision based only on vague impressions, tone, commonsense associations, or isolated clues.

2. Verified evidence comes first:
   - [Evidence Cards] are the core basis for the final judgment.
   - Each evidence card reflects how well one why_this_query has been answered under the current evidence.
   - Prioritize verification_focus, answer_to_why, support_level, remaining_uncertainty, and confidence when forming the final judgment.

3. Use the global report cautiously:
   - [Omni Global Understanding Report] is only supplementary context and must not override or replace local verification results.
   - If a claim in the global report is not actually supported by the evidence cards, it should not be used as a core basis for final labeling.

4. Handle conflict and insufficiency carefully:
   - If different evidence cards conflict with each other, you must point this out and explain which evidence you trust more and why.
   - If most key evidence cards remain inconclusive, weakly_supported, or low-confidence, do not force a strong conclusion.
   - Only output <Harmful> when the core harmful criteria are sufficiently supported.

5. Focus on core criteria, not simple counting:
   - More evidence cards do not automatically mean stronger justification.
   - What matters is whether they jointly support a key harmful-content rule.
   - If the supported evidence concerns only peripheral details, while the core harmful intent, behavior, expression, or value orientation remains unverified, stay cautious.

6. Report uncertainty only when it is decision-relevant:
   - If there are still important unresolved uncertainties or evidence gaps that could materially affect the final judgment, explicitly mention them.
   - If the evidence is already clear, sufficient, and consistent enough for a confident judgment, you do not need to force additional uncertainty statements.

7. Output requirements:
   - The first line must be exactly <Harmful> or <Harmless>
   - Then provide one concise but complete reasoning paragraph
   - The reasoning should include:
     a) which key evidence supports your conclusion
     b) which rule(s) that evidence maps to
     c) any important remaining uncertainty or evidence gaps, but only if they are genuinely decision-relevant
   - Do not output JSON
   - Do not use bullet points or extra section headers
   - Do not restate all evidence cards; summarize only the most important parts

[Evidence Cards]
{evidence_cards_json}

[Omni Global Understanding Report]
{global_report}

[Harmful Content Classification Rules]
{harm_rules}
"""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
VALID_SUPPORT_LEVELS = {
    "strongly_supported",
    "supported",
    "weakly_supported",
    "inconclusive",
    "contradicted",
}
VALID_CONFIDENCE = {"low", "medium", "high"}


@dataclass
class VLRuntime:
    model: Any
    processor: Any
    device: Any


@dataclass
class JudgeRuntime:
    model: Any
    tokenizer: Any
    device: str


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_json(path: str) -> Any:
    return json.loads(_read_text(path))


def _resolve_temp_dir() -> str:
    for key in ("TMPDIR", "TEMP", "TMP"):
        candidate = str(os.environ.get(key, "")).strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # Default to repo-local temporary directory to avoid /tmp space pressure.
    default_dir = Path(__file__).resolve().parent / ".tmp"
    default_dir.mkdir(parents=True, exist_ok=True)
    return str(default_dir)


def _resolve_harm_rules(harm_rules_txt_path: str, use_builtin_harm_rules: bool) -> str:
    if use_builtin_harm_rules:
        return BUILTIN_HARM_RULES.strip()
    if str(harm_rules_txt_path or "").strip():
        return _read_text(harm_rules_txt_path).strip()
    return BUILTIN_HARM_RULES.strip()


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    last_obj: dict[str, Any] | None = None
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    if last_obj is None:
        raise ValueError("No valid JSON object found in model output.")
    return last_obj


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def _normalize_queries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("retrieval_queries")
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        try:
            normalized.append(normalize_query_input(item))
        except Exception:
            continue
    return normalized


def _normalize_evidence_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        maybe = payload.get("evidence_results")
        if isinstance(maybe, list):
            return [x for x in maybe if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _build_evidence_index(evidence_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in evidence_results:
        qid = str(item.get("query_id", "")).strip()
        if qid:
            out[qid] = item
    return out


def _normalize_card_query_type(raw: Any, fallback: str = "Visual") -> str:
    value = str(raw or "").strip().lower()
    if value == "asr":
        return "ASR"
    if value:
        return "Visual"
    return "ASR" if str(fallback).strip().lower() == "asr" else "Visual"


def _normalize_vl_analysis_card(parsed: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    query_id = str(query.get("id", "")).strip()
    query_type = _normalize_card_query_type(query.get("query_type"), fallback="Visual")

    def _s(key: str) -> str:
        return str(parsed.get(key, "") or "").strip()

    parsed_qid = _s("query_id")
    parsed_qtype = _normalize_card_query_type(_s("query_type"), fallback=query_type)

    support_level = _s("support_level").lower()
    if support_level not in VALID_SUPPORT_LEVELS:
        support_level = "inconclusive"

    confidence = _s("confidence").lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    return {
        "query_id": parsed_qid or query_id,
        "query_type": parsed_qtype or query_type,
        "verification_focus": _s("verification_focus"),
        "evidence_summary": _s("evidence_summary"),
        "answer_to_why": _s("answer_to_why"),
        "support_level": support_level,
        "remaining_uncertainty": _s("remaining_uncertainty"),
        "confidence": confidence,
    }


def _build_compact_evidence_cards_for_judge(evidence_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for card in evidence_cards:
        query_stub = {
            "id": str(card.get("query_id", "")).strip(),
            "query_type": _normalize_card_query_type(card.get("query_type"), fallback="Visual"),
        }
        parsed = card.get("vl_analysis")
        normalized = _normalize_vl_analysis_card(parsed if isinstance(parsed, dict) else {}, query_stub)
        compact.append(
            {
                "query_id": query_stub["id"],
                "query_type": query_stub["query_type"],
                "query_text": str(card.get("query_text", "")).strip(),
                "why_this_query": str(card.get("why_this_query", "")).strip(),
                "verification_focus": normalized.get("verification_focus", ""),
                "evidence_summary": normalized.get("evidence_summary", ""),
                "answer_to_why": normalized.get("answer_to_why", ""),
                "support_level": normalized.get("support_level", "inconclusive"),
                "remaining_uncertainty": normalized.get("remaining_uncertainty", ""),
                "confidence": normalized.get("confidence", "low"),
                "vl_parse_error": str(card.get("vl_parse_error", "") or "").strip(),
            }
        )
    return compact


def _to_uint8_image(frame: Any) -> Any:
    import numpy as np

    image = frame.asnumpy() if hasattr(frame, "asnumpy") and callable(frame.asnumpy) else frame
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size > 0 and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class FrameProvider:
    def __init__(self, video_path: str | None):
        self.video_path = (video_path or "").strip()
        self._video = None
        self._cache: dict[int, str] = {}
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="evidence_judge_frames_",
            dir=_resolve_temp_dir(),
        )

    def _open_if_needed(self) -> None:
        if self._video is not None:
            return
        if not self.video_path:
            return
        try:
            from decord import VideoReader, cpu
        except Exception:
            return
        self._video = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)

    def export_frame(self, frame_idx: int) -> str:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        self._open_if_needed()
        if self._video is None:
            return ""
        if frame_idx < 0 or frame_idx >= int(len(self._video)):
            return ""
        try:
            from PIL import Image
        except Exception:
            return ""
        image = _to_uint8_image(self._video[frame_idx])
        path = Path(self._tmpdir.name) / f"frame_{frame_idx:06d}.jpg"
        Image.fromarray(image).save(path, format="JPEG", quality=95)
        self._cache[frame_idx] = str(path)
        return str(path)

    def close(self) -> None:
        self._tmpdir.cleanup()


def _resolve_existing_image(path: str) -> str:
    p = Path(str(path or "").strip()).expanduser()
    if p.is_file():
        return str(p.resolve())
    return ""


def _build_exported_frame_map(evidence_item: dict[str, Any]) -> dict[int, str]:
    exported = evidence_item.get("exported_frames")
    if not isinstance(exported, list):
        return {}
    out: dict[int, str] = {}
    for item in exported:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("frame_idx"))
        except Exception:
            continue
        img = _resolve_existing_image(str(item.get("image_path", "")).strip())
        if img:
            out[idx] = img
    return out


def _sanitize_image_paths(paths: list[str], max_count: int | None = None) -> list[str]:
    out: list[str] = []
    for raw in paths:
        img = _resolve_existing_image(raw)
        if img and img not in out:
            out.append(img)
    if max_count is None:
        return out
    return out[: max(0, int(max_count))]


def _pick_top_supporting_frame(supporting_frames: Any) -> dict[str, Any] | None:
    if not isinstance(supporting_frames, list):
        return None
    valid: list[dict[str, Any]] = []
    for item in supporting_frames:
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_idx"))
            time_sec = float(item.get("time_sec"))
            score = float(item.get("score", 0.0))
        except Exception:
            continue
        valid.append({"frame_idx": frame_idx, "time_sec": time_sec, "score": score})
    if not valid:
        return None
    return max(valid, key=lambda x: (float(x.get("score", 0.0)), -int(x.get("frame_idx", 0))))


def _derive_asr_evidence_for_analysis(evidence_item: dict[str, Any]) -> dict[str, Any]:
    compact = evidence_item.get("evidence_for_analysis")
    if isinstance(compact, dict):
        return dict(compact)

    # ASR evidence card input assumes audio understanding is anchored by a
    # localized visual segment plus the strongest supporting frame.
    asr_result = evidence_item.get("asr_result")
    asr_dict = asr_result if isinstance(asr_result, dict) else {}
    return {
        "query_type": "ASR",
        "evidence_found": bool(evidence_item.get("evidence_found")),
        "main_segment": evidence_item.get("main_segment"),
        "top_supporting_frame": _pick_top_supporting_frame(evidence_item.get("supporting_frames")),
        "asr_text": str(asr_dict.get("text", "") or "").strip(),
        "emotion": asr_dict.get("emotion") if isinstance(asr_dict.get("emotion"), dict) else {},
        "sound_events": asr_dict.get("sound_events") if isinstance(asr_dict.get("sound_events"), list) else [],
        "asr_status": str(evidence_item.get("asr_status", "") or "").strip(),
        "asr_reason": str(evidence_item.get("asr_reason", "") or "").strip(),
    }


def _derive_visual_evidence_for_analysis(evidence_item: dict[str, Any]) -> dict[str, Any]:
    compact = evidence_item.get("evidence_for_analysis")
    if isinstance(compact, dict):
        return dict(compact)
    # Visual evidence card input assumes support mainly comes from the
    # localized segment and multiple supporting frames.
    return {
        "query_type": "Visual",
        "evidence_found": bool(evidence_item.get("evidence_found")),
        "main_segment": evidence_item.get("main_segment"),
        "supporting_frames": evidence_item.get("supporting_frames") if isinstance(evidence_item.get("supporting_frames"), list) else [],
    }


def _collect_query_images(
    *,
    query_type: str,
    evidence_item: dict[str, Any],
    frame_provider: FrameProvider,
    max_visual_images: int,
) -> list[str]:
    exported_map = _build_exported_frame_map(evidence_item)
    image_paths: list[str] = []

    def add_by_frame_idx(frame_idx: int) -> None:
        if frame_idx in exported_map:
            path = exported_map[frame_idx]
        else:
            path = frame_provider.export_frame(frame_idx)
            path = _resolve_existing_image(path) if path else ""
        if path and path not in image_paths:
            image_paths.append(path)

    if query_type == "ASR":
        # ASR route provides one strong visual anchor image to pair with
        # transcript/emotion/sound-event evidence in the prompt.
        asr_evidence = _derive_asr_evidence_for_analysis(evidence_item)
        top = asr_evidence.get("top_supporting_frame")
        if isinstance(top, dict):
            if _resolve_existing_image(str(top.get("image_path", ""))):
                image_paths.append(_resolve_existing_image(str(top.get("image_path", ""))))
            else:
                try:
                    add_by_frame_idx(int(top.get("frame_idx")))
                except Exception:
                    pass
        return _sanitize_image_paths(image_paths, max_count=1)

    # Visual route keeps multiple top-scored supporting frames.
    supporting_frames = evidence_item.get("supporting_frames")
    candidates = supporting_frames if isinstance(supporting_frames, list) else []
    sorted_candidates = sorted(
        [x for x in candidates if isinstance(x, dict)],
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )[: max(1, int(max_visual_images))]
    for item in sorted_candidates:
        if _resolve_existing_image(str(item.get("image_path", ""))):
            path = _resolve_existing_image(str(item.get("image_path", "")))
            if path and path not in image_paths:
                image_paths.append(path)
            continue
        try:
            add_by_frame_idx(int(item.get("frame_idx")))
        except Exception:
            continue
    return _sanitize_image_paths(image_paths, max_count=max(1, int(max_visual_images)))


def _resolve_torch_dtype(device: str, torch_module: Any) -> Any:
    if str(device).startswith("cuda"):
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def _load_vl_runtime(model_path: str, device: str | None) -> VLRuntime:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError("Qwen3-VL 运行依赖未满足（需要 torch + transformers 新版本）。") from exc

    resolved_device = (device or "").strip()
    if not resolved_device:
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_torch_dtype(resolved_device, torch)

    resolved_model_path = (model_path or "").strip() or DEFAULT_QWEN3_VL_MODEL_PATH
    if not Path(resolved_model_path).exists():
        raise FileNotFoundError(f"Qwen3-VL model path not found: {resolved_model_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(resolved_device)
    model.eval()

    processor = AutoProcessor.from_pretrained(resolved_model_path, trust_remote_code=True)
    return VLRuntime(model=model, processor=processor, device=resolved_device)


def _load_judge_runtime(model_path: str, device: str | None) -> JudgeRuntime:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Qwen3-4B 运行依赖未满足（需要 torch + transformers）。") from exc

    resolved_device = (device or "").strip()
    if not resolved_device:
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_torch_dtype(resolved_device, torch)

    resolved_model_path = (model_path or "").strip() or DEFAULT_QWEN3_4B_MODEL_PATH
    if not Path(resolved_model_path).exists():
        raise FileNotFoundError(f"Judge model path not found: {resolved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(resolved_device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return JudgeRuntime(model=model, tokenizer=tokenizer, device=resolved_device)


def _coerce_external_vl_runtime(runtime: Any) -> VLRuntime | None:
    if runtime is None:
        return None
    model = getattr(runtime, "model", None)
    processor = getattr(runtime, "processor", None)
    if model is None or processor is None:
        return None
    device = str(getattr(runtime, "device", "") or getattr(model, "device", ""))
    return VLRuntime(model=model, processor=processor, device=device)


def _coerce_external_judge_runtime(runtime: Any) -> JudgeRuntime | None:
    if runtime is None:
        return None
    model = getattr(runtime, "model", None)
    tokenizer = getattr(runtime, "tokenizer", None)
    if model is None or tokenizer is None:
        return None
    device = str(getattr(runtime, "device", "") or getattr(model, "device", ""))
    return JudgeRuntime(model=model, tokenizer=tokenizer, device=device)


def _run_vl_per_query(
    runtime: VLRuntime,
    *,
    query: dict[str, Any],
    evidence_for_analysis: dict[str, Any],
    image_paths: list[str],
    global_report: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    image_paths = _sanitize_image_paths(image_paths)
    prompt = VL_USER_PROMPT_TEMPLATE.format(
        query_id=str(query.get("id", "")).strip(),
        query_type=str(query.get("query_type", "")).strip(),
        query_text=str(query.get("query_text", "")).strip(),
        why_this_query=str(query.get("why_this_query", "")).strip(),
        evidence_for_analysis_json=json.dumps(evidence_for_analysis, ensure_ascii=False, indent=2),
        global_report=global_report.strip(),
    )
    content: list[dict[str, Any]] = []
    for path in image_paths:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})

    # Some transformers versions expect every message "content" to be a list
    # of typed blocks (including system messages), not a plain string.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": VL_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    # Primary path: follows Qwen3-VL README style.
    try:
        inputs = runtime.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        # Compatibility fallback for environments that require explicit
        # text/images inputs (common in older VL utility stacks).
        try:
            text_prompt = runtime.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Older stacks may still require plain string system content.
            legacy_messages = [
                {"role": "system", "content": VL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            text_prompt = runtime.processor.apply_chat_template(
                legacy_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        processor_kwargs: dict[str, Any] = {
            "text": [text_prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_paths:
            processor_kwargs["images"] = image_paths
        inputs = runtime.processor(**processor_kwargs)

    inputs = inputs.to(runtime.model.device)

    generated_ids = runtime.model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw = runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    try:
        parsed = _extract_last_json_object(raw)
        normalized = _normalize_vl_analysis_card(parsed=parsed, query=query)
        return raw, normalized, None
    except Exception as exc:
        return raw, None, str(exc)


def _run_final_judge(
    runtime: JudgeRuntime,
    *,
    judge_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]
    try:
        text = runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = runtime.tokenizer([text], return_tensors="pt").to(runtime.model.device)
    generated_ids = runtime.model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=runtime.tokenizer.pad_token_id,
        eos_token_id=runtime.tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    return runtime.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _normalize_final_judgement(text: str) -> str:
    cleaned = _strip_think_blocks(text)
    lines = [x.strip() for x in cleaned.splitlines() if x.strip()]
    if not lines:
        return "<Harmless>\n无法解析到有效输出，默认保守给出 Harmless。"

    first = lines[0]
    if first in {"<Harmful>", "<Harmless>"}:
        return "\n".join(lines)

    lower = cleaned.lower()
    label = "<Harmless>"
    if "harmful" in lower:
        label = "<Harmful>"
    if "harmless" in lower and "harmful" not in lower:
        label = "<Harmless>"
    return f"{label}\n{cleaned}"


def _default_save_path(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    return str(p.with_suffix(suffix))


def _print_evidence_query_result(
    *,
    rank: int,
    total: int,
    qid: str,
    query_type: str,
    parsed_json: dict[str, Any] | None,
    parse_error: str | None,
    raw_text: str,
) -> None:
    print(f"[Q{rank}/{total}] result query_id={qid} type={query_type}")
    if isinstance(parsed_json, dict):
        print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
        return

    print(f"[Q{rank}/{total}] parse_error={str(parse_error or '').strip()}")
    if str(raw_text or "").strip():
        preview = str(raw_text).strip()
        if len(preview) > 800:
            preview = preview[:800] + "...(truncated)"
        print(preview)


def run_evidence_judge_pipeline(
    *,
    queries_payload: Any,
    localization_payload: Any,
    global_report: str,
    harm_rules: str,
    video_path: str = "",
    qwen3_vl_model: str = DEFAULT_QWEN3_VL_MODEL_PATH,
    qwen3_vl_device: str | None = None,
    qwen3_vl_max_new_tokens: int = 768,
    max_visual_frames: int = 6,
    judge_model: str = DEFAULT_QWEN3_4B_MODEL_PATH,
    judge_device: str | None = None,
    judge_max_new_tokens: int = 1024,
    vl_runtime: Any | None = None,
    judge_runtime: Any | None = None,
) -> dict[str, Any]:
    queries = _normalize_queries(queries_payload)
    evidence_results = _normalize_evidence_results(localization_payload)
    evidence_index = _build_evidence_index(evidence_results)

    if not queries:
        raise ValueError("queries_payload 中未解析到有效 retrieval_queries。")
    if not evidence_results:
        raise ValueError("localization_payload 中未解析到 evidence_results。")
    if not str(global_report or "").strip():
        raise ValueError("global_report 不能为空。")
    if not str(harm_rules or "").strip():
        raise ValueError("harm_rules 不能为空。")

    frame_provider = FrameProvider(video_path)
    try:
        effective_vl_runtime = _coerce_external_vl_runtime(vl_runtime)
        if effective_vl_runtime is None:
            effective_vl_runtime = _load_vl_runtime(qwen3_vl_model, qwen3_vl_device)
        else:
            print("[提示] 使用外部传入的 Qwen3-VL runtime。")
        evidence_cards: list[dict[str, Any]] = []

        for rank, query in enumerate(queries, start=1):
            qid = str(query.get("id", "")).strip() or f"Q{rank}"
            evidence_item = evidence_index.get(qid, {})
            query_type = _normalize_card_query_type(query.get("query_type"), fallback="Visual")

            if query_type == "ASR":
                evidence_for_analysis = _derive_asr_evidence_for_analysis(evidence_item)
            else:
                evidence_for_analysis = _derive_visual_evidence_for_analysis(evidence_item)

            image_paths = _collect_query_images(
                query_type=query_type,
                evidence_item=evidence_item,
                frame_provider=frame_provider,
                max_visual_images=max(1, int(max_visual_frames)),
            )

            print(f"[Q{rank}/{len(queries)}] analyzing query_id={qid} type={query_type} images={len(image_paths)}")
            raw_text, parsed_json, parse_error = _run_vl_per_query(
                effective_vl_runtime,
                query=query,
                evidence_for_analysis=evidence_for_analysis,
                image_paths=image_paths,
                global_report=global_report,
                max_new_tokens=max(64, int(qwen3_vl_max_new_tokens)),
            )
            _print_evidence_query_result(
                rank=rank,
                total=len(queries),
                qid=qid,
                query_type=query_type,
                parsed_json=parsed_json,
                parse_error=parse_error,
                raw_text=raw_text,
            )

            evidence_cards.append(
                {
                    "query_id": qid,
                    "query_type": query_type,
                    "query_text": str(query.get("query_text", "")).strip(),
                    "why_this_query": str(query.get("why_this_query", "")).strip(),
                    "evidence_for_analysis": evidence_for_analysis,
                    "image_paths": image_paths,
                    "vl_analysis": parsed_json,
                    "vl_analysis_raw": raw_text,
                    "vl_parse_error": parse_error,
                }
            )

        evidence_cards_full_json = json.dumps(
            {"evidence_cards": evidence_cards},
            ensure_ascii=False,
            indent=2,
        )
        compact_cards = _build_compact_evidence_cards_for_judge(evidence_cards)
        evidence_card_compact_json = json.dumps(
            {"evidence_cards": compact_cards},
            ensure_ascii=False,
            indent=2,
        )

        judge_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            evidence_cards_json=evidence_card_compact_json,
            global_report=global_report.strip(),
            harm_rules=harm_rules.strip(),
        )

        effective_judge_runtime = _coerce_external_judge_runtime(judge_runtime)
        if effective_judge_runtime is None:
            effective_judge_runtime = _load_judge_runtime(judge_model, judge_device)
        else:
            print("[提示] 使用外部传入的最终审判 runtime。")
        judge_raw = _run_final_judge(
            effective_judge_runtime,
            judge_prompt=judge_prompt,
            max_new_tokens=max(128, int(judge_max_new_tokens)),
        )
        judge_normalized = _normalize_final_judgement(judge_raw)

        return {
            "evidence_cards_full": evidence_cards,
            "evidence_cards_compact": compact_cards,
            # Backward-compatible alias: "evidence_cards" remains the full cards.
            "evidence_cards": evidence_cards,
            # Backward-compatible key: evidence_card_json now stores full cards.
            "evidence_card_json": evidence_cards_full_json,
            "evidence_card_compact_json": evidence_card_compact_json,
            "judge_prompt": judge_prompt,
            "judge_raw": judge_raw,
            "judge_normalized": judge_normalized,
        }
    finally:
        frame_provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL 逐 query 证据分析 + Qwen3-4B 最终审判")
    parser.add_argument("--queries_json", type=str, required=True, help="base_Omni 的 query JSON 路径")
    parser.add_argument("--localization_json", type=str, required=True, help="focus_localizer 输出 JSON 路径")
    parser.add_argument("--global_report_txt", type=str, required=True, help="Omni 全局理解报告文本路径")
    parser.add_argument(
        "--harm_rules_txt",
        type=str,
        default="",
        help="有害内容分类规则文本路径；留空时默认使用脚本内置规则",
    )
    parser.add_argument(
        "--use_builtin_harm_rules",
        action="store_true",
        help="强制使用脚本内置 harm rules（会忽略 --harm_rules_txt）",
    )
    parser.add_argument("--video_path", type=str, default="", help="视频路径（用于按 frame_idx 回取帧）")
    parser.add_argument("--qwen3_vl_model", type=str, default=DEFAULT_QWEN3_VL_MODEL_PATH, help="Qwen3-VL 模型路径")
    parser.add_argument("--qwen3_vl_device", type=str, default=None, help="Qwen3-VL 运行设备")
    parser.add_argument("--qwen3_vl_max_new_tokens", type=int, default=768, help="Qwen3-VL 每个 query 最大生成 token")
    parser.add_argument("--max_visual_frames", type=int, default=6, help="Visual query 输入帧上限")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_QWEN3_4B_MODEL_PATH, help="审判模型路径")
    parser.add_argument("--judge_device", type=str, default=None, help="审判模型设备")
    parser.add_argument("--judge_max_new_tokens", type=int, default=1024, help="审判模型最大生成 token")
    parser.add_argument("--save_evidence_card_json", type=str, default="", help="evidence_card 输出 JSON 路径")
    parser.add_argument("--save_judge_input_txt", type=str, default="", help="审判输入拼接文本输出路径")
    parser.add_argument("--save_final_txt", type=str, default="", help="最终审判结果输出路径")
    args = parser.parse_args()

    queries_payload = _read_json(args.queries_json)
    localization_payload = _read_json(args.localization_json)
    global_report = _read_text(args.global_report_txt)
    harm_rules = _resolve_harm_rules(args.harm_rules_txt, args.use_builtin_harm_rules)

    result = run_evidence_judge_pipeline(
        queries_payload=queries_payload,
        localization_payload=localization_payload,
        global_report=global_report,
        harm_rules=harm_rules,
        video_path=args.video_path,
        qwen3_vl_model=args.qwen3_vl_model,
        qwen3_vl_device=args.qwen3_vl_device,
        qwen3_vl_max_new_tokens=args.qwen3_vl_max_new_tokens,
        max_visual_frames=args.max_visual_frames,
        judge_model=args.judge_model,
        judge_device=args.judge_device,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )

    evidence_card_json = str(result.get("evidence_card_json", ""))
    judge_prompt = str(result.get("judge_prompt", ""))
    judge_raw = str(result.get("judge_raw", ""))
    judge_normalized = str(result.get("judge_normalized", ""))

    save_evidence_card_json = args.save_evidence_card_json.strip() or _default_save_path(
        args.localization_json,
        ".evidence_card.json",
    )
    save_judge_input_txt = args.save_judge_input_txt.strip() or _default_save_path(
        args.localization_json,
        ".judge_input.txt",
    )
    save_final_txt = args.save_final_txt.strip() or _default_save_path(
        args.localization_json,
        ".judge_final.txt",
    )

    Path(save_evidence_card_json).parent.mkdir(parents=True, exist_ok=True)
    Path(save_judge_input_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(save_final_txt).parent.mkdir(parents=True, exist_ok=True)

    Path(save_evidence_card_json).write_text(evidence_card_json, encoding="utf-8")
    Path(save_judge_input_txt).write_text(judge_prompt, encoding="utf-8")
    Path(save_final_txt).write_text(
        "\n".join(
            [
                "[raw_output]",
                judge_raw,
                "",
                "[normalized_output]",
                judge_normalized,
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("Evidence card saved:", save_evidence_card_json)
    print("Judge input saved:", save_judge_input_txt)
    print("Final judgement saved:", save_final_txt)
    print("=" * 60)
    print(judge_normalized)


if __name__ == "__main__":
    main()
