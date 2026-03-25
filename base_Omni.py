"""
Qwen2.5-Omni 模块 A：全局视频理解（原生视频 + 音频）
用途：输入原生视频，输出自然语言全局理解报告（非 JSON）
并可选衔接模块 C：Query 提炼（文本小模型）
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


GLOBAL_UNDERSTANDING_PROMPT = """You are a multimodal video evidence analyst.

Watch the video and produce a structured global understanding report
optimized for downstream evidence retrieval.

Do NOT output JSON.
Do NOT give a final harmful-content label.

=====================
Key Requirements:
=====================

1. All events MUST include temporal grounding.
   - Use approximate timestamps (e.g., [00:03–00:06])
   - If exact timing is unclear, give a rough segment.

2. Assign each key event an Event ID: (E1, E2, E3...)

3. Unclear points MUST explicitly reference related Event IDs.

4. Focus on making the output useful for:
   - later frame retrieval
   - evidence verification

5. Separate observation and interpretation strictly.

=====================
Output Format:
=====================

[Scene]
- setting, people, roles, context

[Observed Visual Events]
E1 [00:00–00:03]:
- [Observed] ...

E2 [00:03–00:06]:
- [Observed] ...

E3 [00:06–00:10]:
- [Observed] ...

[Observed Audio]
A1 [00:05–00:07]:
- [Observed] speech: "..."
- clarity: clear / unclear

[Main Interpretation]
- [Inferred] ...

[Alternative Interpretation]
- [Inferred] ...

[Key Unclear Points]
U1 (related to E2, A1):
- what is unclear
- what needs verification

U2 (related to E3):
- ...

[Preliminary Conclusion]
- ...
- confidence: low / medium / high"""


QUERY_EXTRACTION_PROMPT = """You are a query extraction assistant for video evidence retrieval.

Input:
A natural-language global video understanding report written by a multimodal model.

Your only task:
Extract 1 to 3 high-value retrieval queries that will help verify the most important unclear points in the report.

Important rules:
1. Do NOT invent any new fact.
2. Only use unclear points, ambiguity, or key decision points explicitly mentioned in the report.
3. Keep each query short, specific, and easy to use for later retrieval or verification.
4. Prefer fewer but more important queries.
5. If the report is already clear enough, output an empty list.

Output valid JSON only in this format:

{
  "retrieval_queries": [
    {
      "id": "Q1",
      "time_hint": "",
      "query_text": "",
      "why_this_query": ""
    }
  ]
}"""


@dataclass
class RetrievalQuery:
    id: str
    time_hint: str
    query_text: str
    why_this_query: str


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
            "retrieval_queries": [asdict(q) for q in self.retrieval_queries],
            "parse_error": self.parse_error,
            "parse_error_message": self.parse_error_message,
        }


@dataclass
class GlobalPipelineResult:
    report_text: str
    query_result: QueryExtractionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_text": self.report_text,
            "query_result": self.query_result.to_dict(),
        }


def build_global_understanding_prompt(user_focus: str = "") -> str:
    """构建模块 A 的用户提示词。"""
    prompt = GLOBAL_UNDERSTANDING_PROMPT
    if user_focus.strip():
        prompt = f"{prompt}\n\nAdditional focus: {user_focus.strip()}"
    return prompt


def build_query_extraction_prompt(report_text: str) -> str:
    """构建模块 C 的提示词（Prompt 2 + 全局报告）。"""
    return f"{QUERY_EXTRACTION_PROMPT}\n\nGlobal report:\n{report_text.strip()}"


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


def _validate_and_normalize_queries(payload: dict[str, Any]) -> tuple[list[RetrievalQuery], list[str]]:
    """对 retrieval_queries 做最低必要校验，并裁剪到最多 3 条。"""
    errors: list[str] = []
    normalized: list[RetrievalQuery] = []

    queries = payload.get("retrieval_queries")
    if not isinstance(queries, list):
        return [], ["retrieval_queries must be a list."]

    for idx, item in enumerate(queries):
        if len(normalized) >= 3:
            break
        if not isinstance(item, dict):
            errors.append(f"retrieval_queries[{idx}] must be an object.")
            continue

        missing_keys = [
            key
            for key in ["id", "time_hint", "query_text", "why_this_query"]
            if key not in item
        ]
        if missing_keys:
            errors.append(f"retrieval_queries[{idx}] missing keys: {', '.join(missing_keys)}")
            continue

        query_text = str(item.get("query_text", "")).strip()
        if not query_text:
            continue

        qid = str(item.get("id", "")).strip() or f"Q{len(normalized) + 1}"
        time_hint = str(item.get("time_hint", "")).strip()
        why_this_query = str(item.get("why_this_query", "")).strip()

        normalized.append(
            RetrievalQuery(
                id=qid,
                time_hint=time_hint,
                query_text=query_text,
                why_this_query=why_this_query,
            )
        )

    return normalized, errors


def run_global_video_understanding(
    video_path: str,
    user_focus: str = "",
    model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    device: str | None = None,
    device_map: str | None = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """模块 A：输入原生视频，返回自然语言全局理解报告。"""

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

    print("\n" + "=" * 60)
    print("模块 A：构造输入并开始推理")
    print("=" * 60)

    system_prompt = (
        "You are a powerful multimodal assistant specialized in video understanding. "
        "Jointly use visual and audio evidence from the same video. "
        "Pay attention to actions, scene context, temporal order, spoken content, tone, "
        "background sounds, and meaningful acoustic cues. "
        "Do not rely on visual stream alone when audio is informative. "
        "Ground conclusions in evidence and clearly distinguish observation from interpretation."
    )

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": build_global_understanding_prompt(user_focus)},
            ],
        },
    ]

    use_audio_in_video = True

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        value = value.to(input_device)
        if value.is_floating_point():
            value = value.to(model_dtype)
        inputs[key] = value

    print(f"输入视频: {video_path}")
    print(f"音频轨道: 启用 (use_audio_in_video={use_audio_in_video})")
    print(f"user_focus: {user_focus if user_focus.strip() else '(none)'}")
    print(f"max_new_tokens: {max_new_tokens}")
    print("开始推理...")

    output_ids = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        max_new_tokens=max_new_tokens,
    )

    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids) and output_ids.shape[1] >= input_ids.shape[1]:
        generated_ids = output_ids[:, input_ids.shape[1] :]
    else:
        generated_ids = output_ids

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    report_text = decoded[0].strip() if isinstance(decoded, list) and decoded else str(decoded).strip()

    print("\n" + "=" * 60)
    print("模块 A：全局理解报告")
    print("=" * 60)
    print(report_text)

    return report_text


def run_query_extraction(
    report_text: str,
    query_model_path: str,
    device: str | None = None,
    max_new_tokens: int = 512,
) -> QueryExtractionResult:
    """模块 C：从全局理解报告中提炼 1~3 条检索 Query（JSON 输出）。"""

    print("\n" + "=" * 60)
    print("模块 C：加载 Query 模型")
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

    print("Query 模型加载完成")

    print("\n" + "=" * 60)
    print("模块 C：开始 Query 提炼")
    print("=" * 60)

    prompt = build_query_extraction_prompt(report_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids) and outputs.shape[1] >= input_ids.shape[1]:
        generated_ids = outputs[:, input_ids.shape[1] :]
    else:
        generated_ids = outputs

    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_text = decoded[0].strip() if isinstance(decoded, list) and decoded else str(decoded).strip()

    print("原始 Query 模型输出:")
    print(raw_text)

    parsed_json: dict[str, Any] | None = None
    retrieval_queries: list[RetrievalQuery] = []
    parse_error = False
    parse_error_message: str | None = None

    try:
        parsed_json = extract_last_json_object(raw_text)
        retrieval_queries, validation_errors = _validate_and_normalize_queries(parsed_json)
        if validation_errors:
            parse_error = True
            parse_error_message = "; ".join(validation_errors)
    except Exception as exc:
        parse_error = True
        parse_error_message = str(exc)

    final_query_json = {"retrieval_queries": [asdict(q) for q in retrieval_queries]}

    print(f"解析后的 Query 数量: {len(retrieval_queries)}")
    print("最终 Query JSON:")
    print(json.dumps(final_query_json, ensure_ascii=False, indent=2))

    return QueryExtractionResult(
        raw_text=raw_text,
        parsed_json=parsed_json,
        retrieval_queries=retrieval_queries,
        parse_error=parse_error,
        parse_error_message=parse_error_message,
    )


def run_global_understanding_and_query_extraction(
    video_path: str,
    user_focus: str = "",
    omni_model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    query_model_path: str = "/sda/yuqifan/HFOCUS/Qwen3-0.6B",
    omni_device: str | None = None,
    omni_device_map: str | None = "auto",
    query_device: str | None = None,
    omni_max_new_tokens: int = 1024,
    query_max_new_tokens: int = 512,
) -> GlobalPipelineResult:
    """总流程：模块 A 生成报告，再由模块 C 提炼 Query。"""

    report_text = run_global_video_understanding(
        video_path=video_path,
        user_focus=user_focus,
        model_path=omni_model_path,
        device=omni_device,
        device_map=omni_device_map,
        max_new_tokens=omni_max_new_tokens,
    )

    query_result = run_query_extraction(
        report_text=report_text,
        query_model_path=query_model_path,
        device=query_device,
        max_new_tokens=query_max_new_tokens,
    )

    return GlobalPipelineResult(report_text=report_text, query_result=query_result)


def _resolve_save_path(video_path: str, save_txt: str) -> Path:
    """解析报告保存路径：支持 auto（同名 .txt）或用户指定路径。"""
    if save_txt == "auto":
        video = Path(video_path)
        return video.with_suffix(".txt")
    return Path(save_txt)


def _resolve_query_json_save_path(video_path: str, save_query_json: str) -> Path:
    """解析 Query JSON 保存路径：支持 auto（同名 .queries.json）或用户指定路径。"""
    if save_query_json == "auto":
        video = Path(video_path)
        return video.with_suffix(".queries.json")
    return Path(save_query_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 模块 A/C 推理脚本")

    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--focus", type=str, default="", help="可选用户关注点")

    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni", help="Omni 模型路径")
    parser.add_argument("--device", type=str, default=None, help="Omni 运行设备，如 cuda:0 / cpu")
    parser.add_argument("--device_map", type=str, default="auto", help="Omni HF device_map，默认 auto")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="模块 A 最大生成 token 数")

    parser.add_argument("--run_query_extraction", action="store_true", help="启用模块 C：Query 提炼")
    parser.add_argument("--query_model", type=str, default="/sda/yuqifan/HFOCUS/Qwen3-0.6B", help="Query 文本模型路径")
    parser.add_argument("--query_device", type=str, default=None, help="Query 模型运行设备，如 cuda:0 / cpu")
    parser.add_argument("--query_max_new_tokens", type=int, default=512, help="模块 C 最大生成 token 数")

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

    args = parser.parse_args()

    report = run_global_video_understanding(
        video_path=args.video,
        user_focus=args.focus,
        model_path=args.model,
        device=args.device,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
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
        )

        query_json_payload = {"retrieval_queries": [asdict(q) for q in query_result.retrieval_queries]}
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
    elif args.save_query_json is not None:
        print("\n[提示] 已指定 --save_query_json，但未启用 --run_query_extraction，跳过保存。")
