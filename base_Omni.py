"""
Qwen2.5-Omni 模块 A：全局视频理解（原生视频 + 音频）
用途：输入原生视频，输出自然语言全局理解报告（非 JSON）
"""

import argparse
from pathlib import Path

import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
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


def build_global_understanding_prompt(user_focus: str = "") -> str:
    """构建模块 A 的用户提示词。"""
    prompt = GLOBAL_UNDERSTANDING_PROMPT
    if user_focus.strip():
        prompt = f"{prompt}\n\nAdditional focus: {user_focus.strip()}"
    return prompt


def _get_module_device(module: torch.nn.Module) -> torch.device | None:
    """返回模块中第一个非-meta 参数所在设备。"""
    try:
        for param in module.parameters():
            if torch.is_tensor(param) and param.device.type != "meta":
                return param.device
    except Exception:
        pass
    return None


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

    # 保持仅文本输出
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

    # 优先仅解码新增 token，避免把 prompt 一并解码进输出
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


def _resolve_save_path(video_path: str, save_txt: str) -> Path:
    """解析报告保存路径：支持 auto（同名 .txt）或用户指定路径。"""
    if save_txt == "auto":
        video = Path(video_path)
        return video.with_suffix(".txt")
    return Path(save_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 模块 A：全局视频理解报告生成")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--focus", type=str, default="", help="可选用户关注点")
    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni", help="模型路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备，如 cuda:0 / cpu")
    parser.add_argument("--device_map", type=str, default="auto", help="HF device_map，默认 auto")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成最大 token 数")
    parser.add_argument(
        "--save_txt",
        nargs="?",
        const="auto",
        default=None,
        help="可选保存报告文本。仅写 --save_txt 时自动保存为同名 .txt；也可指定输出路径",
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
