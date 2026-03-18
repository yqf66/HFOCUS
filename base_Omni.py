"""
Qwen2.5-Omni 原生视频推理（对照组）
用途：与 VideoFramePipeline + 图片序列方案做对比
保持模型配置、prompt 等一致
"""

import argparse
import soundfile as sf
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


def _get_module_device(module: torch.nn.Module) -> torch.device | None:
    """返回模块中第一个非-meta 参数所在设备。"""
    try:
        for param in module.parameters():
            if torch.is_tensor(param) and param.device.type != "meta":
                return param.device
    except Exception:
        pass
    return None


def run_native_video_inference(
    video_path: str,
    prompt: str,
    model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    output_audio_path: str = None,
    disable_talker: bool = False,
    device: str | None = None,
    device_map: str | None = "auto",
):
    """原生视频输入推理，配置与 use_Omni.py 保持一致"""

    # ============ 加载模型（与 use_Omni.py 一致）============
    print("=" * 60)
    print("加载模型")
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

    if disable_talker:
        model.disable_talker()
        print("已禁用 Talker")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("模型加载完成")

    # ============ 构造输入（原生视频）============
    print("\n" + "=" * 60)
    print("构造输入 & 推理")
    print("=" * 60)

    # 使用默认 system prompt，保证音频正常工作
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": 
                "You are a powerful multimodal assistant specialized in video understanding. "
                "You can analyze both visual and audio information in videos, and you should always "
                "jointly use what is seen and what is heard to understand the content. "
                "Pay close attention to actions, objects, scenes, temporal changes, spoken language, "
                "background sounds, tone, and other auditory cues. "
                "Do not rely only on the visual stream when audio provides important evidence. "
                "When answering, prioritize accuracy, ground your response in the video evidence, "
                "and clearly reflect the role of both visual and audio information whenever relevant."
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = True

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        value = value.to(input_device)
        if value.is_floating_point():
            value = value.to(model_dtype)
        inputs[key] = value

    print(f"视频: {video_path}")
    print(f"音频轨道: 由模型原生处理 (use_audio_in_video=True)")
    print(f"Prompt: {prompt}")
    print("开始推理...")

    # ============ 推理 ============
    if disable_talker:
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False,
        )
        audio_output = None
    else:
        try:
            text_ids, audio_output = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
        except RuntimeError as e:
            err = str(e)
            if "Expected all tensors to be on the same device" in err:
                print("[警告] 多卡+Talker 触发跨卡拼接错误，自动回退为仅文本输出（保留多卡 Thinker 推理）。")
                text_ids = model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False,
                )
                audio_output = None
            else:
                raise

    # ============ 输出 ============
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)

    text_output = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    if isinstance(text_output, list):
        for i, t in enumerate(text_output):
            print(f"\n--- 输出 {i+1} ---")
            print(t)
    else:
        print(text_output)

    if audio_output is not None and output_audio_path:
        sf.write(
            output_audio_path,
            audio_output.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"\n语音已保存: {output_audio_path}")

    return text_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 原生视频推理（对照组）")
    parser.add_argument("--video", type=str, default="/sda/yuqifan/HFOCUS/test/3.mp4")
    parser.add_argument("--prompt", type=str, default="请重点分析这段视频中的音频内容，尤其是人物对话、说话顺序、关键语句、语气和情绪变化。"
                                                      "先尽可能准确概括或转写音频中的主要对话内容，再结合视频画面判断是谁在说话、他们在做什么、"
                                                      "场景是什么，以及画面与对话之间的关系。"
                                                      "如果画面信息与音频信息有冲突或补充，请明确说明。"
                                                      "回答时请优先依据音频理解视频含义，不要只根据画面做概括。")
    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni")
    parser.add_argument("--output_audio", type=str, default=None)
    parser.add_argument("--disable_talker", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="运行设备，如 cuda:0 / cpu（默认自动选择）")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="HF device_map，默认 auto（多卡分片）；设为 none 可强制单卡")
    args = parser.parse_args()

    run_native_video_inference(
        video_path=args.video,
        prompt=args.prompt,
        model_path=args.model,
        output_audio_path=args.output_audio,
        disable_talker=args.disable_talker,
        device=args.device,
        device_map=args.device_map,
    )
