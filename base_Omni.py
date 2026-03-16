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


def run_native_video_inference(
    video_path: str,
    prompt: str,
    model_path: str = "/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
    output_audio_path: str = None,
    disable_talker: bool = False,
):
    """原生视频输入推理，配置与 use_Omni.py 保持一致"""

    # ============ 加载模型（与 use_Omni.py 一致）============
    print("=" * 60)
    print("加载模型")
    print("=" * 60)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

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
                "You are Qwen, a virtual human developed by the Qwen Team, "
                "Alibaba Group, capable of perceiving auditory and visual inputs, "
                "as well as generating text and speech."
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
    inputs = inputs.to(model.device).to(model.dtype)

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
        text_ids, audio_output = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

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
    parser.add_argument("--video", type=str, default="/sda/yuqifan/HFOCUS/test/1.mp4")
    parser.add_argument("--prompt", type=str, default="请分析这段视频的内容")
    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni")
    parser.add_argument("--output_audio", type=str, default=None)
    parser.add_argument("--disable_talker", action="store_true")
    args = parser.parse_args()

    run_native_video_inference(
        video_path=args.video,
        prompt=args.prompt,
        model_path=args.model,
        output_audio_path=args.output_audio,
        disable_talker=args.disable_talker,
    )