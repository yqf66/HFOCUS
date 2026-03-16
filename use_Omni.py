"""
Qwen2.5-Omni 推理脚本（集成 VideoFramePipeline）
=====================================================
流程：
    1. VideoFramePipeline 处理视频 → 得到视频流 X'（保底帧 + 事件帧）
    2. ffmpeg 提取视频音频轨道 → 临时 WAV 文件
    3. X' 帧序列（作为多张图片）+ 音频 一起送入 Qwen2.5-Omni 推理
    4. 输出文本结果（可选输出语音）

使用方式：
    python infer_with_pipeline.py --video /path/to/video.mp4 \
        --mode low --fixed_frames 40 \
        --prompt "请分析这段视频内容"
"""

import argparse
import os
import sys
import cv2
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from PIL import Image

# ===================== 导入模型相关 =====================
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ===================== 导入 VideoFramePipeline =====================
from single import VideoFramePipeline, PipelineResult


# ===================== 音频提取 =====================
def extract_audio_from_video(video_path: str, output_wav: str, sr: int = 16000) -> bool:
    """
    用 ffmpeg 从视频中提取音频并转为 WAV 格式
    
    Args:
        video_path: 视频文件路径
        output_wav: 输出 WAV 文件路径
        sr: 采样率，Qwen2.5-Omni 音频编码器通常接受 16kHz
    
    Returns:
        是否成功提取到音频
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",                    # 不要视频流
            "-acodec", "pcm_s16le",   # 16bit PCM
            "-ar", str(sr),           # 采样率
            "-ac", "1",               # 单声道
            output_wav
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(output_wav):
            file_size = os.path.getsize(output_wav)
            if file_size > 44:  # WAV header 至少 44 bytes
                print(f"[音频提取] 成功: {output_wav} ({file_size / 1024:.1f} KB)")
                return True
        # 可能视频没有音频轨道
        print(f"[音频提取] 视频无音频轨道或提取失败")
        return False
    except FileNotFoundError:
        print("[音频提取] 未找到 ffmpeg，请先安装 ffmpeg")
        return False
    except Exception as e:
        print(f"[音频提取] 异常: {e}")
        return False


# ===================== 帧转换 =====================
def frames_to_pil(
    frames: list[np.ndarray],
    tags: list[str] | None = None,
    scores: list[float] | None = None,
    frame_indices: list[int] | None = None,
    fps: float = 25.0,
) -> list[Image.Image]:
    """
    将 pipeline 输出的 RGB numpy 帧转为 PIL Image 列表。
    若提供 tags/scores/frame_indices，则在每帧顶部绘制标注信息，
    让模型能感知该帧的类型（背景/事件）及时间定位。
    """
    pil_images = []
    for i, frame in enumerate(frames):
        # 转为 BGR 供 cv2 绘制，再转回 RGB
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if tags is not None and frame_indices is not None:
            tag   = tags[i]         if i < len(tags)          else "baseline"
            idx   = frame_indices[i] if i < len(frame_indices) else 0
            score = scores[i]       if scores and i < len(scores) else 0.0
            time_sec = idx / fps

            # 半透明顶部色条（事件帧红色，背景帧深灰）
            bar_color = (0, 0, 160) if tag == "event" else (60, 60, 60)  # BGR
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 36), bar_color, -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

            label = f"[{i:03d}] {'EVENT' if tag == 'event' else 'BASE'}"
            if tag == "event":
                label += f"  score={score:.2f}"
            label += f"  t={time_sec:.2f}s  frame#{idx}"

            cv2.putText(
                img, label, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return pil_images


# ===================== 构造 conversation =====================
def build_conversation(
    pil_frames: list[Image.Image],
    audio_path: str | None,
    prompt: str,
    system_prompt: str | None = None,
) -> tuple[list[dict], bool]:
    """
    构造 Qwen2.5-Omni 的 conversation 格式
    
    将 X' 帧作为多张图片传入，音频单独作为 audio 类型传入。
    模型会同时处理视觉帧和音频信息。
    
    Args:
        pil_frames: PIL Image 列表（X' 视频流）
        audio_path: 音频文件路径，None 表示无音频
        prompt: 用户提问文本
        system_prompt: 系统提示词
    
    Returns:
        (conversation, use_audio_in_video)
        use_audio_in_video 在这里设为 False，因为我们不是传视频，
        而是分别传图片和音频
    """
    if system_prompt is None:
        system_prompt = (
            "You are a multimodal AI assistant capable of perceiving "
            "visual and auditory inputs. You will receive a sequence of "
            "video frames (as images) and an audio track extracted from "
            "a video. Analyze them together to answer the user's question."
        )

    # 构造 user content：N 张图片 + 可选音频 + 文本提问
    user_content = []

    # 图片帧（X' 视频流）
    for img in pil_frames:
        user_content.append({"type": "image", "image": img})

    # 音频
    if audio_path and os.path.exists(audio_path):
        user_content.append({"type": "audio", "audio": audio_path})

    # 文本提问（附带帧信息说明）
    frame_info = (
        f"[以上共 {len(pil_frames)} 帧图片，来自视频的关键帧采样。"
        f"每帧左上角标注了帧类型（BASE=背景帧 / EVENT=异常事件帧）、"
        f"事件帧的异常分数（score）以及该帧在原视频中的时间戳（t=秒）。"
        f"请结合这些标注信息理解视频的时序结构"
    )
    if audio_path and os.path.exists(audio_path):
        frame_info += "，同时参考音频内容"
    frame_info += f"，回答以下问题]\n\n{prompt}"

    user_content.append({"type": "text", "text": frame_info})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    # 不使用 use_audio_in_video，因为我们没有传视频
    use_audio_in_video = False

    return conversation, use_audio_in_video


# ===================== 推理主函数 =====================
def run_inference(
    video_path: str,
    prompt: str,
    model_path: str = "Qwen/Qwen2.5-Omni-7B",
    baseline_mode: str = "low",
    baseline_fixed_frames: int = 40,
    baseline_fps: float = 2.0,
    system_prompt: str | None = None,
    output_audio_path: str | None = None,
    disable_talker: bool = False,
    # pipeline 额外参数
    event_sample_fps: int = 10,
    flash_sensitivity: float = 2.5,
    anomaly_sensitivity: float = 2.0,
    max_events: int = 30,
):
    """
    完整推理流程

    Args:
        video_path: 输入视频路径
        prompt: 用户提问
        model_path: 模型路径或 HuggingFace ID
        baseline_mode: "low"(固定帧数) 或 "high"(按FPS)
        baseline_fixed_frames: low 模式固定帧数
        baseline_fps: high 模式每秒帧数
        system_prompt: 自定义系统提示词
        output_audio_path: 输出语音保存路径，None 则不保存
        disable_talker: 是否禁用语音生成（省约 2GB 显存）
        event_sample_fps: 事件检测密集采样帧率
        flash_sensitivity: 闪帧检测灵敏度
        anomaly_sensitivity: 异常检测灵敏度
        max_events: 最大事件帧数

    Returns:
        (text_output, pipeline_result)
    """
    # ============ Step 1: VideoFramePipeline 处理视频 ============
    print("=" * 60)
    print("Step 1: VideoFramePipeline 帧采样")
    print("=" * 60)

    pipeline = VideoFramePipeline(
        baseline_mode=baseline_mode,
        baseline_fixed_frames=baseline_fixed_frames,
        baseline_fps=baseline_fps,
        event_sample_fps=event_sample_fps,
        flash_sensitivity=flash_sensitivity,
        anomaly_sensitivity=anomaly_sensitivity,
        max_events=max_events,
    )

    result = pipeline.process(video_path)
    print(result.summary())

    if not result.frames_X_prime:
        print("[错误] Pipeline 未能提取到任何帧")
        return None, result

    # 转为 PIL Image（带帧类型和时间戳标注）
    pil_frames = frames_to_pil(
        result.frames_X_prime,
        tags=result.source_tags,
        scores=result.merged_scores,
        frame_indices=result.indices_X_prime,
        fps=result.video_info.fps if result.video_info.fps > 0 else 25.0,
    )
    print(f"\nX' 帧数: {len(pil_frames)} (保底 + 事件)")
    if result.event_count > 0:
        print(f"事件帧: {result.event_count} 个 "
              f"(max_score={max(result.scores_a):.3f})")

    # ============ Step 2: 提取音频 ============
    print("\n" + "=" * 60)
    print("Step 2: 提取音频轨道")
    print("=" * 60)

    tmp_audio = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, prefix="omni_audio_"
    )
    tmp_audio_path = tmp_audio.name
    tmp_audio.close()

    has_audio = extract_audio_from_video(video_path, tmp_audio_path)
    audio_path_for_model = tmp_audio_path if has_audio else None

    # ============ Step 3: 加载模型 ============
    print("\n" + "=" * 60)
    print("Step 3: 加载 Qwen2.5-Omni 模型")
    print("=" * 60)

    print(f"模型: {model_path}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2",  # 如需启用取消注释
    )

    if disable_talker:
        model.disable_talker()
        print("已禁用 Talker（不生成语音，节省 ~2GB 显存）")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("模型加载完成")

    # ============ Step 4: 构造输入并推理 ============
    print("\n" + "=" * 60)
    print("Step 4: 构造输入 & 推理")
    print("=" * 60)

    conversation, use_audio_in_video = build_conversation(
        pil_frames=pil_frames,
        audio_path=audio_path_for_model,
        prompt=prompt,
        system_prompt=system_prompt,
    )

    # 预处理
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    # max_pixels 限制每帧分辨率：28*28*256 ≈ 448×448，可按显存调整
    # 25帧 × 1024patch × 每patch token数 → attention mask 可控
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
        max_pixels=28 * 28 * 256,   # ≈ 448×448 per frame
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print(f"输入构造完成, 开始推理...")
    print(f"  图片数: {len(images) if images else 0}")
    print(f"  音频数: {len(audios) if audios else 0}")
    print(f"  视频数: {len(videos) if videos else 0}")

    # 推理
    if disable_talker:
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
        )
        audio_output = None
    else:
        text_ids, audio_output = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
        )

    # 解码文本
    text_output = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # ============ Step 5: 输出结果 ============
    print("\n" + "=" * 60)
    print("Step 5: 推理结果")
    print("=" * 60)

    if isinstance(text_output, list):
        for i, t in enumerate(text_output):
            print(f"\n--- 输出 {i+1} ---")
            print(t)
    else:
        print(text_output)

    # 保存音频
    if audio_output is not None and output_audio_path:
        audio_np = audio_output.reshape(-1).detach().cpu().numpy()
        sf.write(output_audio_path, audio_np, samplerate=24000)
        print(f"\n语音已保存: {output_audio_path}")

    # 清理临时文件
    if os.path.exists(tmp_audio_path):
        os.unlink(tmp_audio_path)

    return text_output, result


# ===================== CLI 入口 =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni 推理（集成 VideoFramePipeline）"
    )
    # 基本参数
    parser.add_argument("--video", type=str, default="/sda/yuqifan/HFOCUS/test/1.mp4",
                        help="输入视频路径")
    parser.add_argument("--prompt", type=str, default="请分析这段视频的内容",
                        help="提问内容")
    parser.add_argument("--model", type=str, default="/sda/yuqifan/HFOCUS/Qwen2.5-Omni",
                        help="模型路径或 HuggingFace ID")

    # Pipeline 参数
    parser.add_argument("--mode", type=str, default="low", choices=["low", "high"],
                        help="保底帧采样模式: low(固定帧数) / high(按FPS)")
    parser.add_argument("--fixed_frames", type=int, default=40,
                        help="low 模式固定帧数")
    parser.add_argument("--sample_fps", type=float, default=2.0,
                        help="high 模式每秒帧数")

    # 事件检测参数
    parser.add_argument("--event_fps", type=int, default=10,
                        help="事件检测密集采样帧率")
    parser.add_argument("--flash_sens", type=float, default=2.5,
                        help="闪帧检测灵敏度")
    parser.add_argument("--anomaly_sens", type=float, default=2.0,
                        help="异常检测灵敏度")
    parser.add_argument("--max_events", type=int, default=20,
                        help="最大事件帧数")

    # 输出参数
    parser.add_argument("--output_audio", type=str, default=None,
                        help="输出语音保存路径（如 output.wav）")
    parser.add_argument("--disable_talker", action="store_true",
                        help="禁用语音生成，节省 ~2GB 显存")

    # 系统提示词
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="自定义系统提示词")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    text_output, pipeline_result = run_inference(
        video_path=args.video,
        prompt=args.prompt,
        model_path=args.model,
        baseline_mode=args.mode,
        baseline_fixed_frames=args.fixed_frames,
        baseline_fps=args.sample_fps,
        system_prompt=args.system_prompt,
        output_audio_path=args.output_audio,
        disable_talker=args.disable_talker,
        event_sample_fps=args.event_fps,
        flash_sensitivity=args.flash_sens,
        anomaly_sensitivity=args.anomaly_sens,
        max_events=args.max_events,
    )