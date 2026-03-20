"""
短视频帧采样流水线（即插即用）
==================================
架构：
    X  = 均匀下采样保底帧（确保全程覆盖，应对全程平缓有害视频）
    a  = 事件突变帧（捕捉转瞬即逝的异常画面）
    X' = a 按时序插入 X 并去重 → 最终输入下游理解分析模块

资源模式：
    low  — 固定总帧数（如40帧），无论视频多长都均匀采样固定数量
    high — 按FPS采样（如每秒2帧），帧数随视频时长线性增长

输出：
    result.frames_X_prime / indices_X_prime  → 合并后的完整视频流 X'
    result.event_windows                      → 事件时间窗（核心输出）
    result.frames_a / indices_a / scores_a   → 兼容输出：高异常候选帧集合 a

使用方式：
    from video_frame_pipeline import VideoFramePipeline

    # 低资源模式：固定采40帧保底
    pipeline = VideoFramePipeline(
        baseline_mode="low",
        baseline_fixed_frames=40,
    )

    # 高资源模式：每秒采2帧保底
    pipeline = VideoFramePipeline(
        baseline_mode="high",
        baseline_fps=2.0,
    )

    result = pipeline.process(video_path="your_video.mp4")
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.signal import find_peaks
import warnings
import os
import logging

warnings.filterwarnings("ignore")

# ===================== 日志 =====================
logger = logging.getLogger("VideoFramePipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(name)s %(levelname)s] %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ===================== 数据结构 =====================
@dataclass
class VideoInfo:
    """视频元信息"""
    path: str = ""
    total_frames: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class PipelineResult:
    """
    流水线输出结果
    同时提供合并流 X' 和独立事件帧 a，下游按需取用
    """
    # ===== 合并帧流 X'（保底 + 事件，按时序排列）=====
    frames_X_prime: List[np.ndarray] = field(default_factory=list)
    indices_X_prime: List[int] = field(default_factory=list)
    source_tags: List[str] = field(default_factory=list)      # "baseline" / "event"
    merged_scores: List[float] = field(default_factory=list)   # 事件帧带分数，保底帧为0.0

    # ===== 独立事件帧 a（仅事件帧，按分数降序）=====
    frames_a: List[np.ndarray] = field(default_factory=list)
    indices_a: List[int] = field(default_factory=list)
    scores_a: List[float] = field(default_factory=list)

    # ===== 事件时间窗（核心输出）=====
    event_windows: List["EventWindow"] = field(default_factory=list)
    window_count: int = 0
    window_peak_indices: List[int] = field(default_factory=list)
    window_dense_index_dict: Dict[int, List[int]] = field(default_factory=dict)

    # ===== 保底帧 X（仅保底帧）=====
    indices_X: List[int] = field(default_factory=list)

    # ===== 元信息 =====
    video_info: VideoInfo = field(default_factory=VideoInfo)
    baseline_mode: str = ""               # "low" / "high"
    baseline_frame_count: int = 0         # 保底帧数量
    event_count: int = 0                  # 兼容字段：高异常候选帧数量
    total_output_frames: int = 0          # X' 总帧数

    def summary(self) -> str:
        mode_desc = (
            f"固定{self.baseline_frame_count}帧"
            if self.baseline_mode == "low"
            else f"按FPS采样{self.baseline_frame_count}帧"
        )
        lines = [
            f"视频: {os.path.basename(self.video_info.path)}",
            f"时长: {self.video_info.duration_sec:.1f}s | "
            f"原始FPS: {self.video_info.fps:.1f} | "
            f"总帧数: {self.video_info.total_frames}",
            f"模式: {self.baseline_mode} ({mode_desc})",
            f"保底帧(X): {self.baseline_frame_count} 帧",
            f"事件时间窗: {self.window_count} 个",
            f"高异常候选帧(a): {self.event_count} 帧",
            f"合并帧(X'): {self.total_output_frames} 帧",
        ]
        if self.event_windows:
            durations = [w.end_time - w.start_time for w in self.event_windows]
            peak_scores = [w.peak_score for w in self.event_windows]
            lines.append(
                f"窗口时长(秒): avg={np.mean(durations):.3f}, "
                f"min={min(durations):.3f}, max={max(durations):.3f}"
            )
            lines.append(
                f"峰值分数: max={max(peak_scores):.3f}, "
                f"min={min(peak_scores):.3f}, avg={np.mean(peak_scores):.3f}"
            )
            for w in self.event_windows:
                lines.append(
                    f"  - W{w.window_id:02d} "
                    f"F[{w.start_frame}-{w.end_frame}] "
                    f"T[{w.start_time:.2f}-{w.end_time:.2f}]s "
                    f"peak=F{w.peak_frame}({w.peak_score:.3f}) "
                    f"dense={len(w.dense_indices)}"
                )
        elif self.scores_a:
            lines.append(
                f"事件帧分数: max={max(self.scores_a):.3f}, "
                f"min={min(self.scores_a):.3f}, "
                f"avg={np.mean(self.scores_a):.3f}"
            )
        return "\n".join(lines)


@dataclass
class EventWindow:
    """异常事件时间窗"""
    window_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    peak_frame: int
    peak_score: float
    dense_indices: List[int] = field(default_factory=list)
    dense_scores: List[float] = field(default_factory=list)


# ===================== 事件帧检测器 =====================
class EventFrameDetector:
    """
    事件突变帧检测器
    职责：从密集采样的灰度帧序列中定位时序突变点
    方法：帧间差异 + 双向闪帧检测 + 直方图分布突变，Z-score异常筛选
    """

    def __init__(
        self,
        flash_sensitivity: float = 2.5,
        anomaly_sensitivity: float = 2.0,
        min_gap_sec: float = 0.03,
        max_events: int = 30,
    ):
        self.flash_sensitivity = flash_sensitivity
        self.anomaly_sensitivity = anomaly_sensitivity
        self.min_gap_sec = min_gap_sec
        self.max_events = max_events

    def _compute_temporal_signals(
        self, grays: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算三组时序信号：
          1. diff_signal:    与前帧的像素级差异
          2. flash_signal:   双向突变度（闪帧核心指标）
          3. feature_signal: 直方图分布突变（巴氏距离）
        """
        n = len(grays)
        small_size = (64, 64)

        smalls = [cv2.resize(g, small_size).astype(np.float32) for g in grays]

        hists = []
        for g in grays:
            h = cv2.calcHist([g], [0], None, [64], [0, 256]).flatten()
            h = h / (h.sum() + 1e-10)
            hists.append(h)

        diff_signal = np.zeros(n)
        flash_signal = np.zeros(n)
        feature_signal = np.zeros(n)

        for i in range(n):
            if i > 0:
                diff_signal[i] = float(np.mean(np.abs(smalls[i] - smalls[i - 1])))

            if 0 < i < n - 1:
                d_prev = float(np.mean(np.abs(smalls[i] - smalls[i - 1])))
                d_next = float(np.mean(np.abs(smalls[i] - smalls[i + 1])))
                d_surround = float(np.mean(np.abs(smalls[i - 1] - smalls[i + 1])))
                flash_signal[i] = max(0.0, (d_prev + d_next) / 2.0 - d_surround)

            if i > 0:
                hist_dist = cv2.compareHist(
                    hists[i].astype(np.float32),
                    hists[i - 1].astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                feature_signal[i] = hist_dist

        return diff_signal, flash_signal, feature_signal

    def detect(
        self, grays: List[np.ndarray], indices: List[int], fps: float
    ) -> Tuple[List[int], List[float]]:
        """
        检测高异常候选帧（不做最小间距稀疏化）
        Returns: (candidate_indices, candidate_scores) 按时间升序排列
        """
        n = len(grays)
        if n < 3:
            return [], []

        diff_signal, flash_signal, feature_signal = (
            self._compute_temporal_signals(grays)
        )

        # (1) 帧间差异异常
        diff_mean, diff_std = diff_signal.mean(), diff_signal.std()
        if diff_std > 1e-10:
            diff_zscore = (diff_signal - diff_mean) / diff_std
            diff_anomaly = np.maximum(diff_zscore - self.anomaly_sensitivity, 0)
        else:
            diff_anomaly = np.zeros(n)

        # (2) 闪帧异常（Z-score + 峰值检测）
        flash_anomaly = np.zeros(n)
        flash_mean, flash_std = flash_signal.mean(), flash_signal.std()
        if flash_std > 1e-10:
            flash_threshold = flash_mean + self.flash_sensitivity * flash_std
            peaks, props = find_peaks(
                flash_signal, height=flash_threshold, distance=1
            )
            if len(peaks) > 0:
                ph = props["peak_heights"]
                ph_max = ph.max()
                if ph_max > 1e-10:
                    for pi, pk in enumerate(peaks):
                        flash_anomaly[pk] = ph[pi] / ph_max

        # (3) 直方图分布突变
        feat_mean, feat_std = feature_signal.mean(), feature_signal.std()
        if feat_std > 1e-10:
            feat_zscore = (feature_signal - feat_mean) / feat_std
            feat_anomaly = np.maximum(feat_zscore - self.anomaly_sensitivity, 0)
        else:
            feat_anomaly = np.zeros(n)

        # 融合
        anomaly_scores = (
            diff_anomaly * 0.30
            + flash_anomaly * 0.45
            + feat_anomaly * 0.25
        )

        # 筛选
        if anomaly_scores.max() < 1e-10:
            return [], []

        anomaly_norm = anomaly_scores / anomaly_scores.max()

        candidates = [
            (indices[i], float(anomaly_norm[i]))
            for i in range(n) if anomaly_norm[i] > 0
        ]
        candidates.sort(key=lambda x: x[0])

        # 兼容保留参数：min_gap_sec / max_events 不再用于最终候选筛选
        candidate_indices = [x[0] for x in candidates]
        candidate_scores = [x[1] for x in candidates]
        return candidate_indices, candidate_scores


# ===================== 主流水线 =====================
class VideoFramePipeline:
    """
    视频帧采样流水线
    X(保底) + a(事件) → X'(合并)

    支持两种保底采样模式：
      low  — 固定帧数，适合下游模型上下文窗口有限的场景
      high — 按FPS采样，适合算力充足的场景
    """

    def __init__(
        self,
        # ===== 保底帧采样配置 =====
        baseline_mode: str = "low",
        baseline_fixed_frames: int = 40,   # low模式：固定采样帧数
        baseline_fps: float = 2.0,         # high模式：每秒采样帧数
        # ===== 事件检测配置 =====
        event_sample_fps: int = 15,
        flash_sensitivity: float = 2.2,
        anomaly_sensitivity: float = 1.6,
        event_min_gap_sec: float = 0.03,
        max_events: int = 30,
        window_merge_gap_sec: float = 0.5,
        window_pad_sec: float = 0.1,
        window_fuse_gap_sec: float = 0.3,
        # ===== 合并配置 =====
        merge_dedup_frames: int = 2,
        # ===== 过滤配置 =====
        thresh_black_screen: float = 10.0,
        thresh_blur: float = 15.0,
        # ===== 相似帧过滤 =====
        sim_filter_enabled: bool = True,
        sim_filter_threshold: float = 0.97,  # 相似度阈值，越高越严格（保留更多帧）
    ):
        """
        Args:
            baseline_mode: "low"(固定帧数) 或 "high"(按FPS)
            baseline_fixed_frames: low模式下的固定采样帧数
            baseline_fps: high模式下的每秒采样帧数
            event_sample_fps: 事件检测的密集采样帧率
            flash_sensitivity: 闪帧检测灵敏度
            anomaly_sensitivity: 一般异常检测灵敏度
            event_min_gap_sec: 兼容参数，默认不用于最终候选筛选
            max_events: 兼容参数，默认不用于最终候选筛选
            window_merge_gap_sec: 候选帧成窗的最大间隔（秒）
            window_pad_sec: 窗口边界前后扩展（秒）
            window_fuse_gap_sec: 二次并窗阈值（基于窗口边界间隔，秒）
            merge_dedup_frames: 合并去重距离（帧索引差<=此值视为重复）
            thresh_black_screen: 黑屏检测阈值
            thresh_blur: 模糊检测阈值
            sim_filter_enabled: 是否启用相似帧过滤
            sim_filter_threshold: 相似度阈值 0~1，超过此值视为重复（建议 0.90~0.95）
        """
        if baseline_mode not in ("low", "high"):
            raise ValueError(
                f"baseline_mode 必须是 'low' 或 'high'，收到: {baseline_mode}"
            )

        self.baseline_mode = baseline_mode
        self.baseline_fixed_frames = baseline_fixed_frames
        self.baseline_fps = baseline_fps
        self.event_sample_fps = event_sample_fps
        self.window_merge_gap_sec = window_merge_gap_sec
        self.window_pad_sec = window_pad_sec
        self.window_fuse_gap_sec = window_fuse_gap_sec
        self.merge_dedup_frames = merge_dedup_frames
        self.thresh_black_screen = thresh_black_screen
        self.thresh_blur = thresh_blur
        self.sim_filter_enabled = sim_filter_enabled
        self.sim_filter_threshold = sim_filter_threshold

        self.event_detector = EventFrameDetector(
            flash_sensitivity=flash_sensitivity,
            anomaly_sensitivity=anomaly_sensitivity,
            min_gap_sec=event_min_gap_sec,
            max_events=max_events,
        )

    # ==================== 视频读取 ====================

    def _open_video(self, video_path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        info = VideoInfo(
            path=video_path,
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if info.fps <= 0:
            info.fps = 25.0
        info.duration_sec = info.total_frames / info.fps
        return cap, info

    def _sequential_read_frames(
        self, cap: cv2.VideoCapture, target_indices: List[int], total_frames: int
    ) -> Dict[int, np.ndarray]:
        """顺序读取指定索引的帧（用grab跳帧，比随机seek快）"""
        frames = {}
        sorted_targets = sorted(set(target_indices))
        if not sorted_targets:
            return frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_pos = 0

        for target in sorted_targets:
            if target >= total_frames:
                break
            skip = target - current_pos
            if skip > 0:
                for _ in range(skip):
                    if not cap.grab():
                        break
                current_pos = target

            ret, frame = cap.read()
            current_pos += 1
            if ret:
                frames[target] = frame

        return frames

    # ==================== X: 保底帧采样 ====================

    def _sample_baseline(self, info: VideoInfo) -> List[int]:
        """
        根据模式生成保底帧索引

        low模式:  无论视频多长，均匀采样固定数量的帧
                  例: 30s视频30fps=900帧, 固定采40帧 → 每22帧取1帧
                  例: 2s视频30fps=60帧, 固定采40帧  → 采40帧（不足时取实际数量）

        high模式: 按FPS采样，帧数随时长线性增长
                  例: 30s视频, 2fps → 60帧
                  例: 120s视频, 2fps → 240帧
        """
        total = info.total_frames

        if self.baseline_mode == "low":
            n_frames = min(self.baseline_fixed_frames, total)
            if n_frames <= 0:
                return []
            # np.linspace 保证首尾帧都被采到，且分布均匀
            raw_indices = np.linspace(0, total - 1, n_frames, dtype=int)
            indices = list(dict.fromkeys(raw_indices.tolist()))  # 去重保序
            logger.info(
                f"保底帧(X) [low模式]: 目标{self.baseline_fixed_frames}帧 → "
                f"实际{len(indices)}帧 (视频共{total}帧)"
            )
        else:  # high
            interval = max(1, int(info.fps / self.baseline_fps))
            indices = list(range(0, total, interval))
            logger.info(
                f"保底帧(X) [high模式]: {self.baseline_fps}fps, "
                f"间隔={interval} → {len(indices)}帧"
            )

        return indices

    # ==================== a: 事件帧检测 ====================

    def _detect_events(
        self, cap: cv2.VideoCapture, info: VideoInfo,
    ) -> Tuple[List[int], List[float]]:
        """密集采样 → 过滤 → 高异常候选帧检测"""
        interval = max(1, int(info.fps / self.event_sample_fps))
        dense_indices = list(range(0, info.total_frames, interval))

        logger.info(
            f"事件检测: 密集采样{len(dense_indices)}帧 (间隔={interval})"
        )

        frame_dict = self._sequential_read_frames(
            cap, dense_indices, info.total_frames
        )

        valid_indices = []
        valid_grays = []

        for idx in dense_indices:
            if idx not in frame_dict:
                continue
            gray = cv2.cvtColor(frame_dict[idx], cv2.COLOR_BGR2GRAY)

            if np.mean(gray) < self.thresh_black_screen:
                continue
            if np.var(cv2.Laplacian(gray, cv2.CV_64F)) < self.thresh_blur:
                continue

            valid_indices.append(idx)
            valid_grays.append(gray)

        logger.info(f"事件检测有效帧: {len(valid_indices)} / {len(dense_indices)}")

        if len(valid_indices) < 3:
            return [], []

        candidate_indices, candidate_scores = self.event_detector.detect(
            valid_grays, valid_indices, info.fps
        )
        logger.info(f"检测到高异常候选帧: {len(candidate_indices)} 个")
        return candidate_indices, candidate_scores

    def _build_event_windows(
        self,
        candidate_indices: List[int],
        candidate_scores: List[float],
        fps: float,
        total_frames: int,
    ) -> List[EventWindow]:
        """
        将按时间排序的高异常候选帧聚合为事件时间窗。
        相邻候选帧间隔 <= window_merge_gap_sec 归入同一窗口。
        """
        if not candidate_indices:
            return []

        merge_gap_frames = max(1, int(round(self.window_merge_gap_sec * fps)))
        pad_frames = max(0, int(round(self.window_pad_sec * fps)))
        max_frame_idx = max(0, total_frames - 1)

        grouped: List[List[Tuple[int, float]]] = []
        cur_group: List[Tuple[int, float]] = [
            (candidate_indices[0], candidate_scores[0])
        ]

        for idx, score in zip(candidate_indices[1:], candidate_scores[1:]):
            if idx - cur_group[-1][0] <= merge_gap_frames:
                cur_group.append((idx, score))
            else:
                grouped.append(cur_group)
                cur_group = [(idx, score)]
        grouped.append(cur_group)

        windows: List[EventWindow] = []
        for wid, group in enumerate(grouped):
            dense_indices = [int(i) for i, _ in group]
            dense_scores = [float(s) for _, s in group]

            start_raw = min(dense_indices)
            end_raw = max(dense_indices)
            start_frame = max(0, start_raw - pad_frames)
            end_frame = min(max_frame_idx, end_raw + pad_frames)

            peak_local_idx = int(np.argmax(np.asarray(dense_scores)))
            peak_frame = dense_indices[peak_local_idx]
            peak_score = dense_scores[peak_local_idx]

            windows.append(EventWindow(
                window_id=wid,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                peak_frame=peak_frame,
                peak_score=peak_score,
                dense_indices=dense_indices,
                dense_scores=dense_scores,
            ))

        return windows

    def _fuse_event_windows(
        self,
        windows: List[EventWindow],
        fps: float,
    ) -> List[EventWindow]:
        """
        二次并窗：若相邻窗口边界间隔很小，则合并为同一事件窗。
        条件：next.start_time - cur.end_time <= window_fuse_gap_sec
        """
        if len(windows) <= 1:
            return windows

        if fps <= 0:
            fps = 25.0
        merged_groups: List[List[EventWindow]] = []
        cur_group: List[EventWindow] = [windows[0]]

        for w in windows[1:]:
            prev = cur_group[-1]
            # 直接按窗口时间边界判定，和 event_windows.txt 的时间语义一致
            gap_sec = w.start_time - prev.end_time
            if gap_sec <= self.window_fuse_gap_sec:
                cur_group.append(w)
            else:
                merged_groups.append(cur_group)
                cur_group = [w]
        merged_groups.append(cur_group)

        fused: List[EventWindow] = []
        for wid, group in enumerate(merged_groups):
            start_frame = min(w.start_frame for w in group)
            end_frame = max(w.end_frame for w in group)

            dense_indices: List[int] = []
            dense_scores: List[float] = []
            for w in group:
                dense_indices.extend(w.dense_indices)
                dense_scores.extend(w.dense_scores)

            pairs = sorted(zip(dense_indices, dense_scores), key=lambda x: x[0])
            dense_indices = [int(p[0]) for p in pairs]
            dense_scores = [float(p[1]) for p in pairs]

            peak_local_idx = int(np.argmax(np.asarray(dense_scores)))
            peak_frame = dense_indices[peak_local_idx]
            peak_score = dense_scores[peak_local_idx]

            fused.append(EventWindow(
                window_id=wid,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                peak_frame=peak_frame,
                peak_score=peak_score,
                dense_indices=dense_indices,
                dense_scores=dense_scores,
            ))

        return fused

    # ==================== 合并 X + a → X' ====================

    def _merge(
        self,
        baseline_indices: List[int],
        event_indices: List[int],
        event_scores: List[float],
    ) -> Tuple[List[int], List[str], List[float]]:
        """
        事件帧按时序插入保底帧，去重合并
        去重规则：事件帧与保底帧索引差 <= merge_dedup_frames 时，
                 用事件帧替换保底帧（保留更有价值的标记）
        """
        event_score_map = dict(zip(event_indices, event_scores))

        merged = {}  # index -> (source_tag, score)

        for idx in baseline_indices:
            merged[idx] = ("baseline", 0.0)

        for idx in event_indices:
            is_dup = False
            for existing_idx in list(merged.keys()):
                if abs(idx - existing_idx) <= self.merge_dedup_frames:
                    if merged[existing_idx][0] == "baseline":
                        del merged[existing_idx]
                        merged[idx] = ("event", event_score_map[idx])
                    is_dup = True
                    break
            if not is_dup:
                merged[idx] = ("event", event_score_map[idx])

        sorted_items = sorted(merged.items(), key=lambda x: x[0])
        m_indices = [item[0] for item in sorted_items]
        m_tags = [item[1][0] for item in sorted_items]
        m_scores = [item[1][1] for item in sorted_items]

        return m_indices, m_tags, m_scores

    # ==================== 相似帧过滤 ====================

    def _filter_similar(
        self,
        frames: List[np.ndarray],
        indices: List[int],
        tags: List[str],
        scores: List[float],
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[float]]:
        """
        过滤 X' 中与前一保留帧高度相似的冗余帧。
        适用场景：稳定运动画面、静止背景段落等相邻帧几乎无变化的情况。

        优先级：event 帧 > baseline 帧。
        当当前帧(event)与已保留帧(baseline)相似时，替换掉已保留帧。
        其余情况丢弃当前帧。
        """
        if len(frames) <= 1:
            return frames, indices, tags, scores

        small_size = (64, 64)
        smalls = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            smalls.append(cv2.resize(gray, small_size).astype(np.float32))

        keep = [True] * len(frames)
        last_kept = 0

        for i in range(1, len(frames)):
            diff = np.mean(np.abs(smalls[i] - smalls[last_kept])) / 255.0
            similarity = 1.0 - diff

            if similarity >= self.sim_filter_threshold:
                # 相似：event 替换 baseline，否则丢弃当前帧
                if tags[i] == "event" and tags[last_kept] == "baseline":
                    keep[last_kept] = False
                    last_kept = i
                else:
                    keep[i] = False
            else:
                last_kept = i

        out_frames  = [f for f, k in zip(frames,  keep) if k]
        out_indices = [v for v, k in zip(indices, keep) if k]
        out_tags    = [v for v, k in zip(tags,    keep) if k]
        out_scores  = [v for v, k in zip(scores,  keep) if k]

        n_removed = keep.count(False)
        if n_removed > 0:
            logger.info(
                f"相似帧过滤: 移除 {n_removed} 帧 "
                f"(阈值={self.sim_filter_threshold:.2f}) "
                f"→ 剩余 {len(out_frames)} 帧"
            )
        return out_frames, out_indices, out_tags, out_scores

    # ==================== 核心接口 ====================

    def process(self, video_path: str) -> PipelineResult:
        """
        核心接口：处理视频

        Returns:
            PipelineResult 包含：
              - X'（合并帧流）: frames_X_prime, indices_X_prime, source_tags, merged_scores
              - a（兼容字段）: frames_a, indices_a, scores_a（高异常候选帧，按分数降序）
              - event_windows: 事件时间窗（核心输出）
              - X（保底帧索引）: indices_X
        """
        result = PipelineResult()
        result.baseline_mode = self.baseline_mode

        try:
            # 1. 打开视频
            cap, info = self._open_video(video_path)
            result.video_info = info
            logger.info(
                f"视频: {os.path.basename(video_path)} | "
                f"{info.total_frames}帧, {info.fps:.1f}fps, "
                f"{info.duration_sec:.1f}s, {info.width}x{info.height}"
            )

            # 2. 保底帧 X
            baseline_indices = self._sample_baseline(info)
            result.indices_X = baseline_indices
            result.baseline_frame_count = len(baseline_indices)

            # 3. 候选异常帧 -> 事件时间窗
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            candidate_indices, candidate_scores = self._detect_events(cap, info)
            raw_windows = self._build_event_windows(
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                fps=info.fps,
                total_frames=info.total_frames,
            )
            result.event_windows = self._fuse_event_windows(raw_windows, info.fps)
            if len(raw_windows) != len(result.event_windows):
                logger.info(
                    f"二次并窗: {len(raw_windows)} -> {len(result.event_windows)} "
                    f"(fuse_gap={self.window_fuse_gap_sec:.2f}s)"
                )
            result.window_count = len(result.event_windows)
            result.window_peak_indices = [w.peak_frame for w in result.event_windows]
            result.window_dense_index_dict = {
                w.window_id: list(w.dense_indices) for w in result.event_windows
            }

            # merge 输入：每个窗口取峰值帧作为代表
            event_indices = [w.peak_frame for w in result.event_windows]
            event_scores = [w.peak_score for w in result.event_windows]
            # 兼容旧输出 a：保留全部高异常候选帧
            result.event_count = len(candidate_indices)

            # 4. 合并 → X'
            merged_indices, source_tags, merged_scores = self._merge(
                baseline_indices, event_indices, event_scores
            )

            # 5. 一次性读取所有需要的帧（X' ∪ a候选 的原始索引）
            all_needed = set(merged_indices) | set(candidate_indices)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_dict = self._sequential_read_frames(
                cap, sorted(all_needed), info.total_frames
            )
            cap.release()

            # 6. 组装 X'（按时序）
            for idx, tag, score in zip(merged_indices, source_tags, merged_scores):
                if idx in frame_dict:
                    frame_rgb = cv2.cvtColor(frame_dict[idx], cv2.COLOR_BGR2RGB)
                    result.frames_X_prime.append(frame_rgb)
                    result.indices_X_prime.append(idx)
                    result.source_tags.append(tag)
                    result.merged_scores.append(score)

            result.total_output_frames = len(result.frames_X_prime)

            # 6b. 相似帧过滤
            if self.sim_filter_enabled and result.frames_X_prime:
                (result.frames_X_prime, result.indices_X_prime,
                 result.source_tags, result.merged_scores) = self._filter_similar(
                    result.frames_X_prime, result.indices_X_prime,
                    result.source_tags, result.merged_scores,
                )
                result.total_output_frames = len(result.frames_X_prime)

            # 7. 组装独立事件帧 a（候选帧，按分数降序）
            paired = sorted(
                zip(candidate_indices, candidate_scores),
                key=lambda x: x[1], reverse=True
            )
            for idx, score in paired:
                if idx in frame_dict:
                    frame_rgb = cv2.cvtColor(frame_dict[idx], cv2.COLOR_BGR2RGB)
                    result.frames_a.append(frame_rgb)
                    result.indices_a.append(idx)
                    result.scores_a.append(score)

            logger.info(
                f"流水线完成: X={len(baseline_indices)}, "
                f"windows={result.window_count}, "
                f"a(candidates)={len(candidate_indices)}, "
                f"merge_peaks={len(event_indices)}, "
                f"X'={result.total_output_frames} | "
                f"模式={self.baseline_mode}"
            )

        except Exception as e:
            logger.error(f"流水线处理失败: {e}", exc_info=True)

        return result


# ===================== 工具函数 =====================

def save_pipeline_result(
    video_path: str,
    result: PipelineResult,
    output_dir: str,
    draw_info: bool = True,
    save_events_separately: bool = True,
    save_events_by_window: bool = True,
):
    """
    保存结果帧到文件夹

    目录结构：
        output_dir/
            X_prime/          ← 合并帧流（按时序编号）
            events/           ← 独立事件帧（按分数排序，可选）
            events_by_window/ ← 每个事件窗的候选帧集合（可选）
    """
    fps = result.video_info.fps if result.video_info.fps > 0 else 25.0

    # ---- 保存 X' ----
    xp_dir = os.path.join(output_dir, "X_prime")
    os.makedirs(xp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    if not result.indices_X_prime:
        logger.info("X' 没有帧需要保存")
    else:
        saved = 0
        for i, (idx, tag, score) in enumerate(
            zip(result.indices_X_prime, result.source_tags, result.merged_scores)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if draw_info:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                color = (0, 0, 180) if tag == "event" else (80, 80, 80)
                cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                time_sec = idx / fps
                text = f"[{i:03d}] F#{idx} | {tag.upper()}"
                if tag == "event":
                    text += f" | S={score:.3f}"
                text += f" | T={time_sec:.2f}s"
                cv2.putText(
                    frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )

            prefix = f"EVENT_{score:.3f}" if tag == "event" else "BASE"
            path = os.path.join(xp_dir, f"{i:03d}_{prefix}_f{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

        logger.info(f"X' 已保存 {saved} 帧到 {xp_dir}")

    # ---- 保存独立事件帧（兼容：高异常候选帧） ----
    if save_events_separately and result.indices_a:
        ev_dir = os.path.join(output_dir, "events")
        os.makedirs(ev_dir, exist_ok=True)

        ev_saved = 0
        for rank, (idx, score) in enumerate(
            zip(result.indices_a, result.scores_a)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if draw_info:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                time_sec = idx / fps
                text = (
                    f"EVENT #{rank+1} | F#{idx} | "
                    f"Score={score:.3f} | T={time_sec:.2f}s"
                )
                cv2.putText(
                    frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1
                )

            path = os.path.join(
                ev_dir, f"{rank:02d}_score{score:.3f}_f{idx:06d}.jpg"
            )
            cv2.imwrite(path, frame)
            ev_saved += 1

        logger.info(f"高异常候选帧已保存 {ev_saved} 帧到 {ev_dir}")

    # ---- 按事件窗保存候选帧 ----
    if save_events_by_window and result.event_windows:
        wb_dir = os.path.join(output_dir, "events_by_window")
        os.makedirs(wb_dir, exist_ok=True)

        for w in result.event_windows:
            sub = (
                f"W{w.window_id:02d}_"
                f"f{w.start_frame:06d}-{w.end_frame:06d}_"
                f"t{w.start_time:.2f}-{w.end_time:.2f}s"
            )
            win_dir = os.path.join(wb_dir, sub)
            os.makedirs(win_dir, exist_ok=True)

            saved_in_win = 0
            dense_pairs = list(zip(w.dense_indices, w.dense_scores))
            dense_pairs.sort(key=lambda x: x[0])
            for j, (idx, score) in enumerate(dense_pairs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                if draw_info:
                    h, w0 = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w0, 54), (0, 60, 200), -1)
                    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                    time_sec = idx / fps
                    line1 = (
                        f"W{w.window_id:02d} | F#{idx} | T={time_sec:.2f}s"
                    )
                    line2 = (
                        f"score={score:.3f} | peak=F{w.peak_frame}"
                    )
                    cv2.putText(
                        frame, line1, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                    )
                    cv2.putText(
                        frame, line2, (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                    )

                mark = "PEAK" if idx == w.peak_frame else "DENSE"
                path = os.path.join(
                    win_dir,
                    f"{j:03d}_{mark}_score{score:.3f}_f{idx:06d}.jpg"
                )
                cv2.imwrite(path, frame)
                saved_in_win += 1

            logger.info(
                f"事件窗 W{w.window_id:02d} 候选帧已保存 {saved_in_win} 帧到 {win_dir}"
            )

    # ---- 保存事件时间窗摘要 ----
    if result.event_windows:
        windows_txt = os.path.join(output_dir, "event_windows.txt")
        with open(windows_txt, "w", encoding="utf-8") as f:
            f.write("Event Windows\n")
            f.write("=" * 60 + "\n")
            for w in result.event_windows:
                line = (
                    f"W{w.window_id:02d} | "
                    f"frame=[{w.start_frame}, {w.end_frame}] | "
                    f"time=[{w.start_time:.3f}, {w.end_time:.3f}]s | "
                    f"peak=F{w.peak_frame} ({w.peak_score:.3f}) | "
                    f"dense={len(w.dense_indices)}\n"
                )
                f.write(line)
                logger.info(
                    f"事件窗 W{w.window_id:02d}: "
                    f"F[{w.start_frame}-{w.end_frame}] "
                    f"T[{w.start_time:.2f}-{w.end_time:.2f}]s "
                    f"peak=F{w.peak_frame}({w.peak_score:.3f}) "
                    f"dense={len(w.dense_indices)}"
                )
        logger.info(f"事件时间窗摘要已保存到 {windows_txt}")

    cap.release()


# ===================== 测试入口 =====================
if __name__ == "__main__":
    import sys

    video_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "/sda/yuqifan/HFOCUS/test/1.mp4"
    )

    # 命令行第二个参数选模式: python xxx.py video.mp4 low/high
    mode = sys.argv[2] if len(sys.argv) > 2 else "low"

    print(f"\n{'='*60}")
    print(f"模式: {mode}")
    print(f"{'='*60}\n")

    pipeline = VideoFramePipeline(
        baseline_mode=mode,
        baseline_fixed_frames=40,    # low模式生效
        baseline_fps=2.0,            # high模式生效
        event_sample_fps=15,
        flash_sensitivity=2.2,
        anomaly_sensitivity=1.6,
        event_min_gap_sec=0.33,
        max_events=30,
        window_merge_gap_sec=0.5,
        window_pad_sec=0.3,
        window_fuse_gap_sec=0.3,
        merge_dedup_frames=2,
    )

    result = pipeline.process(video_path)

    # 摘要
    print(f"\n{'='*60}")
    print(result.summary())
    print(f"{'='*60}")

    # 事件时间窗详情（核心输出）
    if result.event_windows:
        print(f"\n事件时间窗 ({result.window_count} 个):")
        print("-" * 75)
        for w in result.event_windows:
            print(
                f"  W{w.window_id:02d} | "
                f"start/end frame=({w.start_frame}, {w.end_frame}) | "
                f"start/end time=({w.start_time:.2f}s, {w.end_time:.2f}s) | "
                f"peak frame={w.peak_frame} | peak score={w.peak_score:.3f} | "
                f"dense count={len(w.dense_indices)}"
            )

    # X' 详情
    if result.indices_X_prime:
        print(f"\nX' 帧流 ({result.total_output_frames} 帧):")
        print("-" * 55)
        bc = sum(1 for t in result.source_tags if t == "baseline")
        ec = sum(1 for t in result.source_tags if t == "event")
        print(f"  构成: {bc} 保底 + {ec} 事件\n")

        fps = result.video_info.fps if result.video_info.fps > 0 else 25.0
        for i, (idx, tag, score) in enumerate(
            zip(result.indices_X_prime, result.source_tags, result.merged_scores)
        ):
            t = idx / fps
            marker = "⚡" if tag == "event" else "  "
            sc = f"score={score:.3f}" if tag == "event" else ""
            print(
                f"  {marker} [{i:03d}] frame={idx:6d}  "
                f"t={t:6.2f}s  {tag:8s}  {sc}"
            )

    # 独立事件帧（兼容字段：高异常候选帧）
    if result.indices_a:
        fps = result.video_info.fps if result.video_info.fps > 0 else 25.0
        print(f"\n高异常候选帧 a ({result.event_count} 帧, 按分数降序):")
        print("-" * 55)
        for rank, (idx, score) in enumerate(
            zip(result.indices_a, result.scores_a)
        ):
            t = idx / fps
            bar = "█" * int(score * 20)
            print(
                f"  #{rank+1:2d}  frame={idx:6d}  t={t:6.2f}s  "
                f"score={score:.3f}  {bar}"
            )

    # 保存
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join("output_pipeline", f"{video_name}_{mode}")
    save_pipeline_result(
        video_path, result, output_folder,
        draw_info=True,
        save_events_separately=True,
        save_events_by_window=True,
    )
