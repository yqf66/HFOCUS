"""
短视频高风险帧筛选模块（即插即用）
核心功能：基于无监督视觉特征（信息量+变化度+稀有度），对短视频密集采样并筛选高风险帧，
         交给后续有害内容判断模块，无需标注/训练，纯轻量级计算。
使用方式：
    from risk_frame_selector import RiskFrameSelector
    
    selector = RiskFrameSelector(sample_fps=10, top_k=5)
    high_risk_frames = selector.select(video_path="your_video.mp4")
"""

import numpy as np
import cv2
from decord import VideoReader, cpu
from typing import List, Optional
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings("ignore")  # 屏蔽sklearn聚类警告


class RiskFrameSelector:
    """短视频高风险帧筛选器（即插即用）"""
    
    def __init__(
        self,
        sample_fps: int = 10,
        top_k: int = 5,
        min_gap_sec: float = 0.1,
        weight_entropy: float = 0.3,
        weight_diff: float = 0.5,
        weight_rarity: float = 0.2,
        # OPTIMIZE: 将过滤阈值参数化，增加灵活性
        thresh_black_screen: float = 10.0,
        thresh_blur: float = 50.0,
        thresh_static: float = 5.0
    ):
        """
        初始化筛选器
        Args:
            sample_fps: 1秒内采样的帧数（推荐10，兼顾速度和精度）
            top_k: 最终输出的高风险帧数（根据后续模块算力调整）
            min_gap_sec: 避免选中的帧扎堆（比如0.1秒=100ms）
            weight_entropy/weight_diff/weight_rarity: 特征融合权重
            thresh_black_screen: 黑屏检测的灰度均值阈值
            thresh_blur: 模糊检测的拉普拉斯方差阈值
            thresh_static: 静态帧检测的帧间差异阈值
        """
        self.sample_fps = sample_fps
        self.top_k = top_k
        self.min_gap_sec = min_gap_sec
        self.weight_entropy = weight_entropy
        self.weight_diff = weight_diff
        self.weight_rarity = weight_rarity
        # OPTIMIZE: 保存过滤阈值
        self.thresh_black_screen = thresh_black_screen
        self.thresh_blur = thresh_blur
        self.thresh_static = thresh_static
        
    # ===================== 核心工具函数 =====================
    def _calculate_frame_entropy(self, frame: np.ndarray) -> float:
        """计算单帧图像熵（信息量）：熵越高，信息量越大"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in hist_norm if p > 0])
        return float(entropy)
    
    def _calculate_frame_gradient(self, frame: np.ndarray) -> float:
        """计算单帧梯度幅值（边缘强度）：梯度越高，细节越丰富"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_smooth = gaussian_filter(gray, sigma=1)
        grad_x = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(grad_mag))
    
    def _calculate_frame_diff(self, frame: np.ndarray, ref_frame: np.ndarray) -> float:
        """计算帧间差异：差异越大，画面变化越剧烈"""
        # 降采样+转灰度，加速计算
        frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 64))
        ref_gray = cv2.resize(cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY), (64, 64))
        diff = cv2.absdiff(frame_gray, ref_gray)
        return float(np.mean(diff))
    
    def _filter_low_value_frames(self, video: VideoReader, indices: List[int]) -> List[int]:
        """过滤低价值帧：黑屏、全模糊、静态帧"""
        valid_indices = []
        for i, idx in enumerate(indices):
            frame = video[idx].asnumpy()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. 黑屏检测
            if np.mean(gray) < self.thresh_black_screen:
                continue
            
            # 2. 模糊检测
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            if np.var(laplacian) < self.thresh_blur:
                continue
            
            # 3. 静态帧检测
            # OPTIMIZE: 避免重复读取前一帧，如果前一帧在采样列表中，则直接复用
            if i > 0:
                prev_idx = indices[i-1]
                # 只有当两帧在视频中是连续的时候才进行比较
                if idx == prev_idx + 1:
                    prev_frame = video[prev_idx].asnumpy()
                    diff = self._calculate_frame_diff(frame, prev_frame)
                    if diff < self.thresh_static:
                        continue
            
            valid_indices.append(idx)
        
        # 兜底：如果过滤后无有效帧，返回原始采样帧
        return valid_indices if valid_indices else indices[:self.top_k * 2]
    
    def _calculate_risk_scores(self, frames: List[np.ndarray], indices: List[int]) -> List[float]:
        """计算帧的无监督风险分数（核心）"""
        # OPTIMIZE: 传入帧列表，避免重复读取
        
        # 步骤1：提取基础特征（熵+梯度）
        frame_features = []
        for frame in frames:
            entropy = self._calculate_frame_entropy(frame)
            gradient = self._calculate_frame_gradient(frame)
            frame_features.append([entropy, gradient])
        frame_features = np.array(frame_features)
        
        # 归一化特征
        scaler = MinMaxScaler()
        frame_features_norm = scaler.fit_transform(frame_features)
        # 信息量分数（熵+梯度融合）
        info_scores = frame_features_norm[:, 0] * 0.5 + frame_features_norm[:, 1] * 0.5
        
        # 步骤2：计算帧变化度分数
        diff_scores = np.zeros(len(indices))
        if len(indices) >= 2:
            for i in range(len(frames)):
                # 取相邻帧作为参考
                ref_frame = frames[i-1] if i > 0 else frames[1]
                diff_scores[i] = self._calculate_frame_diff(frames[i], ref_frame)
            diff_scores = scaler.fit_transform(diff_scores.reshape(-1, 1)).flatten()
        
        # 步骤3：计算帧稀有度分数（聚类）
        rarity_scores = np.ones(len(indices)) * 0.5
        if len(indices) >= 5:  # 至少5帧才聚类
            # OPTIMIZE: 明确设置n_init以保证结果稳定并兼容新版sklearn
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(frame_features_norm)
            cluster_counts = np.bincount(clusters)
            for i, c in enumerate(clusters):
                rarity_scores[i] = 1.0 - (cluster_counts[c] / len(indices))
        
        # 步骤4：融合最终风险分数
        final_scores = (
            info_scores * self.weight_entropy +
            diff_scores * self.weight_diff +
            rarity_scores * self.weight_rarity
        )
        # 归一化到0~1
        final_scores = scaler.fit_transform(final_scores.reshape(-1, 1)).flatten()
        return [max(0.0, min(1.0, float(s))) for s in final_scores]
    
    def _select_topk_with_gap(self, indices: List[int], scores: List[float]) -> List[int]:
        """选TopK帧，且满足最小时间间隔"""
        # 按分数降序排序
        sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        selected = []
        fps = self._video_fps  # 缓存的视频帧率
        
        # 最小间隔（帧数）
        min_gap_frames = max(1, int(fps * self.min_gap_sec))
        
        # 第一轮：严格满足时间间隔
        for idx, score in sorted_pairs:
            if all(abs(idx - s) >= min_gap_frames for s in selected):
                selected.append(idx)
                if len(selected) == self.top_k:
                    break
        
        # 第二轮：若数量不够，放宽间隔补充
        if len(selected) < self.top_k:
            for idx, score in sorted_pairs:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) == self.top_k:
                        break
        
        return sorted(selected)
    
    # ===================== 对外核心接口 =====================
    def select(self, video_path: str) -> List[int]:
        """
        核心调用接口：筛选短视频的高风险帧。
        如果主方案 (Decord) 失败，会自动切换到备用方案 (OpenCV) 进行完整的风险分析。
        """
        try:
            # =================== 主方案: 使用Decord进行分析 ===================
            print("正在使用主方案 (Decord) 进行风险分析...")
            video = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(video)
            self._video_fps = video.get_avg_fps()
            
            sample_interval = max(1, int(self._video_fps / self.sample_fps))
            sampled_indices = list(range(0, total_frames, sample_interval))
            
            valid_indices = self._filter_low_value_frames(video, sampled_indices)
            valid_frames = [video[idx].asnumpy() for idx in valid_indices]
            
            risk_scores = self._calculate_risk_scores(valid_frames, valid_indices)
            high_risk_frames = self._select_topk_with_gap(valid_indices, risk_scores)
            print("Decord 分析成功。")
            return high_risk_frames
        
        except Exception as e:
            print(f"主方案 (Decord) 分析失败: {e}")
            print("切换到备用方案 (OpenCV) 进行完整风险分析...")
            try:
                # =================== 备用方案: 使用OpenCV进行完整分析 ===================
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"OpenCV 无法打开视频: {video_path}")
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                self._video_fps = fps

                sample_interval = max(1, int(fps / self.sample_fps))
                sampled_indices = list(range(0, total_frames, sample_interval))

                # 使用OpenCV重新实现过滤逻辑
                valid_indices = []
                temp_frames_for_filter = {}
                for i, idx in enumerate(sampled_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame_bgr = cap.read()
                    if not ret: continue
                    
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    if np.mean(gray) < self.thresh_black_screen: continue
                    if np.var(cv2.Laplacian(gray, cv2.CV_64F)) < self.thresh_blur: continue
                    
                    # 静态帧检测
                    if i > 0:
                        prev_idx = sampled_indices[i-1]
                        if idx == prev_idx + 1:
                            # 从缓存或文件中读取前一帧
                            if prev_idx in temp_frames_for_filter:
                                prev_frame_bgr = temp_frames_for_filter[prev_idx]
                            else:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
                                _, prev_frame_bgr = cap.read()
                            
                            if prev_frame_bgr is not None:
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                prev_frame_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
                                if self._calculate_frame_diff(frame_rgb, prev_frame_rgb) < self.thresh_static:
                                    continue
                    
                    valid_indices.append(idx)
                    temp_frames_for_filter[idx] = frame_bgr # 缓存已读取的帧
                
                # 从有效索引中读取所有帧 (BGR格式)
                valid_frames_bgr = []
                for idx in valid_indices:
                    if idx in temp_frames_for_filter:
                        valid_frames_bgr.append(temp_frames_for_filter[idx])
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame_bgr = cap.read()
                        if ret: valid_frames_bgr.append(frame_bgr)
                cap.release()

                # 将帧转为RGB以进行风险计算
                valid_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in valid_frames_bgr]

                risk_scores = self._calculate_risk_scores(valid_frames_rgb, valid_indices)
                high_risk_frames = self._select_topk_with_gap(valid_indices, risk_scores)
                print("OpenCV 备用分析成功。")
                return high_risk_frames

            except Exception as fallback_e:
                print(f"备用方案 (OpenCV) 分析也失败了: {fallback_e}")
                return [] # 彻底失败，返回空列表


# ===================== 测试用例（可直接运行） =====================
if __name__ == "__main__":
    # 优化：保存帧函数，统一使用更兼容的OpenCV进行读取和保存
    def save_frames(video_path: str, frame_indices: List[int], output_dir: str):
        """将指定索引的帧从视频中提取并保存到文件夹"""
        if not frame_indices:
            print("没有需要保存的帧，程序终止。")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录：{output_dir}")

        print(f"准备从 {video_path} 提取 {len(frame_indices)} 帧...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法使用OpenCV打开视频 {video_path} 进行最终保存。")
            return

        saved_count = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
            else:
                print(f"警告：无法读取索引为 {idx} 的帧。")
        cap.release()
        
        if saved_count > 0:
            print(f"成功保存 {saved_count} 帧到目录 {output_dir}")

    # 1. 初始化筛选器
    selector = RiskFrameSelector(
        sample_fps=10,
        top_k=5,
        min_gap_sec=0.1
    )
    
    # 2. 筛选高风险帧
    video_path = "/sda/yuqifan/HFOCUS/test/2.mp4"  # 替换为你的短视频路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join("output_frames", video_name)

    high_risk_frames = selector.select(video_path)
    
    # 3. 输出结果
    print(f"筛选出的高风险帧索引：{high_risk_frames}")

    # 4. 保存选中的帧到文件夹
    save_frames(video_path, high_risk_frames, output_folder)