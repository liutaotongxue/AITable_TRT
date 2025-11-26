"""
情绪识别引擎
===========

封装情绪识别推理逻辑，包含降频、缓存和延迟统计。
"""
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from ..compat import np
from ..core.logger import logger
from ..emotion import EmoNetClassifier


@dataclass
class EmotionResult:
    """情绪识别结果"""
    data: Optional[Dict[str, Any]]  # 情绪数据（emotion, valence, arousal）
    latency_ms: float               # 推理延迟（毫秒，EMA 平滑）
    speed_fps: float                # 理论 FPS（基于延迟计算）
    source: str                     # 数据来源："inference", "cache", "disabled", "error"
    timestamp: float                # 结果时间戳


class EmotionEngine:
    """
    情绪识别引擎

    职责:
    - 管理情绪识别推理降频（每 N 帧执行一次）
    - 缓存结果和超时检测
    - 延迟统计（EMA 平滑）
    - 异常处理

    使用示例:
        engine = EmotionEngine(
            classifier=emotion_classifier,
            interval_frames=10,
            cache_timeout=4.0
        )

        # 主循环中调用
        result = engine.maybe_infer(
            rgb_frame=frame,
            face_bbox=bbox,
            face_present=True,
            face_just_appeared=False
        )

        if result.data:
            print(f"Emotion: {result.data['emotion']}")
    """

    def __init__(
        self,
        classifier: Optional[EmoNetClassifier],
        interval_frames: int = 10,
        cache_timeout: float = 4.0,
        latency_smoothing: float = 0.7
    ):
        """
        初始化情绪识别引擎

        Args:
            classifier: EmoNetClassifier 实例（None 时禁用）
            interval_frames: 推理间隔（帧数）
            cache_timeout: 缓存超时时间（秒）
            latency_smoothing: 延迟 EMA 平滑系数（0-1，越大越平滑）
        """
        self.classifier = classifier
        self.interval_frames = max(1, interval_frames)
        self.cache_timeout = cache_timeout
        self.latency_smoothing = latency_smoothing

        # 内部状态
        self._frame_count = 0
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_timestamp = time.time()
        self._latency_ms = 0.0

        if self.classifier is None:
            logger.info("EmotionEngine: 已禁用（无 classifier）")
        else:
            logger.info(f"EmotionEngine: 已启用（间隔={interval_frames}帧，缓存超时={cache_timeout}s）")

    def should_infer(
        self,
        face_present: bool,
        face_just_appeared: bool = False
    ) -> tuple[bool, str]:
        """
        判断是否需要执行推理（不改变状态）

        Args:
            face_present: 当前帧是否有人脸
            face_just_appeared: 人脸是否刚出现

        Returns:
            tuple[bool, str]: (是否需要推理, 原因说明)
        """
        # 引擎禁用
        if self.classifier is None:
            return (False, "engine_disabled")

        # 无人脸
        if not face_present:
            return (False, "no_face")

        # 判断是否需要推理
        cache_expired = (time.time() - self._last_timestamp) > self.cache_timeout

        if self._last_result is None:
            return (True, "first_inference")
        elif cache_expired:
            return (True, "cache_expired")
        elif face_just_appeared:
            return (True, "face_appeared")
        elif (self._frame_count + 1) % self.interval_frames == 0:
            return (True, "interval_reached")
        else:
            return (False, "use_cache")

    def maybe_infer(
        self,
        rgb_frame: np.ndarray,
        face_bbox: Optional[Dict[str, int]],
        face_present: bool,
        face_just_appeared: bool = False,
        face_roi: Optional[Any] = None
    ) -> EmotionResult:
        """
        根据降频策略决定是否执行推理

        Args:
            rgb_frame: RGB 图像帧
            face_bbox: 人脸边界框 {'x1', 'y1', 'x2', 'y2'}
            face_present: 当前帧是否有人脸
            face_just_appeared: 人脸是否刚出现（无人脸->有人脸切换）
            face_roi: 可选的预提取 RegionROI 对象（如果提供，直接使用而不重新裁剪）

        Returns:
            EmotionResult: 情绪识别结果
        """
        # 引擎禁用
        if self.classifier is None:
            return EmotionResult(
                data=None,
                latency_ms=0.0,
                speed_fps=0.0,
                source="disabled",
                timestamp=time.time()
            )

        # 无人脸
        if not face_present or face_bbox is None:
            return EmotionResult(
                data=self._last_result,
                latency_ms=self._latency_ms,
                speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                source="cache",
                timestamp=self._last_timestamp
            )

        # 累加帧计数
        self._frame_count += 1

        # 判断是否需要推理
        cache_expired = (time.time() - self._last_timestamp) > self.cache_timeout

        should_infer = (
            (self._frame_count % self.interval_frames == 0) or  # 定期推理
            (self._last_result is None) or                      # 首次推理
            cache_expired or                                    # 缓存超时
            face_just_appeared                                  # 人脸刚出现
        )

        # 执行推理
        if should_infer:
            # 日志提示
            if cache_expired and self._last_result is not None:
                logger.info("[Emotion] 缓存超时，强制刷新情绪识别")
            elif face_just_appeared:
                logger.info("[Emotion] 检测到新人脸，刷新情绪识别")

            try:
                # 获取人脸图像（优先使用预提取的 ROI）
                if face_roi is not None:
                    # 使用预提取的 ROI（避免重复裁剪）
                    face_img = face_roi.image
                else:
                    # 回退到手动裁剪（兼容旧代码）
                    x1, y1 = max(0, face_bbox['x1']), max(0, face_bbox['y1'])
                    x2, y2 = min(rgb_frame.shape[1], face_bbox['x2']), min(rgb_frame.shape[0], face_bbox['y2'])

                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"[Emotion] 无效的人脸区域: ({x1},{y1})-({x2},{y2})")
                        return self._get_cached_result()

                    face_img = rgb_frame[y1:y2, x1:x2]

                # 推理计时
                infer_start = time.time()
                batch_results = self.classifier.predict_batch([face_img])
                infer_end = time.time()

                # 推理成功
                if batch_results:
                    emotion_data = batch_results[0]

                    # 更新缓存
                    self._last_result = emotion_data
                    self._last_timestamp = time.time()

                    # 更新延迟统计（EMA 平滑）
                    instant_latency_ms = (infer_end - infer_start) * 1000.0
                    self._latency_ms = (
                        self.latency_smoothing * self._latency_ms +
                        (1 - self.latency_smoothing) * instant_latency_ms
                    )

                    return EmotionResult(
                        data=emotion_data,
                        latency_ms=self._latency_ms,
                        speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                        source="inference",
                        timestamp=self._last_timestamp
                    )
                else:
                    logger.warning("[Emotion] 推理返回空结果")
                    return self._get_cached_result()

            except Exception as e:
                logger.error(f"[Emotion] 推理失败: {e}")
                return EmotionResult(
                    data=self._last_result,
                    latency_ms=self._latency_ms,
                    speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                    source="error",
                    timestamp=self._last_timestamp
                )

        # 使用缓存
        return self._get_cached_result()

    def _get_cached_result(self) -> EmotionResult:
        """返回缓存的结果"""
        return EmotionResult(
            data=self._last_result,
            latency_ms=self._latency_ms,
            speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
            source="cache",
            timestamp=self._last_timestamp
        )

    def reset(self):
        """重置引擎状态"""
        self._frame_count = 0
        self._last_result = None
        self._last_timestamp = time.time()
        self._latency_ms = 0.0
        logger.info("[Emotion] 引擎已重置")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            统计字典（延迟、FPS、缓存状态等）
        """
        return {
            "enabled": self.classifier is not None,
            "frame_count": self._frame_count,
            "latency_ms": self._latency_ms,
            "speed_fps": 1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
            "last_inference_time": self._last_timestamp,
            "cache_valid": self._last_result is not None,
            "cache_age_seconds": time.time() - self._last_timestamp if self._last_result else None
        }
