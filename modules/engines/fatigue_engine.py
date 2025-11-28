"""
疲劳检测引擎
===========

封装疲劳检测推理逻辑（FaceMesh TensorRT），包含延迟统计。
"""
from __future__ import annotations  # Python 3.8 兼容

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from ..compat import np
from ..core.logger import logger


@dataclass
class FatigueResult:
    """疲劳检测结果"""
    data: Optional[Dict[str, Any]]  # 疲劳数据（EAR, MAR, 眨眼次数等）
    latency_ms: float               # 推理延迟（毫秒，EMA 平滑）
    speed_fps: float                # 理论 FPS（基于延迟计算）
    source: str                     # 数据来源："inference", "disabled", "error", "invalid_frame"
    timestamp: float                # 结果时间戳


class FatigueEngine:
    """
    疲劳检测引擎

    职责:
    - 管理疲劳检测推理（每帧执行，无降频）
    - 帧有效性验证（通过 validate_frame）
    - 延迟统计（EMA 平滑）
    - 异常处理

    使用示例:
        engine = FatigueEngine(
            detector=fatigue_detector,
            latency_smoothing=0.7
        )

        # 主循环中调用
        result = engine.infer(
            rgb_frame=frame,
            face_bbox=bbox,
            face_present=True
        )

        if result.data:
            print(f"EAR: {result.data['ear']}, MAR: {result.data['mar']}")
    """

    def __init__(
        self,
        detector: Optional[Any],
        latency_smoothing: float = 0.7
    ):
        """
        初始化疲劳检测引擎

        Args:
            detector: FatigueDetector 实例（None 时禁用）
            latency_smoothing: 延迟 EMA 平滑系数（0-1，越大越平滑）
        """
        self.detector = detector
        self.latency_smoothing = latency_smoothing

        # 内部状态
        self._latency_ms = 0.0
        self._last_timestamp = time.time()

        if self.detector is None:
            logger.info("FatigueEngine: 已禁用（无 detector）")
        else:
            logger.info("FatigueEngine: 已启用（每帧执行）")

    def should_infer(
        self,
        face_present: bool
    ) -> tuple[bool, str]:
        """
        判断是否需要执行推理（不改变状态）

        Args:
            face_present: 当前帧是否有人脸

        Returns:
            tuple[bool, str]: (是否需要推理, 原因说明)
        """
        # 引擎禁用
        if self.detector is None:
            return (False, "engine_disabled")

        # 无人脸
        if not face_present:
            return (False, "no_face")

        # 疲劳检测每帧都执行
        return (True, "every_frame")

    def infer(
        self,
        rgb_frame: np.ndarray,
        face_bbox: Optional[Dict[str, int]],
        face_present: bool,
        face_roi: Optional[Any] = None
    ) -> FatigueResult:
        """
        执行疲劳检测推理

        Args:
            rgb_frame: RGB 图像帧
            face_bbox: 人脸边界框 {'x1', 'y1', 'x2', 'y2'}
            face_present: 当前帧是否有人脸
            face_roi: 可选的预提取 RegionROI 对象（如果提供，直接使用而不重新裁剪）

        Returns:
            FatigueResult: 疲劳检测结果
        """
        # 引擎禁用
        if self.detector is None:
            return FatigueResult(
                data=None,
                latency_ms=0.0,
                speed_fps=0.0,
                source="disabled",
                timestamp=time.time()
            )

        # 无人脸
        if not face_present or face_bbox is None:
            return FatigueResult(
                data=None,
                latency_ms=self._latency_ms,
                speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                source="invalid_frame",
                timestamp=time.time()
            )

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
                    logger.warning(f"[Fatigue] 无效的人脸区域: ({x1},{y1})-({x2},{y2})")
                    return FatigueResult(
                        data=None,
                        latency_ms=self._latency_ms,
                        speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                        source="invalid_frame",
                        timestamp=time.time()
                    )

                face_img = rgb_frame[y1:y2, x1:x2]

            # 验证帧是否适合检测
            if not self.detector.validate_frame(face_img, is_cropped_face=True):
                return FatigueResult(
                    data=None,
                    latency_ms=self._latency_ms,
                    speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                    source="invalid_frame",
                    timestamp=time.time()
                )

            # 推理计时
            infer_start = time.time()
            fatigue_data = self.detector.detect_fatigue(face_img, is_cropped_face=True)
            infer_end = time.time()

            # 推理成功
            if fatigue_data:
                # 更新延迟统计（EMA 平滑）
                instant_latency_ms = (infer_end - infer_start) * 1000.0
                self._latency_ms = (
                    self.latency_smoothing * self._latency_ms +
                    (1 - self.latency_smoothing) * instant_latency_ms
                )

                # 添加延迟数据到结果
                fatigue_data['latency_ms'] = self._latency_ms
                fatigue_data['speed_fps'] = 1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0

                self._last_timestamp = time.time()

                return FatigueResult(
                    data=fatigue_data,
                    latency_ms=self._latency_ms,
                    speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                    source="inference",
                    timestamp=self._last_timestamp
                )
            else:
                logger.warning("[Fatigue] 推理返回空结果")
                return FatigueResult(
                    data=None,
                    latency_ms=self._latency_ms,
                    speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                    source="error",
                    timestamp=time.time()
                )

        except Exception as e:
            logger.error(f"[Fatigue] 推理失败: {e}")
            return FatigueResult(
                data=None,
                latency_ms=self._latency_ms,
                speed_fps=1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
                source="error",
                timestamp=time.time()
            )

    def reset(self):
        """重置引擎状态"""
        self._latency_ms = 0.0
        self._last_timestamp = time.time()
        logger.info("[Fatigue] 引擎已重置")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            统计字典（延迟、FPS 等）
        """
        return {
            "enabled": self.detector is not None,
            "latency_ms": self._latency_ms,
            "speed_fps": 1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0,
            "last_inference_time": self._last_timestamp
        }
