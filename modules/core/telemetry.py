"""
遥测数据构建模块
===============

负责构建、格式化和输出系统遥测数据（Telemetry）。
"""
import json
import time
from typing import Optional, Dict, Any, List
from ..compat import np
from .logger import logger


def format_floats(obj: Any) -> Any:
    """
    递归格式化对象中的浮点数为3位小数

    Args:
        obj: 要格式化的对象（dict、list、float、ndarray等）

    Returns:
        格式化后的对象
    """
    if isinstance(obj, dict):
        return {k: format_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_floats(item) for item in obj]
    elif isinstance(obj, float):
        return round(obj, 3)
    elif isinstance(obj, np.ndarray):
        return [round(float(x), 3) for x in obj.tolist()]
    elif isinstance(obj, np.generic):
        val = obj.item()
        return round(val, 3) if isinstance(val, float) else val
    else:
        return obj


class TelemetryBuilder:
    """
    遥测数据构建器

    职责:
    - 从检测结果构建 telemetry 字典
    - 格式化浮点数为固定精度
    - 打印 JSON 格式的 telemetry

    使用示例:
        builder = TelemetryBuilder(
            print_enabled=True,
            print_interval=5
        )

        # 有检测结果时
        telemetry = builder.build(
            results=results,
            stable_dist=0.5,
            shoulder_midpoint_to_desk=0.3,
            frame_count=100,
            global_fps=30.0
        )

        # 无检测结果时
        telemetry = builder.build_empty(
            frame_count=101,
            global_fps=30.0,
            reuse_last=True
        )
    """

    def __init__(
        self,
        print_enabled: bool = True,
        print_interval: int = 5
    ):
        """
        初始化遥测构建器

        Args:
            print_enabled: 是否启用 telemetry 打印
            print_interval: 打印间隔（每N帧打印一次）
        """
        self.print_enabled = print_enabled
        self.print_interval = max(1, print_interval)
        self._last_telemetry: Optional[Dict[str, Any]] = None

    def build(
        self,
        results: Dict[str, Any],
        stable_dist: Optional[float],
        shoulder_midpoint: Optional[List[float]],
        shoulder_midpoint_to_desk: Optional[float],
        frame_count: int,
        global_fps: float,
        left_shoulder: Optional[List[float]] = None,
        right_shoulder: Optional[List[float]] = None,
        left_eye: Optional[List[float]] = None,
        right_eye: Optional[List[float]] = None,
        nose: Optional[List[float]] = None,
        async_stats: Optional[Dict[str, Any]] = None,
        camera_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        构建 telemetry 数据（有检测结果时）

        Args:
            results: 系统检测结果字典
            stable_dist: 稳定的眼距（米）
            shoulder_midpoint: 肩部中点坐标（3D）
            shoulder_midpoint_to_desk: 肩部中点到桌面的距离（米）
            frame_count: 当前帧计数
            global_fps: 全局帧率
            left_shoulder: 左肩 3D 坐标（可选）
            right_shoulder: 右肩 3D 坐标（可选）
            left_eye: 左眼 3D 坐标（可选）
            right_eye: 右眼 3D 坐标（可选）
            nose: 鼻子 3D 坐标（可选）
            async_stats: 异步推理统计信息（可选）
            camera_telemetry: 相机恢复统计信息（可选）

        Returns:
            telemetry 字典
        """
        telemetry = {
            # 眼距数据（厘米）
            "distance_cm": stable_dist * 100 if stable_dist is not None else None,

            # 情绪识别数据
            "emotion": {
                "prediction": results.get("emotion", {}).get("emotion") if results.get("emotion") else None,
                "valence": results.get("emotion", {}).get("valence") if results.get("emotion") else None,
                "arousal": results.get("emotion", {}).get("arousal") if results.get("emotion") else None,
            },

            # 疲劳检测数据
            "fatigue": {
                "level": results.get("fatigue", {}).get("fatigue_level") if results.get("fatigue") else None,
            },

            # 姿态角度数据
            "posture": {
                "forward_lean": results.get("pose", {}).get("angles", {}).get("head_forward_angle") if results.get("pose", {}).get("angles") else None,
                "shoulder_tilt": results.get("pose", {}).get("angles", {}).get("shoulder_tilt_angle") if results.get("pose", {}).get("angles") else None,
                "head_roll": results.get("pose", {}).get("angles", {}).get("head_roll_angle") if results.get("pose", {}).get("angles") else None,
            },

            # 3D 关键点坐标（左肩、右肩、左眼、右眼、鼻子、肩部中点）
            # 优先使用直接传入的参数，否则从 results 中提取
            "keypoints_3d": {
                "left_shoulder": left_shoulder if left_shoulder is not None else (
                    results.get("pose", {}).get("keypoints_3d", {}).get("left_shoulder") if results.get("pose", {}).get("keypoints_3d") else None
                ),
                "right_shoulder": right_shoulder if right_shoulder is not None else (
                    results.get("pose", {}).get("keypoints_3d", {}).get("right_shoulder") if results.get("pose", {}).get("keypoints_3d") else None
                ),
                "left_eye": left_eye if left_eye is not None else (
                    results.get("pose", {}).get("keypoints_3d", {}).get("left_eye_center") if results.get("pose", {}).get("keypoints_3d") else None
                ),
                "right_eye": right_eye if right_eye is not None else (
                    results.get("pose", {}).get("keypoints_3d", {}).get("right_eye_center") if results.get("pose", {}).get("keypoints_3d") else None
                ),
                "nose": nose if nose is not None else (
                    results.get("pose", {}).get("keypoints_3d", {}).get("nose") if results.get("pose", {}).get("keypoints_3d") else None
                ),
                "shoulder_midpoint": shoulder_midpoint,
            },

            # 肩部中点到桌面的距离（米）
            "shoulder_midpoint_to_desk_m": shoulder_midpoint_to_desk,

            # 时间戳和帧信息
            "timestamp": time.time(),
            "frame_count": frame_count,
            "global_fps": global_fps if global_fps > 0 else None,
        }

        # 添加异步推理统计（如果提供）
        if async_stats:
            telemetry["async_inference"] = self._format_async_stats(async_stats)

        # 添加相机恢复统计（如果提供）
        if camera_telemetry:
            telemetry["camera"] = {
                "status": camera_telemetry.get("status", "unknown"),
                "soft_restarts": camera_telemetry.get("total_soft_restarts", 0),
                "reopens": camera_telemetry.get("total_reopens", 0),
                "usb_resets": camera_telemetry.get("total_usb_resets", 0),
                "consecutive_failures": camera_telemetry.get("consecutive_failures", 0),
                "frame_age_ms": camera_telemetry.get("frame_age_ms", 0)
            }

        # 缓存当前 telemetry
        self._last_telemetry = telemetry

        # 打印 telemetry（如果启用）
        self._print_if_enabled(telemetry, frame_count)

        return telemetry

    def build_empty(
        self,
        frame_count: int,
        global_fps: float,
        reuse_last: bool = True,
        camera_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        构建空 telemetry 数据（无检测结果时）

        Args:
            frame_count: 当前帧计数
            global_fps: 全局帧率
            reuse_last: 是否复用上一帧数据（更新时间戳）
            camera_telemetry: 相机恢复统计信息（可选）

        Returns:
            telemetry 字典
        """
        if reuse_last and self._last_telemetry is not None:
            # 沿用上一帧数据，但更新时间戳和状态
            telemetry = self._last_telemetry.copy()
            telemetry["timestamp"] = time.time()
            telemetry["frame_count"] = frame_count
            telemetry["global_fps"] = global_fps if global_fps > 0 else None
            telemetry["status"] = "no_detection"  # 标记为无检测状态
        else:
            # 创建全新的空 telemetry
            telemetry = {
                "distance_cm": None,
                "emotion": {"prediction": None, "valence": None, "arousal": None},
                "fatigue": {"level": None},
                "posture": {"forward_lean": None, "shoulder_tilt": None, "head_roll": None},
                "keypoints_3d": {
                    "left_shoulder": None,
                    "right_shoulder": None,
                    "left_eye": None,
                    "right_eye": None,
                    "nose": None,
                    "shoulder_midpoint": None
                },
                "shoulder_midpoint_to_desk_m": None,
                "timestamp": time.time(),
                "frame_count": frame_count,
                "global_fps": global_fps if global_fps > 0 else None,
                "status": "no_detection"  # 标记为无检测状态
            }

        # 添加相机恢复统计（如果提供）
        if camera_telemetry:
            telemetry["camera"] = {
                "status": camera_telemetry.get("status", "unknown"),
                "soft_restarts": camera_telemetry.get("total_soft_restarts", 0),
                "reopens": camera_telemetry.get("total_reopens", 0),
                "usb_resets": camera_telemetry.get("total_usb_resets", 0),
                "consecutive_failures": camera_telemetry.get("consecutive_failures", 0),
                "frame_age_ms": camera_telemetry.get("frame_age_ms", 0)
            }

        # 更新缓存
        self._last_telemetry = telemetry

        # 打印 telemetry（如果启用）
        self._print_if_enabled(telemetry, frame_count)

        return telemetry

    def _print_if_enabled(self, telemetry: Dict[str, Any], frame_count: int):
        """
        根据配置打印 telemetry（带格式化）

        Args:
            telemetry: 要打印的 telemetry 字典
            frame_count: 当前帧计数
        """
        if not self.print_enabled:
            return

        if frame_count % self.print_interval != 0:
            return

        # 格式化浮点数为3位小数
        formatted_telemetry = format_floats(telemetry)

        # 打印 JSON 格式（确保 flush，避免缓冲）
        print(f"\n[Telemetry] {json.dumps(formatted_telemetry, ensure_ascii=False)}\n", flush=True)

    def get_last_telemetry(self) -> Optional[Dict[str, Any]]:
        """
        获取上一帧的 telemetry 数据

        Returns:
            上一帧的 telemetry 字典，如果没有则返回 None
        """
        return self._last_telemetry

    def reset(self):
        """重置 telemetry 缓存"""
        self._last_telemetry = None

    def _format_async_stats(self, async_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化异步推理统计信息（简化版本，仅包含关键指标）

        Args:
            async_stats: 原始异步统计数据

        Returns:
            格式化后的统计数据
        """
        formatted = {}

        for engine_name, stats in async_stats.items():
            if not isinstance(stats, dict):
                continue

            formatted[engine_name] = {
                "tasks_submitted": stats.get("tasks_submitted", 0),
                "tasks_processed": stats.get("tasks_processed", 0),
                "tasks_dropped": stats.get("tasks_dropped", 0),
                "last_inference_ms": round(stats.get("last_inference_time", 0) * 1000, 1) if stats.get("last_inference_time") else None,
                "avg_inference_ms": round(
                    (stats.get("total_inference_time", 0) / stats.get("tasks_processed", 1)) * 1000, 1
                ) if stats.get("tasks_processed", 0) > 0 else None,
                "worker_active": stats.get("worker_active", False),
                "errors": stats.get("errors", 0)
            }

        return formatted
