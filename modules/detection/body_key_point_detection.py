"""
人体关键点检测模块（TensorRT 专用版本）
仅支持 TensorRT .engine 文件，不依赖 Ultralytics
"""
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from ..core.logger import logger

try:
    from .trt_pose_detector import TRTPoseDetector
    TRT_POSE_AVAILABLE = True
except ImportError:
    TRT_POSE_AVAILABLE = False


class BodyKeyPointDetector:
    """
    人体关键点检测器（TensorRT 专用）

    仅接受 .engine 文件，直接使用 TensorRT 推理
    不再支持 .pt 文件或 Ultralytics 后端
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: tuple = (640, 640),
        **kwargs  # 兼容旧代码传入的 device/use_half 参数（忽略）
    ):
        """
        初始化人体关键点检测器（TensorRT 专用）

        Args:
            model_path: TensorRT 引擎文件路径（必须是 .engine 文件）
            confidence_threshold: 检测置信度阈值（默认 0.25，适用于 sigmoid 后的概率值）
            iou_threshold: NMS IoU 阈值
            input_size: 模型输入尺寸 (height, width)
            **kwargs: 兼容参数（device, use_half 等），被忽略

        Raises:
            ValueError: 如果 model_path 不是 .engine 文件
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果 TensorRT 不可用
        """
        engine_path = Path(model_path).resolve()

        # 严格检查：必须是 .engine 文件
        if engine_path.suffix != ".engine":
            raise ValueError(
                f"BodyKeyPointDetector 现在仅支持 TensorRT 引擎文件 (.engine)。\n"
                f"收到的文件: {engine_path}\n"
                f"请使用 trtexec 或 export_yolo_pose_onnx.py 将模型转换为 .engine 格式。"
            )

        if not engine_path.exists():
            raise FileNotFoundError(f"未找到 TensorRT 引擎文件: {engine_path}")

        if not TRT_POSE_AVAILABLE:
            raise RuntimeError(
                "TensorRT 后端不可用。请确保已安装 tensorrt 和 pycuda。\n"
                "安装命令: pip install tensorrt pycuda"
            )

        self.model_path = str(engine_path)
        self.confidence_threshold = float(max(0.0, min(1.0, confidence_threshold)))
        self.backend = "tensorrt"  # 固定为 TensorRT

        # 忽略传入的 device/use_half 参数（TensorRT 自动管理 GPU）
        if 'device' in kwargs or 'use_half' in kwargs:
            logger.debug("TensorRT 模式下忽略 device/use_half 参数（自动管理 GPU）")

        logger.info(f"使用 TensorRT 后端加载模型: {self.model_path}")

        # 初始化 TensorRT 检测器
        self.detector = TRTPoseDetector(
            engine_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
        )

        logger.info(
            f"人体关键点检测器初始化完成 (TensorRT), "
            f"置信度阈值: {self.confidence_threshold}, "
            f"输入尺寸: {input_size}"
        )

    def detect_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测单帧图像中的人体关键点

        Args:
            frame: 输入图像 (H, W, 3)，支持 BGR/RGB

        Returns:
            包含检测结果的字典，若未检测到人体则返回 None
            格式:
            {
                'detections': [
                    {
                        'bbox': {'x1': float, 'y1': float, 'x2': float, 'y2': float},
                        'confidence': float,
                        'class_id': int,
                        'keypoints': [
                            {'index': int, 'x': float, 'y': float, 'confidence': float},
                            ...
                        ]
                    },
                    ...
                ],
                'inference_time_ms': float,
                'device': 'TensorRT'
            }
        """
        if frame is None or frame.size == 0:
            logger.warning("检测输入为空，跳过本帧")
            return None

        # 直接调用 TensorRT 检测器
        return self.detector.detect_keypoints(frame)

    def set_confidence_threshold(self, threshold: float):
        """
        更新检测置信度阈值

        Args:
            threshold: 新的置信度阈值 [0.0, 1.0]
        """
        self.confidence_threshold = float(max(0.0, min(1.0, threshold)))
        self.detector.confidence_threshold = self.confidence_threshold
        logger.info(f"人体关键点检测置信度阈值更新为: {self.confidence_threshold}")

    def warmup(self, image_shape: tuple = (640, 640, 3), runs: int = 3):
        """
        预热模型，减少首帧延迟

        Args:
            image_shape: 预热图像尺寸 (H, W, C)
            runs: 预热运行次数
        """
        logger.info(f"开始进行 TensorRT 人体关键点检测器预热 ({runs} 次)")
        self.detector.warmup(runs=runs)
        logger.info("TensorRT 预热完成")

    def get_model_info(self) -> Dict:
        """
        返回模型状态信息

        Returns:
            包含模型路径、后端、性能统计等信息的字典
        """
        info = self.detector.get_model_info()
        info["model_path"] = self.model_path
        info["backend"] = "TensorRT"  # 确保后端标识一致
        return info


# 向后兼容别名（如果其他模块使用了旧名称）
__all__ = ['BodyKeyPointDetector']
