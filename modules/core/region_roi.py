"""
通用 ROI 数据结构
================

封装图像区域（人脸、人体等）的图像和元数据，支持多线程安全访问。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time
from ..compat import np


@dataclass(frozen=True)
class RegionROI:
    """
    通用 ROI（Region of Interest）数据结构

    职责：
    - 封装裁剪后的图像区域和元数据（支持人脸、人体、手部等多种区域）
    - 保证多线程安全（只读 numpy 数组）
    - 提供坐标映射功能（ROI -> 原图）

    使用示例：
        # 人脸 ROI
        face_roi = RegionROI(
            bbox={'x1': 100, 'y1': 50, 'x2': 300, 'y2': 250},
            image=face_img,
            scale=1.0,
            offset=(0, 0),
            timestamp=time.time(),
            confidence=0.95,
            roi_type='face'  # 可选：明确标识类型
        )

        # 人体 ROI
        person_roi = RegionROI(
            bbox={'x1': 80, 'y1': 80, 'x2': 320, 'y2': 480},
            image=person_img,
            scale=1.0,
            offset=(0, 0),
            timestamp=time.time(),
            confidence=0.88,
            roi_type='person'  # 可选：明确标识类型
        )

        # 多线程安全读取
        emotion_img = face_roi.image  # 只读，不会被修改

        # 按需 resize
        resized = face_roi.resize((256, 256))
    """

    bbox: Dict[str, int]  # 原图中的边界框 {'x1', 'y1', 'x2', 'y2', 'confidence'}
    image: np.ndarray     # 裁剪后的 RGB/BGR 图像（只读）
    scale: float          # ROI 相对原 bbox 的缩放比例（如果做了 resize）
    offset: Tuple[int, int]  # ROI 左上角相对 bbox 的偏移（如果扩展了 margin）
    timestamp: float      # 提取时间戳
    confidence: float = 1.0  # 检测置信度
    roi_type: Optional[str] = None  # ROI 类型标识（'face', 'person', 'hand' 等）

    def __post_init__(self):
        """
        初始化后处理：确保 numpy 数组只读

        这样可以安全地在多个线程间共享 ROI，
        避免意外修改导致的数据竞争。
        """
        if self.image.flags.writeable:
            # 创建只读副本
            img_copy = self.image.copy()
            img_copy.setflags(write=False)
            object.__setattr__(self, 'image', img_copy)

    def resize(self, target_size: Tuple[int, int]) -> np.ndarray:
        """
        按需 resize ROI（不修改原图）

        Args:
            target_size: 目标尺寸 (width, height)

        Returns:
            np.ndarray: resize 后的图像（新副本）

        示例：
            emotion_img = roi.resize((256, 256))  # EmoNet 需要 256x256
        """
        import cv2
        return cv2.resize(self.image, target_size)

    def map_to_original(self, x: int, y: int) -> Tuple[int, int]:
        """
        将 ROI 内坐标映射回原图坐标

        Args:
            x, y: ROI 内坐标

        Returns:
            (orig_x, orig_y): 原图坐标

        应用场景：
            - FaceMesh 关键点需要映射回原图
            - 可视化时需要在原图上绘制
        """
        orig_x = self.bbox['x1'] + self.offset[0] + x / self.scale
        orig_y = self.bbox['y1'] + self.offset[1] + y / self.scale
        return (int(orig_x), int(orig_y))

    @property
    def size(self) -> Tuple[int, int]:
        """返回 ROI 尺寸 (width, height)"""
        return (self.image.shape[1], self.image.shape[0])

    @property
    def roi_rgb(self) -> np.ndarray:
        """
        返回 RGB 格式的 ROI 图像（向后兼容属性）

        注意：如果原图是 BGR 格式，此属性仍返回原图（不做转换）
        调用方需要自行处理颜色空间转换
        """
        return self.image

    @property
    def width(self) -> int:
        """ROI 宽度"""
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """ROI 高度"""
        return self.image.shape[0]

    @property
    def is_valid(self) -> bool:
        """检查 ROI 是否有效"""
        return (
            self.image is not None and
            self.image.size > 0 and
            self.size[0] > 0 and
            self.size[1] > 0
        )

    def __repr__(self):
        type_str = f", type={self.roi_type}" if self.roi_type else ""
        return (
            f"RegionROI(bbox={self.bbox}, size={self.size}, "
            f"confidence={self.confidence:.2f}{type_str}, timestamp={self.timestamp:.3f})"
        )
