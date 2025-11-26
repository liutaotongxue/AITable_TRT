"""
ROI 提取管理器
==============

统一的人脸 ROI 裁剪逻辑，避免重复裁剪。
"""
import time
from typing import Optional, Dict, Tuple
from ..compat import np
from ..core.region_roi import RegionROI
from ..core.logger import logger


class ROIManager:
    """
    ROI 提取管理器

    职责：
    - 统一的 bbox 校验和裁剪逻辑
    - 支持可配置的 margin（扩展边界）
    - 集中容错和统计

    使用示例：
        roi_manager = ROIManager(margin=0.1, min_size=20)

        # 提取 ROI
        face_roi = roi_manager.extract(rgb_frame, face_bbox)

        if face_roi:
            # 传递给多个引擎（避免重复裁剪）
            emotion_engine.infer(face_roi, ...)
            fatigue_engine.infer(face_roi, ...)

        # 查看统计
        stats = roi_manager.get_stats()
    """

    def __init__(
        self,
        margin: float = 0.0,
        min_size: int = 20,
        max_size: int = 1920
    ):
        """
        初始化 ROI 管理器

        Args:
            margin: bbox 扩展比例（0.1 = 扩展 10%）
            min_size: 最小 ROI 尺寸（像素）
            max_size: 最大 ROI 尺寸（像素）
        """
        self.margin = margin
        self.min_size = min_size
        self.max_size = max_size

        # 统计信息
        self._stats = {
            'total': 0,           # 总请求数
            'invalid_bbox': 0,    # 无效 bbox
            'too_small': 0,       # 尺寸过小
            'too_large': 0,       # 尺寸过大
            'success': 0          # 成功提取
        }

    def extract(
        self,
        rgb_frame: np.ndarray,
        face_bbox: Dict[str, int],
        resize: Optional[Tuple[int, int]] = None,
        roi_type: Optional[str] = None
    ) -> Optional[RegionROI]:
        """
        提取 ROI（支持人脸、人体等多种区域）

        Args:
            rgb_frame: 原始 RGB 图像 (H, W, 3)
            face_bbox: 检测的 bbox {'x1', 'y1', 'x2', 'y2', 'confidence'}
            resize: 可选的目标尺寸 (width, height)
                   如果不指定，保持原始尺寸
            roi_type: ROI 类型标识（'face', 'person', 'hand' 等）

        Returns:
            RegionROI 或 None（提取失败）

        示例：
            # 提取人脸 ROI
            face_roi = roi_manager.extract(rgb_frame, face_bbox, resize=(256, 256), roi_type='face')

            # 提取人体 ROI
            person_roi = roi_manager.extract(rgb_frame, person_bbox, roi_type='person')
        """
        self._stats['total'] += 1

        # 1. 校验 bbox
        if not self._validate_bbox(face_bbox, rgb_frame.shape):
            self._stats['invalid_bbox'] += 1
            logger.warning(f"[ROIManager] 无效 bbox: {face_bbox}")
            return None

        # 2. 扩展 margin 并 clamp 到图像边界
        x1, y1, x2, y2, offset = self._expand_margin(
            face_bbox, rgb_frame.shape, self.margin
        )

        # 3. 检查尺寸
        roi_w, roi_h = x2 - x1, y2 - y1

        if roi_w < self.min_size or roi_h < self.min_size:
            self._stats['too_small'] += 1
            logger.warning(f"[ROIManager] ROI 尺寸过小: {roi_w}x{roi_h}")
            return None

        if roi_w > self.max_size or roi_h > self.max_size:
            self._stats['too_large'] += 1
            logger.debug(f"[ROIManager] ROI 尺寸过大: {roi_w}x{roi_h}")
            # 不返回 None，允许继续处理（会被 resize 缩小）

        # 4. 裁剪（创建副本，避免修改原图）
        face_img = rgb_frame[y1:y2, x1:x2].copy()

        # 5. Resize（可选）
        if resize:
            import cv2
            face_img = cv2.resize(face_img, resize)
            scale = resize[0] / roi_w
        else:
            scale = 1.0

        self._stats['success'] += 1

        # 6. 封装为 RegionROI
        return RegionROI(
            bbox=face_bbox,
            image=face_img,
            scale=scale,
            offset=offset,
            timestamp=time.time(),
            confidence=face_bbox.get('confidence', 1.0),
            roi_type=roi_type  # 添加类型标识
        )

    def _validate_bbox(
        self,
        bbox: Dict[str, int],
        frame_shape: Tuple[int, ...]
    ) -> bool:
        """
        校验 bbox 是否有效

        Args:
            bbox: {'x1', 'y1', 'x2', 'y2'}
            frame_shape: 图像 shape (H, W, C)

        Returns:
            bool: 是否有效
        """
        try:
            h, w = frame_shape[:2]
            x1, y1 = bbox['x1'], bbox['y1']
            x2, y2 = bbox['x2'], bbox['y2']

            # 基本检查
            if x2 <= x1 or y2 <= y1:
                logger.debug(f"[ROIManager] bbox 宽高为负: ({x1},{y1})-({x2},{y2})")
                return False

            # 边界检查（允许部分超出，后续会 clamp）
            if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
                logger.debug(f"[ROIManager] bbox 完全超出图像边界")
                return False

            return True

        except (KeyError, TypeError) as e:
            logger.warning(f"[ROIManager] bbox 格式错误: {e}")
            return False

    def _expand_margin(
        self,
        bbox: Dict[str, int],
        frame_shape: Tuple[int, ...],
        margin: float
    ) -> Tuple[int, int, int, int, Tuple[int, int]]:
        """
        扩展 bbox margin 并 clamp 到图像边界

        Args:
            bbox: 原始 bbox
            frame_shape: 图像 shape
            margin: 扩展比例（0.1 = 10%）

        Returns:
            (x1, y1, x2, y2, offset):
                - x1, y1, x2, y2: 扩展后的坐标
                - offset: (offset_x, offset_y) 左上角偏移量
        """
        h, w = frame_shape[:2]
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']

        # 计算 margin
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        margin_w = int(bbox_w * margin)
        margin_h = int(bbox_h * margin)

        # 扩展并 clamp
        x1_exp = max(0, x1 - margin_w)
        y1_exp = max(0, y1 - margin_h)
        x2_exp = min(w, x2 + margin_w)
        y2_exp = min(h, y2 + margin_h)

        # 计算偏移（用于坐标映射）
        offset = (x1_exp - x1, y1_exp - y1)

        return x1_exp, y1_exp, x2_exp, y2_exp, offset

    def extract_dual(
        self,
        rgb_frame: np.ndarray,
        face_bbox: Optional[Dict[str, int]] = None,
        person_bbox: Optional[Dict[str, int]] = None,
        face_resize: Optional[Tuple[int, int]] = None,
        person_resize: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Optional[RegionROI]]:
        """
        同时提取 face 和 person ROI（face + person 关联场景）

        Args:
            rgb_frame: 原始 RGB 图像 (H, W, 3)
            face_bbox: 人脸 bbox（可选）{'x1', 'y1', 'x2', 'y2', 'confidence'}
            person_bbox: 人体 bbox（可选）{'x1', 'y1', 'x2', 'y2', 'confidence'}
            face_resize: face ROI 的目标尺寸（可选）
            person_resize: person ROI 的目标尺寸（可选）

        Returns:
            {'face_roi': RegionROI|None, 'person_roi': RegionROI|None}

        注意：
            - face_bbox 为 None 时，只提取 person_roi（BODY_ONLY 模式）
            - person_bbox 为 None 时，只提取 face_roi（FACE_ONLY 模式）
            - 提取失败时，对应的 roi 为 None
            - face_roi 和 person_roi 都使用 RegionROI（通用结构），通过 roi_type 字段区分

        示例：
            # FULL 模式（有脸+有人体）
            rois = roi_manager.extract_dual(
                rgb_frame,
                face_bbox={'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'confidence': 0.95},
                person_bbox={'x1': 80, 'y1': 80, 'x2': 220, 'y2': 400, 'confidence': 0.88}
            )
            # rois['face_roi'].roi_type == 'face'
            # rois['person_roi'].roi_type == 'person'

            # BODY_ONLY 模式（无脸，仅人体）
            rois = roi_manager.extract_dual(
                rgb_frame,
                face_bbox=None,
                person_bbox={'x1': 80, 'y1': 80, 'x2': 220, 'y2': 400, 'confidence': 0.88}
            )
            # rois['face_roi'] == None
            # rois['person_roi'].roi_type == 'person'
        """
        result = {
            'face_roi': None,
            'person_roi': None
        }

        # 提取 face ROI（如果有）
        if face_bbox:
            result['face_roi'] = self.extract(
                rgb_frame, face_bbox,
                resize=face_resize,
                roi_type='face'  # 明确标识为人脸 ROI
            )

        # 提取 person ROI（如果有）
        if person_bbox:
            result['person_roi'] = self.extract(
                rgb_frame, person_bbox,
                resize=person_resize,
                roi_type='person'  # 明确标识为人体 ROI
            )

        return result

    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            统计字典：
                - total: 总请求数
                - invalid_bbox: 无效 bbox 数
                - too_small: 过小 ROI 数
                - success: 成功提取数
        """
        return self._stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        for key in self._stats:
            self._stats[key] = 0
        logger.info("[ROIManager] 统计信息已重置")

    def __repr__(self):
        success_rate = (
            self._stats['success'] / self._stats['total'] * 100
            if self._stats['total'] > 0 else 0
        )
        return (
            f"ROIManager(margin={self.margin}, "
            f"success_rate={success_rate:.1f}%, "
            f"stats={self._stats})"
        )
