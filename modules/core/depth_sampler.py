"""
深度采样器
=========
在人体/人脸 bbox 内做多点采样，忽略无效/异常深度，输出深度估计和置信度。

解决的问题：
1. 采到桌面/空白区域深度乱跳
2. 前后两人深度一样导致选错人
3. 深度不可靠时（被遮挡、边缘噪声）误判
"""
from typing import Dict, Tuple, Optional
import numpy as np
from .logger import logger


class DepthSampler:
    """
    多点深度采样器

    在 bbox 的上半身区域进行网格采样，过滤无效点后计算鲁棒深度估计。

    使用示例:
        sampler = DepthSampler(
            max_depth_mm=2500.0,
            min_valid_ratio=0.25,
            grid_size=(5, 3),
            vertical_band=(0.1, 0.6)
        )

        result = sampler.sample(depth_frame, face_bbox)
        if result['depth_m'] is not None:
            print(f"深度: {result['depth_m']:.2f}m, 置信度: {result['depth_conf']:.2f}")
    """

    def __init__(
        self,
        max_depth_mm: float = 2500.0,
        min_valid_ratio: float = 0.25,
        grid_size: Tuple[int, int] = (5, 3),
        vertical_band: Tuple[float, float] = (0.1, 0.6),
    ):
        """
        初始化深度采样器

        Args:
            max_depth_mm: 超过此值视为无效点（毫米）
            min_valid_ratio: 有效点比例低于此值认为深度不可靠
            grid_size: (nx, ny) 采样网格大小
            vertical_band: 只在 bbox 的这一段高度内采样（避免桌面/腿）
                - (0.1, 0.6) 表示在 bbox 高度的 10%~60% 区域内采样
                - 对于人体框：上半身大概在这个区域
                - 对于人脸框：整个脸部区域
        """
        self.max_depth_mm = max_depth_mm
        self.min_valid_ratio = min_valid_ratio
        self.grid_w, self.grid_h = grid_size
        self.band_top, self.band_bottom = vertical_band

        logger.debug(
            f"DepthSampler 初始化: max_depth={max_depth_mm}mm, "
            f"min_valid_ratio={min_valid_ratio}, "
            f"grid={grid_size}, vertical_band={vertical_band}"
        )

    def sample(self, depth_frame: np.ndarray, bbox) -> Dict:
        """
        对一个 bbox 进行多点采样

        Args:
            depth_frame: 深度图（毫米单位的 uint16 或 float）
            bbox: BBox 对象或有 x1/y1/x2/y2 属性的对象

        Returns:
            {
                'depth_m': Optional[float],  # 估计深度（米），None 表示不可靠
                'valid_ratio': float,        # 有效采样点比例（0.0~1.0）
                'depth_conf': float,         # 深度置信度（0.0~1.0）
                'raw_depths_mm': list,       # 原始有效深度值（调试用）
            }

        采样策略：
            1. 在 bbox 的 vertical_band 区域内生成网格采样点
            2. 过滤无效点（0 值、超出 max_depth_mm、超出图像边界）
            3. 计算有效点比例（valid_ratio）
            4. 如果有效点比例太低，返回 None + 低置信度
            5. 使用四分位数去除离群值，计算鲁棒中位数深度
            6. 置信度 = f(有效点比例, 深度一致性)
        """
        result = {
            'depth_m': None,
            'valid_ratio': 0.0,
            'depth_conf': 0.0,
            'raw_depths_mm': [],
        }

        if depth_frame is None:
            return result

        # 获取 bbox 坐标
        try:
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"[DepthSampler] bbox 坐标无效: {e}")
            return result

        # bbox 尺寸检查
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return result

        # 计算采样区域（vertical_band）
        sample_y1 = int(y1 + h * self.band_top)
        sample_y2 = int(y1 + h * self.band_bottom)

        if sample_y2 <= sample_y1:
            sample_y2 = sample_y1 + 1

        # 生成网格采样点
        img_h, img_w = depth_frame.shape[:2]
        sample_points = []

        for ix in range(self.grid_w):
            # 水平方向均匀分布，避开边缘
            px = x1 + int(w * (ix + 0.5) / self.grid_w)
            for iy in range(self.grid_h):
                # 垂直方向在 band 内均匀分布
                py = sample_y1 + int((sample_y2 - sample_y1) * (iy + 0.5) / self.grid_h)

                # 边界检查
                if 0 <= px < img_w and 0 <= py < img_h:
                    sample_points.append((px, py))

        if not sample_points:
            return result

        # 采样深度值
        valid_depths = []
        total_points = len(sample_points)

        for px, py in sample_points:
            depth_val = float(depth_frame[py, px])

            # 过滤无效值
            if depth_val <= 0:
                continue
            if depth_val > self.max_depth_mm:
                continue

            valid_depths.append(depth_val)

        result['raw_depths_mm'] = valid_depths.copy()

        # 计算有效点比例
        valid_ratio = len(valid_depths) / total_points if total_points > 0 else 0.0
        result['valid_ratio'] = valid_ratio

        # 有效点太少，不可靠
        if valid_ratio < self.min_valid_ratio or len(valid_depths) < 3:
            result['depth_conf'] = valid_ratio * 0.5  # 低置信度
            return result

        # 使用四分位数去除离群值
        valid_depths_arr = np.array(valid_depths)
        q1, q3 = np.percentile(valid_depths_arr, [25, 75])
        iqr = q3 - q1

        # IQR 过滤（1.5 倍 IQR 规则）
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_depths = valid_depths_arr[
            (valid_depths_arr >= lower_bound) & (valid_depths_arr <= upper_bound)
        ]

        if len(filtered_depths) < 2:
            # 过滤后点太少，使用原始中位数
            filtered_depths = valid_depths_arr

        # 计算鲁棒深度估计（中位数）
        depth_median_mm = float(np.median(filtered_depths))
        result['depth_m'] = depth_median_mm / 1000.0

        # 计算深度一致性（标准差越小越一致）
        depth_std = float(np.std(filtered_depths))
        # 归一化：假设 50mm 标准差为一致性 0.5
        consistency = max(0.0, 1.0 - depth_std / 100.0)

        # 综合置信度 = valid_ratio * (0.5 + 0.5 * consistency)
        # valid_ratio 高 + 深度一致 -> 高置信度
        result['depth_conf'] = valid_ratio * (0.5 + 0.5 * consistency)

        return result

    def sample_with_fallback(
        self,
        depth_frame: np.ndarray,
        face_bbox,
        person_bbox
    ) -> Dict:
        """
        带回退的深度采样：优先用人脸，回退到人体上半身

        Args:
            depth_frame: 深度图
            face_bbox: 人脸 bbox（可能为 None）
            person_bbox: 人体 bbox（可能为 None）

        Returns:
            与 sample() 相同格式的结果，额外增加 'source' 字段
        """
        result = {
            'depth_m': None,
            'valid_ratio': 0.0,
            'depth_conf': 0.0,
            'raw_depths_mm': [],
            'source': None,
        }

        # 优先尝试人脸 bbox
        if face_bbox is not None:
            face_result = self.sample(depth_frame, face_bbox)
            if face_result['depth_conf'] > 0.3:  # 人脸深度可信
                face_result['source'] = 'face'
                return face_result

        # 回退到人体 bbox
        if person_bbox is not None:
            person_result = self.sample(depth_frame, person_bbox)
            person_result['source'] = 'person'
            return person_result

        result['source'] = 'none'
        return result
