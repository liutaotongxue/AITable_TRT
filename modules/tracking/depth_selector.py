"""
DepthSelector - 基于深度的主目标选择器

选人策略：
1. 最近的人优先（深度最小）
2. 但不会频繁切换（需要显著更近才切换）
3. 支持深度范围过滤
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DepthSample:
    """深度采样结果"""
    depth_mm: float  # 深度值（毫米）
    valid: bool  # 是否有效
    sample_count: int  # 有效采样点数量
    confidence: float  # 置信度


class DepthSelector:
    """
    基于深度的主目标选择器
    
    特性：
    - 选择深度最小（最近）的人作为主目标
    - 切换需要显著深度差（避免抖动）
    - 支持深度范围过滤
    - 使用区域中位数采样（抗噪声）
    """
    
    def __init__(
        self,
        depth_range_mm: Tuple[float, float] = (200, 1500),
        switch_depth_delta_mm: float = 100,
        sample_region_ratio: float = 0.3,
        min_valid_samples: int = 10,
    ):
        """
        Args:
            depth_range_mm: 有效深度范围 (min, max)，单位毫米
            switch_depth_delta_mm: 切换主目标需要的深度差，单位毫米
            sample_region_ratio: 采样区域占 bbox 的比例（中心区域）
            min_valid_samples: 最少有效采样点数量
        """
        self.depth_range_mm = depth_range_mm
        self.switch_depth_delta_mm = switch_depth_delta_mm
        self.sample_region_ratio = sample_region_ratio
        self.min_valid_samples = min_valid_samples
        
        # 当前主目标 ID
        self._current_primary_id: Optional[int] = None
    
    def sample_depth(
        self,
        depth_frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> DepthSample:
        """
        在边界框区域内采样深度
        
        Args:
            depth_frame: 深度图 (H, W)，单位毫米
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            DepthSample 对象
        """
        if depth_frame is None or depth_frame.size == 0:
            return DepthSample(
                depth_mm=0,
                valid=False,
                sample_count=0,
                confidence=0,
            )
        
        x1, y1, x2, y2 = bbox
        h, w = depth_frame.shape[:2]
        
        # 计算采样区域（中心部分）
        box_w = x2 - x1
        box_h = y2 - y1
        
        margin_x = box_w * (1 - self.sample_region_ratio) / 2
        margin_y = box_h * (1 - self.sample_region_ratio) / 2
        
        sample_x1 = int(max(0, x1 + margin_x))
        sample_y1 = int(max(0, y1 + margin_y))
        sample_x2 = int(min(w, x2 - margin_x))
        sample_y2 = int(min(h, y2 - margin_y))
        
        # 确保区域有效
        if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
            return DepthSample(
                depth_mm=0,
                valid=False,
                sample_count=0,
                confidence=0,
            )
        
        # 提取采样区域
        region = depth_frame[sample_y1:sample_y2, sample_x1:sample_x2]
        
        # 过滤有效深度值
        min_depth, max_depth = self.depth_range_mm
        valid_mask = (region > min_depth) & (region < max_depth) & (region > 0)
        valid_depths = region[valid_mask]
        
        if len(valid_depths) < self.min_valid_samples:
            return DepthSample(
                depth_mm=0,
                valid=False,
                sample_count=len(valid_depths),
                confidence=0,
            )
        
        # 使用中位数（抗噪声）
        depth_mm = float(np.median(valid_depths))
        
        # 计算置信度（基于有效样本比例）
        total_samples = region.size
        valid_ratio = len(valid_depths) / total_samples if total_samples > 0 else 0
        confidence = min(1.0, valid_ratio * 2)  # 50% 有效就是满置信度
        
        return DepthSample(
            depth_mm=depth_mm,
            valid=True,
            sample_count=len(valid_depths),
            confidence=confidence,
        )
    
    def select_primary(
        self,
        candidates: List[Dict],
        depth_frame: np.ndarray,
    ) -> Optional[int]:
        """
        选择主目标
        
        Args:
            candidates: 候选目标列表，每个元素需要包含：
                - 'track_id': int
                - 'bbox': (x1, y1, x2, y2)
            depth_frame: 深度图
            
        Returns:
            选中的 track_id，或 None
        """
        if not candidates:
            self._current_primary_id = None
            return None
        
        # 计算每个候选的深度
        depths = {}
        for cand in candidates:
            track_id = cand['track_id']
            bbox = cand['bbox']
            
            sample = self.sample_depth(depth_frame, bbox)
            if sample.valid:
                depths[track_id] = sample.depth_mm
        
        if not depths:
            # 没有有效深度，保持当前选择或选第一个
            if self._current_primary_id in [c['track_id'] for c in candidates]:
                return self._current_primary_id
            self._current_primary_id = candidates[0]['track_id']
            return self._current_primary_id
        
        # 找到最近的候选
        closest_id = min(depths, key=depths.get)
        closest_depth = depths[closest_id]
        
        # 检查是否需要切换
        if self._current_primary_id is not None:
            if self._current_primary_id in depths:
                current_depth = depths[self._current_primary_id]
                
                # 只有当新目标显著更近时才切换
                if current_depth - closest_depth < self.switch_depth_delta_mm:
                    return self._current_primary_id
        
        # 切换到新目标
        self._current_primary_id = closest_id
        return closest_id
    
    def get_current_primary_id(self) -> Optional[int]:
        """获取当前主目标 ID"""
        return self._current_primary_id
    
    def reset(self):
        """重置选择器状态"""
        self._current_primary_id = None
