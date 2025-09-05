"""
距离处理器模块
"""
import numpy as np
from collections import deque
from typing import Optional
from ..core.logger import logger
from ..core.constants import Constants


class DistanceProcessor:
    """距离处理器"""
    
    def __init__(self, window_size=None):
        self.window_size = window_size or Constants.SMOOTHING_WINDOW
        self.buffer = deque(maxlen=self.window_size)
        self.last_valid = None
        
    def add_measurement(self, distance: float) -> Optional[float]:
        """添加测量值并返回稳定的距离"""
        if distance is None:
            return self.last_valid
        
        self.buffer.append(distance)
        
        if len(self.buffer) >= 3:
            weights = np.exp(np.linspace(-1, 0, len(self.buffer)))
            weights = weights / weights.sum()
            stable_distance = np.average(list(self.buffer), weights=weights)
            self.last_valid = stable_distance
            return stable_distance
        
        self.last_valid = distance
        return distance
    
    def get_stability_score(self) -> float:
        """获取稳定性评分"""
        if len(self.buffer) < 3:
            return 0.0
        std = np.std(list(self.buffer))
        return max(0, 1 - (std / 0.02))
    
    def reset(self):
        """重置处理器"""
        self.buffer.clear()
        self.last_valid = None