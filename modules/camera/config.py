"""
相机模块配置文件
"""
from typing import List
import numpy as np


class CameraSettings:
    """相机设置参数"""
    # 分辨率设置
    RGB_RESOLUTION = (1280, 1024)
    DEPTH_RESOLUTION = (1280, 1024)
    
    # 深度范围设置（毫米）
    MIN_VALID_DEPTH = 200
    MAX_VALID_DEPTH = 1500
    
    # 滤波参数
    SPATIAL_FILTER_SIZE = (5, 5)
    SPATIAL_FILTER_SIGMA = 1.0
    TEMPORAL_FILTER_WINDOW = 10
    
    # 深度采样参数
    DEPTH_NEIGHBOR_SIZE = 5  # 深度值邻域采样大小


class FilterSettings:
    """滤波器设置"""
    # 时域滤波
    TEMPORAL_HISTORY_SIZE = 10
    TEMPORAL_FILTER_METHOD = 'median'  # 'median' or 'mean'
    
    # 空域滤波
    SPATIAL_FILTER_ENABLED = True
    SPATIAL_KERNEL_SIZE = 5
    SPATIAL_SIGMA = 1.0
    
    # 深度滤波阈值
    DEPTH_STD_THRESHOLD = 10  # 深度标准差阈值（mm）
    MIN_VALID_DEPTH_RATIO = 0.3  # 最小有效深度比例


class ProcessingSettings:
    """处理参数设置"""
    # 3D坐标转换
    COORDINATE_UNIT = 'mm'  # 'mm' or 'm'
    
    # 深度采样窗口
    DEPTH_WINDOW_SIZES = [3, 5, 7]
    
    # 关键点处理
    KEYPOINT_DEPTH_NEIGHBOR = 5
    
    # 畸变校正
    ENABLE_UNDISTORTION = True
    UNDISTORTION_ALPHA = 1.0  # 0-1之间，控制视野保留程度


class ParametersSetting:
    """全局参数设置（兼容旧代码）"""
    # 深度历史缓存（用于时域滤波）
    depth_history: List[np.ndarray] = []
    
    # 相机参数缓存
    camera_intrinsics = None
    camera_distortion = None
    
    @classmethod
    def reset_history(cls):
        """重置深度历史"""
        cls.depth_history = []
    
    @classmethod
    def get_temporal_window_size(cls):
        """获取时域窗口大小"""
        return FilterSettings.TEMPORAL_HISTORY_SIZE


# 兼容旧代码的别名
ParamatersSetting = ParametersSetting  # 保持兼容性（注意原代码有拼写错误）
PS = ParametersSetting  # 简写别名