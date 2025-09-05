"""
相机模块
"""
from .tof_manager import TOFCameraManager
from .intrinsics import CameraIntrinsicsManager
from .image_processor import ImageDataProcessor
from .coordinate_converter import CoordinateConverter
from .config import (
    CameraSettings, 
    FilterSettings, 
    ProcessingSettings, 
    ParametersSetting,
    ParamatersSetting,  # 保持拼写错误的兼容性
    PS
)

__all__ = [
    'TOFCameraManager', 
    'CameraIntrinsicsManager', 
    'ImageDataProcessor',
    'CoordinateConverter',
    'CameraSettings',
    'FilterSettings',
    'ProcessingSettings',
    'ParametersSetting',
    'ParamatersSetting',
    'PS'
]