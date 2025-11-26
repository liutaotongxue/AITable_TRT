"""
相机模块（TensorRT-Only 架构）

核心组件：
- TOFCameraManager: TOF 相机管理器
- CameraIntrinsicsManager: 相机内参管理
- ImageDataProcessor: 图像数据处理

历史遗留代码已清理（2025-01-19）：
- 已删除未使用的配置类（CameraSettings, FilterSettings 等）
- 已删除未使用的坐标转换工具（CoordinateConverter 等）
- 已删除未使用的点云工具（pointcloud_utils 等）
"""
from .tof_manager import TOFCameraManager
from .intrinsics import CameraIntrinsicsManager
from .image_processor import ImageDataProcessor

__all__ = [
    'TOFCameraManager',
    'CameraIntrinsicsManager',
    'ImageDataProcessor',
]