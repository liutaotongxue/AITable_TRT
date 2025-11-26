"""
检测模块 - TensorRT-only 架构
"""
from .distance_processor import DistanceProcessor
from .trt_face_detector import TRTFaceDetector
from .trt_pose_detector import TRTPoseDetector
from .roi_manager import ROIManager

__all__ = [
    'DistanceProcessor',
    'TRTFaceDetector',
    'TRTPoseDetector',
    'ROIManager',
]