"""
检测模块 - TensorRT-only 架构 (Pose-Only)

注意：TRTFaceDetector 已废弃，现在使用 YOLO Pose 关键点作为眼睛位置来源。
"""
from .distance_processor import DistanceProcessor
from .trt_pose_detector import TRTPoseDetector
from .roi_manager import ROIManager

# TRTFaceDetector 已废弃（保留文件供参考，但不再导出）
# from .trt_face_detector import TRTFaceDetector

__all__ = [
    'DistanceProcessor',
    'TRTPoseDetector',
    'ROIManager',
]