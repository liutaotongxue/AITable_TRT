"""
SimpleTracker 模块 - 基于 Pose 的统一人员跟踪

核心功能：
- 单 YOLO-Pose 检测器作为唯一人员来源
- IoU 轨迹关联 + 深度选人
- 从 keypoints 推导 face_bbox

使用方法：
    from modules.tracking import SimpleTracker
    
    tracker = SimpleTracker(
        depth_range_mm=(200, 1500),
        switch_depth_delta_mm=100,
        min_keep_frames=15,
    )
    
    # 在主循环中
    tracks = tracker.update(pose_detections, depth_frame)
    target = tracker.get_primary_target()
    
    if target:
        face_bbox = target.face_bbox  # 从 keypoints 推导
        keypoints = target.keypoints
        depth_mm = target.depth_mm
"""

from .simple_tracker import SimpleTracker, TrackedPerson
from .face_deriver import FaceDeriver
from .depth_selector import DepthSelector

__all__ = [
    'SimpleTracker',
    'TrackedPerson', 
    'FaceDeriver',
    'DepthSelector',
]
