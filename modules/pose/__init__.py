"""
Pose Detection Module (YOLO Pose Implementation)

Features:
- Human keypoint detection (shoulders, eyes)
- Posture angle calculation (forward lean, shoulder tilt, head roll)
- 3D coordinate conversion support (with TOF depth data)
- Quality-aware detection with fallback strategies

Implementation: YOLO Pose (migrated from MediaPipe, fully compatible interface)
"""
from .eye_shoulder_detector_medp_pose import (
    GetPose3dCoords,
    QUALITY_FLAG_KEY,
    QUALITY_FLAG_LEFT_EYE_FALLBACK,
    QUALITY_FLAG_RIGHT_EYE_FALLBACK,
)
from .get_pose_angles import CalculatePostureAngles
from .orientation_filter import HeadOrientationFilter

__all__ = [
    "GetPose3dCoords",
    "CalculatePostureAngles",
    "HeadOrientationFilter",
    # Quality flag constants (for quality-aware applications)
    "QUALITY_FLAG_KEY",
    "QUALITY_FLAG_LEFT_EYE_FALLBACK",
    "QUALITY_FLAG_RIGHT_EYE_FALLBACK",
]
