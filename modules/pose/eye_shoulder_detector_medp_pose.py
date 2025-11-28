"""
Human body pose keypoint detection module (TensorRT YOLO Pose implementation)
Migrated from MediaPipe to YOLO Pose, maintaining interface compatibility
TensorRT-Only version: requires .engine files
"""
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

from ..core.logger import logger
from ..detection.body_key_point_detection import BodyKeyPointDetector


# YOLO Pose COCO 17-keypoint indices
# Reference: https://docs.ultralytics.com/tasks/pose/
YOLO_KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Project-required keypoints (maintain compatibility with original MediaPipe version)
REQUIRED_KEYPOINTS = ["left_shoulder", "right_shoulder"]

# Keypoint confidence threshold (filter out low-confidence detections)
DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

# Quality flag keys (centralized to avoid hardcoding)
QUALITY_FLAG_KEY = "_quality_flags"
QUALITY_FLAG_LEFT_EYE_FALLBACK = "left_eye_fallback"
QUALITY_FLAG_RIGHT_EYE_FALLBACK = "right_eye_fallback"


class GetPose3dCoords:
    """
    Pose keypoint detector (YOLO Pose implementation)

    Features:
    - Detect human keypoints (left/right shoulders, left/right eyes)
    - Provide same interface as original MediaPipe version
    - Support 3D coordinate conversion (with TOF depth data)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        keypoint_confidence_threshold: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        external_detector=None,  # 可选：外部传入的检测器（避免重复加载模型）
    ):
        """
        Initialize pose detector (TensorRT-Only)

        Args:
            model_path: TensorRT YOLO Pose engine path (.engine file)
                       If not provided, reads from system_config.json
            confidence_threshold: Detection confidence threshold [0.0, 1.0]
            keypoint_confidence_threshold: Individual keypoint confidence threshold
            external_detector: 外部传入的检测器实例（需有 detect_keypoints 方法）
                              如果传入，则跳过内部创建，直接使用外部检测器
        """
        # 如果传入外部检测器，直接使用，不创建新的
        if external_detector is not None:
            self.detector = external_detector
            logger.info("GetPose3dCoords: 使用外部传入的检测器（共享模型）")
        else:
            # 内部创建检测器
            if model_path is None:
                # Read from config (TensorRT-Only architecture)
                from ..core.config_loader import get_config
                config = get_config()

                # 使用 resolve_model_path 解析模型路径（自动处理 primary/fallback）
                resolved_path = config.resolve_model_path("yolo_pose")

                if resolved_path is None:
                    # 获取配置信息用于错误提示
                    pose_config = config.models.get("yolo_pose")
                    expected = pose_config.get("primary") if pose_config else "models/yolov8n-pose_fp16.engine"
                    raise FileNotFoundError(
                        f"YOLO Pose TensorRT engine not found.\n"
                        f"Expected: {expected}\n"
                        f"Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)"
                    )

                model_path = str(resolved_path)

            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"YOLO Pose model file not found: {model_path}\n"
                    f"Please provide a valid .engine file"
                )

            try:
                self.detector = BodyKeyPointDetector(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                )
                logger.info(f"TensorRT YOLO Pose detector initialized successfully: {model_path}")
            except Exception as e:
                logger.error(f"TensorRT YOLO Pose detector initialization failed: {e}")
                raise RuntimeError(f"Pose detector initialization failed: {e}") from e

        # Keypoint confidence threshold
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)

        # Statistics - 基础统计
        self._detection_count = 0
        self._failed_count = 0

        # Statistics - 详细失败原因统计
        self._failure_stats = {
            "no_detections": 0,        # 未检测到人体
            "missing_shoulders": 0,     # 缺少必需的肩部关键点
            "low_confidence": 0,        # 关键点置信度过低
            "out_of_bounds": 0,         # 关键点超出图像边界
        }

    def detect_posture_with_mediapipe(
        self, rgb_image
    ) -> Optional[Dict[str, Union[Tuple[int, int], Dict[str, bool]]]]:
        """
        Detect pose keypoints (compatible with original MediaPipe interface)

        Note:
        - Method name kept as detect_posture_with_mediapipe for compatibility
        - Actually uses YOLO Pose detector implementation
        - Returns merged dictionary containing both keypoint coordinates and quality metadata

        Args:
            rgb_image: RGB/BGR image (numpy array), shape=(H, W, 3)

        Returns:
            Optional[Dict[str, Union[Tuple[int, int], Dict[str, bool]]]]:
            Dictionary containing keypoint coordinates and optional quality metadata.
            Returns None if no person detected or missing required keypoints.

            Structure when detection succeeds:
            {
                'left_shoulder': (x, y),             # Tuple[int, int] - Required
                'right_shoulder': (x, y),            # Tuple[int, int] - Required
                'left_eye_center': (x, y),           # Tuple[int, int] - Optional
                'right_eye_center': (x, y),          # Tuple[int, int] - Optional
                '_quality_flags': {                  # Dict[str, bool] - Optional metadata
                    'left_eye_fallback': bool,       #   True if using ear as left eye
                    'right_eye_fallback': bool,      #   True if using ear as right eye
                }
            }

            Quality Flags Explanation:
            - The '_quality_flags' key uses underscore prefix (metadata convention)
            - Only present if fallback was used (ear substituted for eye)
            - Allows callers to detect degraded quality detections
            - Older code can safely ignore this field (backwards compatible)

            Implementation Note:
            - This method merges keypoint coordinates and quality metadata into a single dict
            - Alternative design would return Tuple[Dict[coordinates], Dict[flags]]
            - Current design chosen for backwards compatibility with existing callers
        """
        if rgb_image is None or rgb_image.size == 0:
            logger.warning("Pose detection input image is empty")
            return None

        self._detection_count += 1

        # Call YOLO Pose detection
        detection_result = self.detector.detect_keypoints(rgb_image)

        if detection_result is None or not detection_result.get("detections"):
            # No person detected
            self._failed_count += 1
            self._failure_stats["no_detections"] += 1
            return None

        # Select best detection (already sorted by confidence in _select_best_detection)
        detections = detection_result["detections"]
        if not detections:
            self._failed_count += 1
            return None

        # Sort by detection confidence and select the highest one
        best_detection = self._select_best_detection(detections)
        if best_detection is None:
            self._failed_count += 1
            return None

        keypoints = best_detection.get("keypoints", [])
        if not keypoints:
            self._failed_count += 1
            return None

        # Build keypoint mapping (index -> keypoint data)
        kp_map = {kp["index"]: kp for kp in keypoints}

        # Extract required keypoints (with quality flags)
        key_points_2d, quality_flags = self._extract_required_keypoints(kp_map, rgb_image.shape)

        # Validate required keypoints exist
        if not self._validate_keypoints(key_points_2d):
            self._failed_count += 1
            self._failure_stats["missing_shoulders"] += 1
            return None

        # Add quality flags for degraded detections (currently unused after removing ear fallback)
        if quality_flags:
            key_points_2d[QUALITY_FLAG_KEY] = quality_flags

        # Periodic statistics output
        if self._detection_count % 100 == 0:
            success_rate = (
                (self._detection_count - self._failed_count) / self._detection_count * 100
            )
            logger.info(
                f"Pose detection stats: total={self._detection_count}, "
                f"success={self._detection_count - self._failed_count}, "
                f"failed={self._failed_count}, "
                f"success_rate={success_rate:.1f}%"
            )

        return key_points_2d

    def _select_best_detection(self, detections: list) -> Optional[Dict]:
        """
        Select best detection from multiple detections

        Strategy:
        1. Sort by detection confidence (primary)
        2. Fallback: sort by bounding box area (when confidence missing)
        3. Future: Could add keypoint quality scoring

        Args:
            detections: List of detection results

        Returns:
            Detection with highest confidence, or None if all invalid
        """
        if not detections:
            return None

        # Check if any detection has confidence field
        has_confidence = any(
            d.get("confidence") is not None and d.get("confidence") > 0
            for d in detections
        )

        if has_confidence:
            # Primary strategy: Sort by detection confidence (descending)
            sorted_detections = sorted(
                detections,
                key=lambda d: d.get("confidence", 0.0),
                reverse=True
            )
        else:
            # Fallback strategy: Sort by bounding box area (larger = closer/more prominent)
            # Assumes TensorRT export or old Ultralytics version without confidence
            logger.warning(
                "Detection confidence unavailable, using bounding box area as fallback"
            )
            sorted_detections = sorted(
                detections,
                key=lambda d: self._calculate_bbox_area(d),
                reverse=True
            )

        # Return highest confidence (or largest bbox) detection
        return sorted_detections[0]

    @staticmethod
    def _calculate_bbox_area(detection: Dict) -> float:
        """
        Calculate bounding box area from detection result.

        Args:
            detection: Detection dictionary with 'bbox' key

        Returns:
            float: Area of bounding box (width * height), or 0.0 if bbox missing
        """
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) < 4:
            return 0.0

        # bbox format: [x1, y1, x2, y2] or [x, y, w, h]
        # Assume [x1, y1, x2, y2] format from YOLO
        x1, y1, x2, y2 = bbox[:4]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height

    def _extract_required_keypoints(
        self, kp_map: Dict, image_shape: Tuple
    ) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, bool]]:
        """
        Extract required keypoints from YOLO detection result

        Args:
            kp_map: YOLO keypoint mapping {index: keypoint data}
            image_shape: Image dimensions (H, W, C)

        Returns:
            Tuple[Dict[str, Tuple[int, int]], Dict[str, bool]]:
            - key_points_2d: Dictionary mapping keypoint names to (x, y) coordinates
                {
                    'left_shoulder': (x, y),
                    'right_shoulder': (x, y),
                    'left_eye_center': (x, y),  # Optional
                    'right_eye_center': (x, y), # Optional
                }
            - quality_flags: Dictionary marking degraded/fallback detections
                {
                    'left_eye_fallback': True,  # Present only if fallback used
                    'right_eye_fallback': True, # Present only if fallback used
                }
                Empty dict if no fallbacks were needed.
        """
        key_points_2d = {}
        quality_flags = {}
        h, w = image_shape[:2]

        # Left shoulder (YOLO index 5) - REQUIRED
        left_shoulder_idx = YOLO_KEYPOINT_INDICES["left_shoulder"]
        if left_shoulder_idx in kp_map:
            kp = kp_map[left_shoulder_idx]
            if self._is_keypoint_valid(kp, w, h):
                x, y = int(kp["x"]), int(kp["y"])
                key_points_2d["left_shoulder"] = (x, y)

        # Right shoulder (YOLO index 6) - REQUIRED
        right_shoulder_idx = YOLO_KEYPOINT_INDICES["right_shoulder"]
        if right_shoulder_idx in kp_map:
            kp = kp_map[right_shoulder_idx]
            if self._is_keypoint_valid(kp, w, h):
                x, y = int(kp["x"]), int(kp["y"])
                key_points_2d["right_shoulder"] = (x, y)

        # Left eye (YOLO index 1) - OPTIONAL
        left_eye_idx = YOLO_KEYPOINT_INDICES["left_eye"]
        if left_eye_idx in kp_map:
            kp = kp_map[left_eye_idx]
            if self._is_keypoint_valid(kp, w, h):
                x, y = int(kp["x"]), int(kp["y"])
                key_points_2d["left_eye_center"] = (x, y)

        # Right eye (YOLO index 2) - OPTIONAL
        right_eye_idx = YOLO_KEYPOINT_INDICES["right_eye"]
        if right_eye_idx in kp_map:
            kp = kp_map[right_eye_idx]
            if self._is_keypoint_valid(kp, w, h):
                x, y = int(kp["x"]), int(kp["y"])
                key_points_2d["right_eye_center"] = (x, y)

        # Nose (YOLO index 0) - OPTIONAL (used for head forward angle when eyes missing)
        nose_idx = YOLO_KEYPOINT_INDICES["nose"]
        if nose_idx in kp_map:
            kp = kp_map[nose_idx]
            if self._is_keypoint_valid(kp, w, h):
                x, y = int(kp["x"]), int(kp["y"])
                key_points_2d["nose"] = (x, y)

        # 耳朵fallback已移除，改用鼻子补偿（在角度计算层面处理）

        return key_points_2d, quality_flags

    def _is_keypoint_valid(self, kp: Dict, width: int, height: int, track_stats: bool = True) -> bool:
        """
        Check if keypoint is valid (within image bounds and above confidence threshold)

        Args:
            kp: Keypoint data dictionary
            width: Image width
            height: Image height
            track_stats: Whether to track failure statistics (default: True)

        Returns:
            True if valid, False otherwise
        """
        # Check coordinates within bounds first (mandatory check)
        x, y = kp.get("x", -1), kp.get("y", -1)
        if not (0 <= x < width and 0 <= y < height):
            if track_stats:
                self._failure_stats["out_of_bounds"] += 1
            return False

        # Check confidence if available
        # Note: Some Ultralytics versions don't return per-keypoint confidence
        # If confidence field is missing or None, skip confidence check
        confidence = kp.get("confidence")
        if confidence is not None:
            # Only apply threshold if confidence is available
            if confidence < self.keypoint_confidence_threshold:
                if track_stats:
                    self._failure_stats["low_confidence"] += 1
                return False

        return True

    @staticmethod
    def _validate_keypoints(key_points_2d: Dict) -> bool:
        """
        Validate required keypoints exist

        Args:
            key_points_2d: Keypoint dictionary

        Returns:
            True if all required keypoints present, False otherwise
        """
        # At least need left and right shoulders to calculate pose angles
        for required_key in REQUIRED_KEYPOINTS:
            if required_key not in key_points_2d:
                return False
        return True

    def set_confidence_threshold(self, threshold: float):
        """
        Update detection confidence threshold

        Args:
            threshold: New confidence threshold [0.0, 1.0]
        """
        self.detector.set_confidence_threshold(threshold)
        logger.info(f"Pose detection confidence threshold updated: {threshold}")

    def set_keypoint_confidence_threshold(self, threshold: float):
        """
        Update keypoint confidence threshold

        Args:
            threshold: New keypoint confidence threshold [0.0, 1.0]
        """
        self.keypoint_confidence_threshold = float(max(0.0, min(1.0, threshold)))
        logger.info(f"Keypoint confidence threshold updated: {threshold}")

    def warmup(self, image_shape: Tuple[int, int, int] = (640, 640, 3), runs: int = 1):
        """
        Warmup model to reduce first-frame latency

        Args:
            image_shape: Warmup image dimensions (H, W, C)
            runs: Number of warmup runs
        """
        logger.info("Starting pose detector warmup...")
        self.detector.warmup(image_shape=image_shape, runs=runs)
        logger.info("Pose detector warmup completed")

    def get_model_info(self) -> dict:
        """
        Get model status information

        Returns:
            Dictionary containing model configuration and performance statistics
        """
        info = self.detector.get_model_info()
        info["detection_count"] = self._detection_count
        info["failed_count"] = self._failed_count
        info["keypoint_confidence_threshold"] = self.keypoint_confidence_threshold
        if self._detection_count > 0:
            info["success_rate"] = (
                (self._detection_count - self._failed_count) / self._detection_count
            )
        return info

    def get_failure_statistics(self) -> dict:
        """
        获取详细的失败统计信息（用于诊断和优化）

        Returns:
            dict: 详细统计信息 {
                "total_detections": int,
                "successful_detections": int,
                "failed_detections": int,
                "success_rate": float,
                "failure_breakdown": {
                    "no_detections": int,
                    "missing_shoulders": int,
                    "low_confidence": int,
                    "out_of_bounds": int,
                }
            }

        Example:
            >>> detector = GetPose3dCoords()
            >>> # ... 处理若干帧 ...
            >>> stats = detector.get_failure_statistics()
            >>> print(f"检测成功率: {stats['success_rate']:.1f}%")
            >>> if stats['failure_breakdown']['no_detections'] > 100:
            >>>     print("警告: 大量帧未检测到人体，检查相机位置/光照")
        """
        successful = self._detection_count - self._failed_count
        stats = {
            "total_detections": self._detection_count,
            "successful_detections": successful,
            "failed_detections": self._failed_count,
            "success_rate": (successful / self._detection_count * 100) if self._detection_count > 0 else 0.0,
            "failure_breakdown": self._failure_stats.copy(),
        }
        return stats

    def reset_stats(self):
        """Reset all statistics"""
        self._detection_count = 0
        self._failed_count = 0
        self._failure_stats = {
            "no_detections": 0,
            "missing_shoulders": 0,
            "low_confidence": 0,
            "out_of_bounds": 0,
        }
        logger.info("Pose detector statistics reset")

    @staticmethod
    def filter_keypoints_only(
        detection_result: Optional[Dict[str, Union[Tuple[int, int], Dict[str, bool]]]]
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Filter out metadata from detection result, returning only keypoint coordinates.

        This helper method allows legacy code that directly iterates over detection
        results (assuming all values are (x, y) tuples) to work correctly.

        Args:
            detection_result: Detection result from detect_posture_with_mediapipe()

        Returns:
            Optional[Dict[str, Tuple[int, int]]]: Dictionary containing only keypoint
            coordinates (no metadata). Returns None if input is None.

        Example:
            # Legacy code that assumes all values are tuples
            result = detector.detect_posture_with_mediapipe(image)
            keypoints_only = detector.filter_keypoints_only(result)
            if keypoints_only:
                for name, (x, y) in keypoints_only.items():
                    # Safe to unpack - guaranteed to be tuples
                    draw_point(image, x, y)
        """
        if detection_result is None:
            return None

        # Filter out metadata keys (those starting with underscore)
        return {
            k: v for k, v in detection_result.items()
            if not k.startswith("_") and isinstance(v, tuple)
        }

    @staticmethod
    def extract_quality_flags(
        detection_result: Optional[Dict[str, Union[Tuple[int, int], Dict[str, bool]]]]
    ) -> Dict[str, bool]:
        """
        Extract quality flags from detection result.

        Args:
            detection_result: Detection result from detect_posture_with_mediapipe()

        Returns:
            Dict[str, bool]: Quality flags dictionary. Returns empty dict if no
            quality flags present or if input is None.

        Example:
            result = detector.detect_posture_with_mediapipe(image)
            flags = detector.extract_quality_flags(result)

            if flags.get(QUALITY_FLAG_LEFT_EYE_FALLBACK):
                logger.warning("Left eye using ear fallback")
            if flags.get(QUALITY_FLAG_RIGHT_EYE_FALLBACK):
                logger.warning("Right eye using ear fallback")
        """
        if detection_result is None:
            return {}

        return detection_result.get(QUALITY_FLAG_KEY, {})

    # YOLO Pose method alias (preferred name for TensorRT YOLO Pose backend)
    # PoseEngine calls this method; it delegates to detect_posture_with_mediapipe
    detect_posture_with_yolo = detect_posture_with_mediapipe
