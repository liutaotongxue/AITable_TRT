"""
TensorRT-Only 疲劳检测器 - 集成EAR和PERCLOS双重标准
使用 TensorRT FaceMesh 进行人脸关键点检测
包含卡内基梅隆研究所PERCLOS黄金标准实现
"""
import cv2
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple
from ..core.logger import logger

try:
    from .tensorrt_facemesh import TensorRTFaceMesh, create_facemesh
    TENSORRT_AVAILABLE = True
except ImportError:
    logger.error("TensorRT FaceMesh not available - required for TensorRT-Only architecture")
    TENSORRT_AVAILABLE = False
    raise ImportError(
        "TensorRT FaceMesh is required but not available.\n"
        "Ensure tensorrt and pycuda are installed."
    )


class TRTFatigueDetector:
    """
    TensorRT 疲劳检测器 - 集成EAR和PERCLOS算法
    使用 TensorRT FaceMesh 进行关键点检测
    """

    def __init__(self, model_path: Optional[str] = None, perclos_window=30, fps=8):
        """
        初始化检测器（TensorRT-Only）
        Args:
            model_path: TensorRT FaceMesh 引擎路径（可选，从 system_config.json 读取如果为 None）
            perclos_window: PERCLOS计算时间窗口（秒），默认30秒
            fps: 预期帧率，用于计算窗口大小

        Raises:
            FileNotFoundError: FaceMesh 引擎文件不存在
            RuntimeError: TensorRT 初始化失败
        """
        # 从配置文件获取模型路径（如果未提供）
        if model_path is None:
            from ..core.config_loader import get_config
            config = get_config()

            # 使用 resolve_model_path 解析模型路径（自动处理 primary/fallback）
            resolved_path = config.resolve_model_path("facemesh")

            if resolved_path is None:
                # 获取配置信息用于错误提示
                facemesh_config = config.models.get("facemesh")
                expected = facemesh_config.get("primary") if facemesh_config else "models/facemesh_fp16.engine"
                raise FileNotFoundError(
                    f"FaceMesh TensorRT engine not found.\n"
                    f"Expected: {expected}\n"
                    f"Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)"
                )

            model_path = str(resolved_path)
            logger.info(f"Using FaceMesh engine from config: {model_path}")

        # 初始化 TensorRT FaceMesh
        try:
            self.face_mesh = TensorRTFaceMesh(model_path=model_path)
            logger.info(f"[OK] TRTFatigueDetector initialized with TensorRT FaceMesh: {model_path}")
        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize TensorRT FaceMesh: {e}")
            raise RuntimeError(f"TensorRT FaceMesh initialization failed: {e}") from e

        self.enabled = True

        # 眼部关键点索引（MediaPipe 标准）
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # EAR参数
        self.EAR_THRESHOLD = 0.22  # 眼睛闭合阈值（0.22 平衡敏感度和误报）
        self.EAR_CONSEC_FRAMES = 3  # 连续帧数阈值（防止快速眨眼误判）
        self.EAR_CLOSE_TIME_MS = 300  # 基于时间的闭眼阈值（毫秒），优先于帧数

        # PERCLOS参数（使用时间戳而非固定帧率）
        self.PERCLOS_WINDOW = perclos_window  # 时间窗口（秒）
        self.FPS = fps  # 预期帧率（仅用于初始化缓冲区大小）
        self.WINDOW_SIZE = self.PERCLOS_WINDOW * fps  # 窗口帧数（初始估计）

        # 时间戳追踪（用于基于时间的 PERCLOS 计算）
        self._eye_close_start_ms: Optional[float] = None  # 闭眼开始时间

        # PERCLOS疲劳等级阈值（卡内基梅隆标准）
        self.PERCLOS_NORMAL = 20      # 正常阈值 < 20%
        self.PERCLOS_MILD = 40        # 轻度疲劳 20%-40%
        # 严重疲劳 > 40%

        # 状态追踪
        self.frame_count = 0
        self.drowsy_frames = 0
        self.eye_closed_frames = 0

        # PERCLOS计算缓冲区（使用滑动窗口）
        self.eye_state_buffer = deque(maxlen=self.WINDOW_SIZE)
        self.timestamp_buffer = deque(maxlen=self.WINDOW_SIZE)

        # 统计数据
        self.perclos_history = deque(maxlen=100)  # 保存历史PERCLOS值
        self.ear_history = deque(maxlen=100)       # 保存历史EAR值

        # 初始化时间
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        logger.info(f"TRT Fatigue detector initialized: PERCLOS window={perclos_window}s")
        logger.info(f"PERCLOS阈值: Normal<{self.PERCLOS_NORMAL}%, Mild:{self.PERCLOS_NORMAL}-{self.PERCLOS_MILD}%, Severe>{self.PERCLOS_MILD}%")

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        计算眼睛纵横比（EAR）
        Args:
            eye_points: 眼部关键点坐标 (N, 2)
        Returns:
            EAR值
        """
        if len(eye_points) < 6:
            return 0.0

        # 计算垂直距离
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])

        # 计算水平距离
        C = distance.euclidean(eye_points[0], eye_points[3])

        # 避免除零
        if C < 1e-6:
            return 0.0

        # EAR公式
        ear = (A + B) / (2.0 * C)
        return ear

    def validate_frame(self, frame: np.ndarray, is_cropped_face: bool = False) -> bool:
        """验证输入帧是否有效；is_cropped_face 兼容上层接口，当前逻辑忽略"""
        if frame is None or frame.size == 0:
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        return True

    def detect_fatigue(self, rgb_frame: np.ndarray, face_bbox: dict = None, is_cropped_face: bool = False) -> Dict:
        """
        检测疲劳状态

        Args:
            rgb_frame: RGB 格式图像 (H, W, 3)
            face_bbox: 人脸边界框 {'x1': int, 'y1': int, 'x2': int, 'y2': int}
                      如果提供，将裁剪人脸区域后再处理，提高关键点精度

        Returns:
            疲劳检测结果字典
        """
        if not self.validate_frame(rgb_frame, is_cropped_face=is_cropped_face):
            return self._get_default_results()

        try:
            frame_h, frame_w = rgb_frame.shape[:2]

            # 根据 is_cropped_face 决定是否需要裁剪
            if is_cropped_face:
                # 已经是裁剪好的人脸图像，直接使用
                face_roi = rgb_frame
            else:
                # 必须提供人脸框才进行疲劳检测（避免全帧误判）
                if face_bbox is None:
                    return None

                # 获取边界框坐标
                x1, y1 = int(face_bbox['x1']), int(face_bbox['y1'])
                x2, y2 = int(face_bbox['x2']), int(face_bbox['y2'])

                # 扩展边界框（上下左右各扩展 25%），确保包含完整人脸
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                expand_w = int(bbox_w * 0.25)
                expand_h = int(bbox_h * 0.25)

                roi_x1 = max(0, x1 - expand_w)
                roi_y1 = max(0, y1 - expand_h)
                roi_x2 = min(frame_w, x2 + expand_w)
                roi_y2 = min(frame_h, y2 + expand_h)

                # 裁剪人脸 ROI
                face_roi = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if face_roi.size == 0:
                logger.warning("Face ROI is empty")
                return None

            roi_h, roi_w = face_roi.shape[:2]

            # 使用 TensorRT FaceMesh 处理裁剪后的人脸区域
            results = self.face_mesh.process(face_roi)

            if not results.multi_face_landmarks:
                return self._get_default_results()

            # 获取第一个人脸的关键点
            face_landmarks = results.multi_face_landmarks[0]

            # 提取眼部关键点（坐标相对于 ROI）
            left_eye_points = []
            right_eye_points = []

            for idx in self.LEFT_EYE:
                lm = face_landmarks.landmark[idx]
                # 关键点坐标是相对于 ROI 的归一化坐标
                left_eye_points.append([lm.x * roi_w, lm.y * roi_h])

            for idx in self.RIGHT_EYE:
                lm = face_landmarks.landmark[idx]
                right_eye_points.append([lm.x * roi_w, lm.y * roi_h])

            left_eye_points = np.array(left_eye_points)
            right_eye_points = np.array(right_eye_points)

            # 计算 EAR
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # 保存 EAR 历史
            self.ear_history.append(avg_ear)

            # 判断眼睛是否低于阈值（瞬时）
            is_below_threshold = avg_ear < self.EAR_THRESHOLD
            current_time = time.time()
            current_time_ms = current_time * 1000

            # 基于时间的闭眼判断（避免快速眨眼误判）
            if is_below_threshold:
                if self._eye_close_start_ms is None:
                    self._eye_close_start_ms = current_time_ms
                    self.eye_closed_frames = 1
                else:
                    self.eye_closed_frames += 1

                # 只有闭眼持续超过阈值时间才计为真正闭眼
                close_duration_ms = current_time_ms - self._eye_close_start_ms
                is_closed = (close_duration_ms >= self.EAR_CLOSE_TIME_MS or
                            self.eye_closed_frames >= self.EAR_CONSEC_FRAMES)
            else:
                self._eye_close_start_ms = None
                self.eye_closed_frames = 0
                is_closed = False

            # 更新 PERCLOS 缓冲区（使用时间戳计算实际窗口）
            self.eye_state_buffer.append(1 if is_closed else 0)
            self.timestamp_buffer.append(current_time)

            # 计算 PERCLOS
            perclos = self._calculate_perclos()
            self.perclos_history.append(perclos)

            # 判断疲劳等级（基于 PERCLOS）
            if perclos < self.PERCLOS_NORMAL:
                fatigue_level = "Normal"
                perclos_status = "Normal"
            elif perclos < self.PERCLOS_MILD:
                fatigue_level = "Mild fatigue"
                perclos_status = "Mild fatigue"
            else:
                fatigue_level = "Severe fatigue"
                perclos_status = "Severe fatigue"

            # 检查 PERCLOS 是否已经收集足够数据（基于时间而非帧数）
            window_duration = self._get_window_duration()
            perclos_valid = window_duration >= (self.PERCLOS_WINDOW * 0.5)  # 至少50%窗口时间

            # 构建结果
            fatigue_results = {
                'enabled': True,
                'ear': avg_ear,
                'ear_avg': avg_ear,  # 兼容可视化层
                'left_ear': left_ear,
                'right_ear': right_ear,
                'eye_closed': is_closed,
                'eye_closed_frames': self.eye_closed_frames,
                'drowsy_alert': is_closed and self.eye_closed_frames >= self.EAR_CONSEC_FRAMES,
                'perclos': perclos,
                'perclos_valid': perclos_valid,
                'perclos_status': perclos_status,
                'fatigue_level': fatigue_level,
                'fatigue_level_code': 0 if fatigue_level == "Normal" else (1 if fatigue_level == "Mild fatigue" else 2),
                'fatigue_color': (0, 255, 0) if fatigue_level == "Normal" else ((0, 255, 255) if fatigue_level == "Mild fatigue" else (0, 0, 255)),
                'buffer_size': len(self.eye_state_buffer),
                'window_size': self.WINDOW_SIZE,
                'window_progress': min(100.0, (window_duration / self.PERCLOS_WINDOW) * 100),
                'fps': 1.0 / self.global_frame_interval if hasattr(self, 'global_frame_interval') and self.global_frame_interval > 0 else self.FPS
            }

            self.frame_count += 1
            return fatigue_results

        except Exception as e:
            logger.error(f"Fatigue detection error: {e}")
            return self._get_default_results()

    def _calculate_perclos(self) -> float:
        """
        计算 PERCLOS（眼睛闭合百分比）- 基于时间窗口

        使用时间戳计算实际窗口内的闭眼比例，而非假设固定帧率。
        只考虑 PERCLOS_WINDOW 秒内的数据。
        """
        if len(self.eye_state_buffer) == 0 or len(self.timestamp_buffer) == 0:
            return 0.0

        current_time = time.time()
        window_start = current_time - self.PERCLOS_WINDOW

        # 找到窗口内的数据
        closed_count = 0
        total_count = 0

        for state, ts in zip(self.eye_state_buffer, self.timestamp_buffer):
            if ts >= window_start:
                total_count += 1
                if state == 1:
                    closed_count += 1

        if total_count == 0:
            return 0.0

        perclos = (closed_count / total_count) * 100.0
        return perclos

    def _get_window_duration(self) -> float:
        """获取当前缓冲区覆盖的实际时间（秒）"""
        if len(self.timestamp_buffer) < 2:
            return 0.0
        return self.timestamp_buffer[-1] - self.timestamp_buffer[0]

    def _get_default_results(self) -> Dict:
        """返回默认结果（无人脸检测到）"""
        return {
            'enabled': False,
            'ear': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'eye_closed': False,
            'eye_closed_frames': 0,
            'perclos': 0.0,
            'perclos_valid': False,
            'perclos_status': 'N/A',
            'fatigue_level': 'N/A',
            'buffer_size': 0,
            'window_size': self.WINDOW_SIZE
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'frame_count': self.frame_count,
            'avg_ear': np.mean(self.ear_history) if self.ear_history else 0.0,
            'avg_perclos': np.mean(self.perclos_history) if self.perclos_history else 0.0,
            'buffer_fill': len(self.eye_state_buffer) / self.WINDOW_SIZE * 100
        }

    def close(self):
        """释放资源"""
        if hasattr(self.face_mesh, 'close'):
            self.face_mesh.close()
        logger.info("TRTFatigueDetector closed")


# 向后兼容别名
FatigueDetector = TRTFatigueDetector
