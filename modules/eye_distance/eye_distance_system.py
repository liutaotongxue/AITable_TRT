"""
眼距监测系统核心模块
"""
import time
import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional
from datetime import datetime
import cv2

from ..core import Constants, logger
from ..camera import ImageDataProcessor
from ..camera.depth_camera_interface import DepthCameraInterface
from ..detection import DistanceProcessor
from ..visualization import EnhancedVisualizer


class EyeDistanceSystem:
    """
    眼距监测系统 - 集成所有功能模块
    依赖 DepthCameraInterface 接口（接口驱动设计）
    """

    def __init__(self,
                 camera: DepthCameraInterface,
                 plane_model: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.25),
                 depth_range: Tuple[float, float] = (200, 1500)):
        """
        初始化眼距监测系统

        Args:
            camera: DepthCameraInterface 实例（必需）
            plane_model: 平面模型参数
            depth_range: 深度范围(毫米)

        Raises:
            TypeError: 如果 camera 未实现 DepthCameraInterface

        注意:
            眼睛位置由 Pose Tracker 提供（YOLO Pose 关键点），本模块仅负责：
            - 深度采样
            - 3D 坐标转换
            - 距离计算
        """
        # 验证接口类型
        if not isinstance(camera, DepthCameraInterface):
            raise TypeError(
                f"camera must implement DepthCameraInterface, got {type(camera).__name__}"
            )

        # 相机接口
        self.camera = camera

        # 相机参数初始化
        self.intrinsics_manager = camera.intrinsics_manager
        if self.intrinsics_manager and camera.camera_available:
            self.camera_params_rgb = self.intrinsics_manager.get_rgb_intrinsics_dict()
            self.camera_params_depth = self.intrinsics_manager.get_depth_intrinsics_dict()
            self.init_camera_params_from_sdk()
        else:
            # TOF相机是必需组件，不允许使用默认参数运行
            error_msg = (
                "TOF camera is not available - cannot initialize EyeDistanceSystem!\n"
                "原因：TOF相机是系统必需组件，不允许使用默认内参运行。\n"
                "使用默认内参会导致：\n"
                "- 距离计算完全错误\n"
                "- 姿态角度失真\n"
                "- 疲劳检测误判\n"
                "\n"
                "请确保TOF相机已连接并成功初始化。"
            )
            logger.error(error_msg)
            raise RuntimeError("EyeDistanceSystem requires a working TOF camera")

        # 平面参数
        self.a, self.b, self.c, self.d = plane_model
        self.plane_norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)

        # 深度范围
        self.min_depth, self.max_depth = depth_range

        # 距离处理器（平滑和稳定化）
        self.distance_processor = DistanceProcessor(Constants.SMOOTHING_WINDOW)

        # 从相机获取 SDK 常量，传给 ImageDataProcessor
        sdk_classes = camera.get_sdk_classes() if hasattr(camera, 'get_sdk_classes') else None
        self.image_processor = ImageDataProcessor(sdk_classes=sdk_classes)

        self.visualizer = EnhancedVisualizer()

        # 跟踪变量
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)

        # 分辨率管理
        if self.intrinsics_manager:
            self.rgb_width = self.intrinsics_manager.rgb_intrinsics['width']
            self.rgb_height = self.intrinsics_manager.rgb_intrinsics['height']
        else:
            self.rgb_width = 640
            self.rgb_height = 480

        # SDK 若返回 0x0 分辨率时使用安全默认值，避免后续除零
        if self.rgb_width <= 0 or self.rgb_height <= 0:
            logger.warning(
                f"Invalid intrinsics resolution {self.rgb_width}x{self.rgb_height}, "
                "falling back to 640x480"
            )
            self.rgb_width = 640
            self.rgb_height = 480
            if self.intrinsics_manager:
                # 同步更新当前内参分辨率，后续窗口计算不再得到 0
                self.intrinsics_manager.update_resolution(self.rgb_width, self.rgb_height)

        self.resolution_updated = False

        logger.info("Eye Distance System initialized (Pose-based eye detection)")
        logger.info(f"Initial resolution: {self.rgb_width}x{self.rgb_height}")

    def init_camera_params_from_sdk(self):
        """从 SDK 初始化相机参数"""
        # RGB相机参数
        self.rgb_fx = self.camera_params_rgb['fx']
        self.rgb_fy = self.camera_params_rgb['fy']
        self.rgb_cx = self.camera_params_rgb['cx']
        self.rgb_cy = self.camera_params_rgb['cy']
        
        # 深度相机参数
        self.depth_fx = self.camera_params_depth['fx']
        self.depth_fy = self.camera_params_depth['fy']
        self.depth_cx = self.camera_params_depth['cx']
        self.depth_cy = self.camera_params_depth['cy']
        self.depth_width = self.intrinsics_manager.depth_intrinsics['width']
        self.depth_height = self.intrinsics_manager.depth_intrinsics['height']
        
        # 立体标定参数
        stereo_params = self.intrinsics_manager.get_stereo_params_dict()
        self.R_depth_to_rgb = stereo_params['R']
        self.T_depth_to_rgb = stereo_params['T']
        
        logger.info("Camera parameters initialized from SDK")
    
    def init_default_camera_params(self):
        """初始化默认相机参数（当相机硬件不可用时）"""
        # RGB相机参数
        self.rgb_fx = self.camera_params_rgb['fx']
        self.rgb_fy = self.camera_params_rgb['fy']
        self.rgb_cx = self.camera_params_rgb['cx']
        self.rgb_cy = self.camera_params_rgb['cy']
        
        # 深度相机参数
        self.depth_fx = self.camera_params_depth['fx']
        self.depth_fy = self.camera_params_depth['fy']
        self.depth_cx = self.camera_params_depth['cx']
        self.depth_cy = self.camera_params_depth['cy']
        self.depth_width = 640
        self.depth_height = 480
        
        # 立体标定参数（使用单位矩阵作为默认值）
        self.R_depth_to_rgb = np.eye(3)
        self.T_depth_to_rgb = np.zeros((3, 1))
        
        logger.info("Using default camera parameters (hardware not available)")
    
    def update_camera_resolution(self, rgb_frame: np.ndarray):
        """动态更新相机分辨率"""
        if rgb_frame is None or self.resolution_updated:
            return
        
        new_height, new_width = rgb_frame.shape[:2]
        
        if new_width != self.rgb_width or new_height != self.rgb_height:
            logger.info(f"Updating resolution from {self.rgb_width}x{self.rgb_height} to {new_width}x{new_height}")
            
            if self.intrinsics_manager:
                # 使用SDK内参管理器更新分辨率
                self.intrinsics_manager.update_resolution(new_width, new_height)
                
                # 获取更新后的内参
                updated_rgb = self.intrinsics_manager.get_rgb_intrinsics_dict()
                self.rgb_fx = updated_rgb['fx']
                self.rgb_fy = updated_rgb['fy']
                self.rgb_cx = updated_rgb['cx']
                self.rgb_cy = updated_rgb['cy']
            
            self.rgb_width = new_width
            self.rgb_height = new_height
            self.resolution_updated = True
    
    def get_robust_depth(self, depth_map: np.ndarray, x: int, y: int) -> Optional[float]:
        """获取稳定深度值"""
        h, w = depth_map.shape
        
        # 快速路径：检查中心点
        center_depth = depth_map[y, x]
        if self.min_depth < center_depth < self.max_depth:
            return float(center_depth)
        
        # 多窗口采样
        for window_size in Constants.DEPTH_WINDOW_SIZES:
            half = window_size // 2
            
            x_start = max(0, x - half)
            x_end = min(w, x + half + 1)
            y_start = max(0, y - half)
            y_end = min(h, y + half + 1)
            
            window = depth_map[y_start:y_end, x_start:x_end]
            valid_depths = window[(window > self.min_depth) & (window < self.max_depth)]
            
            if len(valid_depths) >= window.size * Constants.MIN_VALID_DEPTH_RATIO:
                return float(np.median(valid_depths))
        
        return None
    
    def pixel_to_3d(self, x: int, y: int, depth_mm: float) -> Tuple[float, float, float]:
        """像素坐标转3D坐标"""
        z = depth_mm / 1000.0
        x_3d = (x - self.rgb_cx) * z / self.rgb_fx
        y_3d = -(y - self.rgb_cy) * z / self.rgb_fy
        return (x_3d, y_3d, z)
    
    def calculate_distance_to_plane(self, point_3d: Tuple[float, float, float]) -> float:
        """计算点到平面的距离"""
        x, y, z = point_3d
        numerator = abs(self.a * x + self.b * y + self.c * z + self.d)
        distance = numerator / self.plane_norm
        return distance
    
    def process_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        [DEPRECATED] 旧的帧处理方法，已废弃。

        现在使用 DetectionOrchestrator 的 _process_with_tracker_target() 方法，
        眼睛位置由 YOLO Pose 关键点提供，不再使用独立的人脸检测器。

        Returns:
            (None, no_detection_visualization): 始终返回无检测结果
        """
        logger.warning(
            "process_frame() 已废弃，请使用 DetectionOrchestrator 模式。"
            "眼睛位置现由 YOLO Pose 关键点提供。"
        )
        self.frame_count += 1
        return None, self.visualizer.draw_no_detection(rgb_frame.copy(), "DEPRECATED")
    
    def reset(self):
        """重置系统"""
        self.frame_count = 0
        self.processing_times.clear()
        self.resolution_updated = False
        self.distance_processor = DistanceProcessor(Constants.SMOOTHING_WINDOW)
