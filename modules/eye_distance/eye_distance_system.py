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
from ..camera import TOFCameraManager, ImageDataProcessor
from ..detection import SimpleFaceDetector, DistanceProcessor
from ..visualization import EnhancedVisualizer


class EyeDistanceSystem:
    """眼距监测系统 - 集成所有功能模块"""
    
    def __init__(self, 
                 camera_manager: TOFCameraManager = None,
                 plane_model: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.25),
                 model_path: str = 'yolov8n-face.pt',
                 depth_range: Tuple[float, float] = (200, 1500)):
        """
        初始化眼距监测系统
        
        Args:
            camera_manager: TOF相机管理器
            plane_model: 平面模型参数
            model_path: YOLO模型路径
            depth_range: 深度范围(毫米)
        """
        # 相机管理器
        self.camera_manager = camera_manager
        self.intrinsics_manager = None
        if camera_manager:
            self.intrinsics_manager = camera_manager.intrinsics_manager
            if self.intrinsics_manager and camera_manager.camera_available:
                self.camera_params_rgb = self.intrinsics_manager.get_rgb_intrinsics_dict()
                self.camera_params_depth = self.intrinsics_manager.get_depth_intrinsics_dict()
                self.init_camera_params_from_sdk()
            else:
                # 使用默认参数当相机不可用时
                self.camera_params_rgb = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
                self.camera_params_depth = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
                self.init_default_camera_params()
        
        # 平面参数
        self.a, self.b, self.c, self.d = plane_model
        self.plane_norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        
        # 深度范围
        self.min_depth, self.max_depth = depth_range
        
        # 初始化组件
        self.face_detector = SimpleFaceDetector(model_path, Constants.FACE_CONFIDENCE_THRESHOLD)
        self.distance_processor = DistanceProcessor(Constants.SMOOTHING_WINDOW)
        self.image_processor = ImageDataProcessor()
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
        self.resolution_updated = False
        
        logger.info("Eye Distance System initialized")
        logger.info(f"Face detector: YOLO model loaded")
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
        """处理单帧"""
        self.frame_count += 1
        start_time = time.time()
        
        # 动态更新相机分辨率
        self.update_camera_resolution(rgb_frame)
        
        # 检测人脸
        detection = self.face_detector.detect_face(rgb_frame)
        
        if not detection:
            return None, self.visualizer.draw_no_detection(rgb_frame.copy(), 
                                                          "YOLO")
        
        # 计算距离
        distances = []
        eye_results = {}
        depth_available = False
        
        for eye_name, eye_pos in [('left', detection['left_eye']), ('right', detection['right_eye'])]:
            # 深度图已对齐，直接使用RGB坐标
            depth_value = self.get_robust_depth(depth_frame, eye_pos[0], eye_pos[1])
            
            if depth_value:
                coord_3d = self.pixel_to_3d(eye_pos[0], eye_pos[1], depth_value)
                distance = self.calculate_distance_to_plane(coord_3d)
                
                distances.append(distance)
                eye_results[eye_name] = {
                    'position': eye_pos,
                    'depth': depth_value,
                    'coord_3d': coord_3d,
                    'distance': distance
                }
                depth_available = True
            else:
                eye_results[eye_name] = {
                    'position': eye_pos,
                    'depth': None,
                    'coord_3d': None,
                    'distance': None
                }
        
        # 处理距离
        raw_distance = np.mean(distances) if distances else None
        stable_distance = None
        
        if raw_distance:
            stable_distance = self.distance_processor.add_measurement(raw_distance)
        
        # 获取稳定性评分
        stability_score = self.distance_processor.get_stability_score()
        
        
        
        # 处理时间
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        
        # 构建结果
        results = {
            'frame': self.frame_count,
            'raw_distance': raw_distance,
            'stable_distance': stable_distance,
            'stability_score': stability_score,
            'eye_results': eye_results,
            'detection': detection,
            'process_time': process_time,
            'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'depth_available': depth_available
        }
        
        # 可视化
        visualization = self.visualizer.draw_visualization(
            rgb_frame.copy(), 
            results,
            "YOLO Face Model"
        )
        
        return results, visualization
    
    def reset(self):
        """重置系统"""
        self.frame_count = 0
        self.processing_times.clear()
        self.resolution_updated = False
        self.distance_processor = DistanceProcessor(Constants.SMOOTHING_WINDOW)