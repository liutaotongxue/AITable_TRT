"""
3D坐标转换器
"""
import numpy as np
from typing import Tuple, Dict, Optional
from ..core.logger import logger


class CoordinateConverter:
    """3D坐标转换器 - 处理2D到3D的坐标转换"""
    
    @staticmethod
    def get_keypoint_3d_coords_m(keypoint_2d: Tuple[int, int], 
                                depth_mm: np.ndarray, 
                                camera_params: np.ndarray) -> Optional[list]:
        """将2D关键点转换为3D坐标
        
        Args:
            keypoint_2d: 2D像素坐标 (x, y)
            depth_mm: 深度图像数组（毫米）
            camera_params: 相机内参矩阵 (3x3)
            
        Returns:
            3D坐标 [x, y, z] （毫米）或None
        """
        try:
            pixel_x, pixel_y = keypoint_2d
            
            # 从邻域获取深度值
            from .image_processor import ImageDataProcessor
            depth_mm_value = ImageDataProcessor.get_depth_from_neighbor(
                depth_mm, keypoint_2d, neighbor_size=5
            )
            
            if depth_mm_value is None:
                return None
                
            z_mm = depth_mm_value
            x_mm = (pixel_x - camera_params[0, 2]) * z_mm / camera_params[0, 0]
            y_mm = (pixel_y - camera_params[1, 2]) * z_mm / camera_params[1, 1]
            
            return [x_mm, y_mm, z_mm]
            
        except Exception as e:
            logger.error(f"Error converting 2D to 3D coordinates: {e}")
            return None

    @staticmethod
    def get_keypoints_3d_coords_m(keypoints_2d: Dict[str, Tuple[int, int]], 
                                 depth_mm: np.ndarray,
                                 camera_params: np.ndarray) -> Dict[str, Optional[list]]:
        """批量将2D关键点转换为3D坐标
        
        Args:
            keypoints_2d: 2D关键点字典 {名称: (x, y)}
            depth_mm: 深度图像数组（毫米）
            camera_params: 相机内参矩阵
            
        Returns:
            3D关键点字典 {名称: [x, y, z]}
        """
        keypoints_3d = {}
        
        for key, value in keypoints_2d.items():
            keypoints_3d[key] = CoordinateConverter.get_keypoint_3d_coords_m(
                value, depth_mm, camera_params
            )
            
        return keypoints_3d
    
    @staticmethod
    def pixel_to_3d_point(pixel_x: int, pixel_y: int, 
                         depth_value: float, 
                         fx: float, fy: float, 
                         cx: float, cy: float) -> Tuple[float, float, float]:
        """将像素坐标和深度值转换为3D点坐标
        
        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth_value: 深度值（毫米）
            fx, fy: 相机焦距
            cx, cy: 光心坐标
            
        Returns:
            3D点坐标 (x, y, z) （毫米）
        """
        x = (pixel_x - cx) * depth_value / fx
        y = (pixel_y - cy) * depth_value / fy
        z = depth_value
        
        return x, y, z
    
    @staticmethod
    def batch_pixel_to_3d(pixel_coords: np.ndarray, 
                         depth_values: np.ndarray,
                         camera_matrix: np.ndarray) -> np.ndarray:
        """批量将像素坐标转换为3D坐标
        
        Args:
            pixel_coords: 像素坐标数组 (N, 2)
            depth_values: 对应的深度值 (N,)
            camera_matrix: 相机内参矩阵 (3, 3)
            
        Returns:
            3D坐标数组 (N, 3)
        """
        try:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # 向量化计算
            x_3d = (pixel_coords[:, 0] - cx) * depth_values / fx
            y_3d = (pixel_coords[:, 1] - cy) * depth_values / fy
            z_3d = depth_values
            
            return np.column_stack([x_3d, y_3d, z_3d])
            
        except Exception as e:
            logger.error(f"Error in batch pixel to 3D conversion: {e}")
            return np.array([])
    
    @staticmethod
    def transform_3d_points(points_3d: np.ndarray, 
                           rotation_matrix: np.ndarray, 
                           translation_vector: np.ndarray) -> np.ndarray:
        """对3D点进行旋转和平移变换
        
        Args:
            points_3d: 3D点坐标 (N, 3)
            rotation_matrix: 旋转矩阵 (3, 3)
            translation_vector: 平移向量 (3, 1) 或 (3,)
            
        Returns:
            变换后的3D点坐标 (N, 3)
        """
        try:
            if translation_vector.ndim == 1:
                translation_vector = translation_vector.reshape(-1, 1)
            
            # 应用旋转和平移：P' = R*P + T
            transformed_points = (rotation_matrix @ points_3d.T + translation_vector).T
            
            return transformed_points
            
        except Exception as e:
            logger.error(f"Error transforming 3D points: {e}")
            return points_3d
    
    @staticmethod
    def compute_distance_3d(point1: Tuple[float, float, float], 
                           point2: Tuple[float, float, float]) -> float:
        """计算两个3D点之间的欧氏距离
        
        Args:
            point1: 第一个3D点 (x, y, z)
            point2: 第二个3D点 (x, y, z)
            
        Returns:
            欧氏距离（毫米）
        """
        try:
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error computing 3D distance: {e}")
            return 0.0
    
    @staticmethod
    def project_3d_to_2d(points_3d: np.ndarray, 
                        camera_matrix: np.ndarray,
                        distortion_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """将3D点投影到2D图像平面
        
        Args:
            points_3d: 3D点坐标 (N, 3)
            camera_matrix: 相机内参矩阵 (3, 3)
            distortion_coeffs: 畸变系数（可选）
            
        Returns:
            2D像素坐标 (N, 2)
        """
        try:
            # 齐次坐标投影
            points_2d_homo = (camera_matrix @ points_3d.T).T
            
            # 除以深度得到像素坐标
            points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
            
            # TODO: 如果需要，可以添加畸变校正
            if distortion_coeffs is not None:
                logger.warning("Distortion correction not implemented yet")
            
            return points_2d
            
        except Exception as e:
            logger.error(f"Error projecting 3D to 2D: {e}")
            return np.array([])