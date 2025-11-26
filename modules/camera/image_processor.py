"""
图像数据处理器
"""
import numpy as np
import cv2
import ctypes
import struct
from typing import Tuple, Optional, Dict, Any
from ctypes import string_at
from ..core.logger import logger


class ImageDataProcessor:
    """
    图像数据处理器

    需要从 TOFCameraManager.get_sdk_classes() 获取 SDK 常量
    """

    # 默认常量值（仅作为后备，正常情况应使用 SDK 提供的值）
    DEFAULT_IMAGE_TYPES = {
        'ImageType_YUV422': 0,
        'ImageType_RGB8_Planar': 1,
        'ImageType_Mono8': 2,
        'ImageType_Depth': 3,
        'ImageType_Rgbd': 4,
    }

    def __init__(self, sdk_classes: Optional[Dict[str, Any]] = None):
        """
        初始化图像数据处理器

        Args:
            sdk_classes: SDK 常量字典，从 TOFCameraManager.get_sdk_classes() 获取
                        包含 ImageType_YUV422, ImageType_RGB8_Planar 等常量
        """
        if sdk_classes:
            # 使用 SDK 提供的真实常量
            self.ImageType_YUV422 = sdk_classes.get('ImageType_YUV422', self.DEFAULT_IMAGE_TYPES['ImageType_YUV422'])
            self.ImageType_RGB8_Planar = sdk_classes.get('ImageType_RGB8_Planar', self.DEFAULT_IMAGE_TYPES['ImageType_RGB8_Planar'])
            self.ImageType_Mono8 = sdk_classes.get('ImageType_Mono8', self.DEFAULT_IMAGE_TYPES['ImageType_Mono8'])
            self.ImageType_Depth = sdk_classes.get('ImageType_Depth', self.DEFAULT_IMAGE_TYPES['ImageType_Depth'])
            self.ImageType_Rgbd = sdk_classes.get('ImageType_Rgbd', self.DEFAULT_IMAGE_TYPES['ImageType_Rgbd'])
            logger.debug("ImageDataProcessor initialized with SDK constants")
        else:
            # 使用默认值（警告：可能与实际 SDK 不匹配）
            self.ImageType_YUV422 = self.DEFAULT_IMAGE_TYPES['ImageType_YUV422']
            self.ImageType_RGB8_Planar = self.DEFAULT_IMAGE_TYPES['ImageType_RGB8_Planar']
            self.ImageType_Mono8 = self.DEFAULT_IMAGE_TYPES['ImageType_Mono8']
            self.ImageType_Depth = self.DEFAULT_IMAGE_TYPES['ImageType_Depth']
            self.ImageType_Rgbd = self.DEFAULT_IMAGE_TYPES['ImageType_Rgbd']
            logger.warning("ImageDataProcessor initialized with DEFAULT constants - may not match SDK!")

    def validate_image_data(self, image_data) -> bool:
        """验证图像数据有效性"""
        return all([
            hasattr(image_data, 'nWidth'), hasattr(image_data, 'nHeight'),
            hasattr(image_data, 'nDataLen'), hasattr(image_data, 'pData'),
            image_data.nWidth > 0, image_data.nHeight > 0,
            image_data.nDataLen > 0, image_data.pData
        ])

    def extract_raw_data(self, image_data, dtype=np.uint8):
        """提取原始数据"""
        if image_data.enImageType == self.ImageType_Depth:
            element_size = 2
            dtype = np.uint16
            c_type = ctypes.c_uint16
        else:
            element_size = 1
            c_type = ctypes.c_ubyte

        num_elements = image_data.nDataLen // element_size

        if hasattr(image_data.pData, 'contents'):
            data_ptr = ctypes.cast(image_data.pData, ctypes.POINTER(c_type * num_elements))
            return np.array(data_ptr.contents, dtype=dtype)
        else:
            return np.frombuffer(
                (c_type * num_elements).from_address(image_data.pData),
                dtype=dtype
            )

    def process_frame_data(self, frame_data) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """处理帧数据，返回 RGB 和深度图"""
        rgb_frame = None
        depth_frame = None

        for i in range(frame_data.nImageCount):
            image_data = frame_data.stImageData[i]

            if not self.validate_image_data(image_data):
                continue

            try:
                if image_data.enImageType == self.ImageType_Depth:
                    depth_data = self.extract_raw_data(image_data)
                    depth_frame = depth_data.reshape((image_data.nHeight, image_data.nWidth))

                elif image_data.enImageType == self.ImageType_Mono8:
                    mono_data = self.extract_raw_data(image_data)
                    mono_frame = mono_data.reshape((image_data.nHeight, image_data.nWidth))
                    rgb_frame = cv2.cvtColor(mono_frame, cv2.COLOR_GRAY2BGR)

                elif image_data.enImageType == self.ImageType_YUV422:
                    yuv_data = self.extract_raw_data(image_data)
                    yuv_frame = yuv_data.reshape((image_data.nHeight, image_data.nWidth * 2))
                    rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YUYV)

                elif image_data.enImageType == self.ImageType_RGB8_Planar:
                    rgb_data = self.extract_raw_data(image_data)
                    height, width = image_data.nHeight, image_data.nWidth
                    plane_size = height * width

                    r_plane = rgb_data[:plane_size].reshape((height, width))
                    g_plane = rgb_data[plane_size:2*plane_size].reshape((height, width))
                    b_plane = rgb_data[2*plane_size:3*plane_size].reshape((height, width))
                    rgb_frame = np.stack([b_plane, g_plane, r_plane], axis=2)

                elif image_data.enImageType == self.ImageType_Rgbd:
                    rgbd_data = self.extract_raw_data(image_data, dtype=np.uint8)
                    height, width = image_data.nHeight, image_data.nWidth

                    if len(rgbd_data) == height * width * 5:
                        rgbd_reshaped = rgbd_data.reshape((height, width, 5))
                        rgb_frame = rgbd_reshaped[:, :, :3]
                        depth_raw = rgbd_reshaped[:, :, 3:5]
                        depth_frame = (depth_raw[:, :, 1].astype(np.uint16) << 8) | depth_raw[:, :, 0].astype(np.uint16)
                        logger.debug(f"Processed RGBD image: RGB {rgb_frame.shape}, Depth {depth_frame.shape}")

            except Exception as e:
                logger.error(f"Error processing image type {image_data.enImageType}: {e}")
                continue

        return rgb_frame, depth_frame
    
    @staticmethod
    def get_depth_value_mm(depth_data_bytes: bytes, length: int, 
                          depth_resolution: Tuple[int, int] = (640, 480),
                          enable_spatial_filter: bool = True,
                          enable_temporal_filter: bool = True) -> np.ndarray:
        """将深度图像字节数据转换为毫米值
        
        Args:
            depth_data_bytes: 深度图像的字节数据
            length: 深度图像的数据长度
            depth_resolution: 深度图像分辨率
            enable_spatial_filter: 启用空间滤波
            enable_temporal_filter: 启用时域滤波
            
        Returns:
            深度图像数组（单位：毫米）
        """
        try:
            strMode = string_at(depth_data_bytes, length)
            
            # 解包深度数据（16位无符号整数）
            sValue = struct.unpack('H' * int(len(strMode) / 2), strMode)
            
            # 重塑为深度图像
            depth_width, depth_height = depth_resolution
            sValue = np.array(sValue).reshape(depth_height, depth_width).astype(np.float32)
            
            # 应用滤波
            if enable_spatial_filter:
                sValue = ImageDataProcessor.apply_spatial_filter(sValue)
            
            if enable_temporal_filter:
                depth_mm = ImageDataProcessor.apply_temporal_filter(sValue)
            else:
                depth_mm = sValue
                
            return depth_mm
            
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")
            raise
    
    @staticmethod
    def apply_temporal_filter(depth_mm: np.ndarray, 
                             history_size: int = 10,
                             filter_method: str = 'median') -> np.ndarray:
        """应用时域滤波
        
        Args:
            depth_mm: 深度图像数组（毫米）
            history_size: 历史缓存大小
            filter_method: 滤波方法 ('median' 或 'mean')
            
        Returns:
            滤波后的深度图像
        """
        try:
            # 获取全局历史缓存
            if not hasattr(ImageDataProcessor, '_depth_history'):
                ImageDataProcessor._depth_history = []
            
            ImageDataProcessor._depth_history.append(depth_mm.copy())
            
            # 保持历史记录大小
            if len(ImageDataProcessor._depth_history) > history_size:
                ImageDataProcessor._depth_history.pop(0)
            
            if len(ImageDataProcessor._depth_history) == 1:
                return ImageDataProcessor._depth_history[0]
            
            # 时域滤波
            depth_stack = np.stack(ImageDataProcessor._depth_history, axis=0)
            
            if filter_method == 'median':
                filtered = np.median(depth_stack, axis=0)
            elif filter_method == 'mean':
                filtered = np.mean(depth_stack, axis=0)
            else:
                filtered = depth_stack[-1]  # 使用最新帧
                
            return filtered
            
        except Exception as e:
            logger.error(f"Error in temporal filtering: {e}")
            return depth_mm
    
    @staticmethod
    def apply_spatial_filter(depth_mm: np.ndarray, 
                           kernel_size: int = 5,
                           sigma: float = 1.0) -> np.ndarray:
        """应用空间滤波
        
        Args:
            depth_mm: 深度图像数组（毫米）
            kernel_size: 核大小
            sigma: 高斯核标准差
            
        Returns:
            滤波后的深度图像
        """
        try:
            filtered = cv2.GaussianBlur(
                depth_mm,
                (kernel_size, kernel_size),
                sigma
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error in spatial filtering: {e}")
            return depth_mm
    
    @staticmethod
    def get_depth_from_neighbor(depth_mm: np.ndarray, 
                               keypoint_rgb: Tuple[int, int], 
                               neighbor_size: int = 5,
                               min_depth: int = 200,
                               max_depth: int = 1500,
                               min_valid_ratio: float = 0.3,
                               depth_std_threshold: float = 10.0) -> Optional[float]:
        """从邻域获取稳定的深度值
        
        Args:
            depth_mm: 深度图像数组（毫米）
            keypoint_rgb: RGB图像中的关键点坐标 (x, y)
            neighbor_size: 邻域大小
            min_depth: 最小有效深度
            max_depth: 最大有效深度
            min_valid_ratio: 最小有效深度比例
            depth_std_threshold: 深度标准差阈值
            
        Returns:
            有效深度值（毫米）或None
        """
        try:
            half_size = neighbor_size // 2
            x_min = max(0, keypoint_rgb[0] - half_size)
            x_max = min(depth_mm.shape[1], keypoint_rgb[0] + half_size + 1)
            y_min = max(0, keypoint_rgb[1] - half_size)
            y_max = min(depth_mm.shape[0], keypoint_rgb[1] + half_size + 1)
            
            # 提取邻域
            neighborhood = depth_mm[y_min:y_max, x_min:x_max]
            
            # 过滤有效深度值
            valid_depths = neighborhood[
                (neighborhood > min_depth) & (neighborhood < max_depth)
            ]
            
            if len(valid_depths) == 0:
                logger.debug(f"No valid depth values found in neighborhood around {keypoint_rgb}")
                return None
            
            # 检查有效深度比例
            valid_ratio = len(valid_depths) / neighborhood.size
            if valid_ratio < min_valid_ratio:
                logger.debug(f"Insufficient valid depth ratio: {valid_ratio:.2f}")
                return None
            
            # 使用中值作为稳定的深度估计
            keypoint_depth_median = np.median(valid_depths)
            
            # 检查深度值的稳定性
            if len(valid_depths) > 3:
                depth_std = np.std(valid_depths)
                if depth_std > depth_std_threshold:
                    logger.debug(f"High depth variance: std={depth_std:.2f}mm")
            
            return float(keypoint_depth_median)
            
        except Exception as e:
            logger.error(f"Error getting depth from neighbor: {e}")
            return None