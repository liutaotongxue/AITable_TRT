"""
相机内参管理器
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from ctypes import pointer
from ..core.logger import logger

# 导入SDK定义
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
camera_sdk_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'camera')
if camera_sdk_dir not in sys.path:
    sys.path.insert(0, camera_sdk_dir)

try:
    from Mv3dRgbdImport.Mv3dRgbdDefine import (
        MV3D_RGBD_CAMERA_PARAM, 
        MV3D_RGBD_OK,
        ParamType_Int
    )  # type: ignore
    from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd  # type: ignore
    CAMERA_API_AVAILABLE = True
except (ImportError, RuntimeError):
    # Create dummy classes for when camera is not available
    class MV3D_RGBD_CAMERA_PARAM:
        pass
    
    class Mv3dRgbd:
        pass
    
    MV3D_RGBD_OK = 0
    CAMERA_API_AVAILABLE = False


class CameraIntrinsicsManager:
    """管理相机内参参数"""
    
    def __init__(self, camera: Mv3dRgbd):
        """
        初始化内参管理器
        
        Args:
            camera: TOF相机实例
        """
        self.camera = camera
        self.camera_available = CAMERA_API_AVAILABLE
        
        if not self.camera_available:
            # Create default/dummy parameters when camera is not available
            self.camera_params = None
            self.rgb_intrinsics = None
            self.depth_intrinsics = None
            self.depth_to_rgb_extrinsics = None
            return
            
        # 获取相机参数
        self.camera_params = self._get_camera_params()
        
        # 解析内参
        self.rgb_intrinsics = self._parse_rgb_intrinsics()
        self.depth_intrinsics = self._parse_depth_intrinsics()
        self.depth_to_rgb_extrinsics = self._parse_extrinsics()
        
        # 当前分辨率下的内参（动态更新）
        self.current_rgb_intrinsics = self.rgb_intrinsics.copy()
        self.current_depth_intrinsics = self.depth_intrinsics.copy()
        
        logger.info("Camera intrinsics initialized from SDK")
        logger.info(f"RGB intrinsics: fx={self.rgb_intrinsics['fx']:.2f}, fy={self.rgb_intrinsics['fy']:.2f}, "
                   f"cx={self.rgb_intrinsics['cx']:.2f}, cy={self.rgb_intrinsics['cy']:.2f}, "
                   f"resolution={self.rgb_intrinsics['width']}x{self.rgb_intrinsics['height']}")
        logger.info(f"Depth intrinsics: fx={self.depth_intrinsics['fx']:.2f}, fy={self.depth_intrinsics['fy']:.2f}, "
                   f"cx={self.depth_intrinsics['cx']:.2f}, cy={self.depth_intrinsics['cy']:.2f}, "
                   f"resolution={self.depth_intrinsics['width']}x{self.depth_intrinsics['height']}")
        
    def _get_camera_params(self) -> MV3D_RGBD_CAMERA_PARAM:
        """从SDK获取相机参数"""
        stCameraParam = MV3D_RGBD_CAMERA_PARAM()
        ret = self.camera.MV3D_RGBD_GetCameraParam(pointer(stCameraParam))
        
        if ret != MV3D_RGBD_OK:
            logger.warning(f"Failed to get camera params from SDK: {ret:#x}")
            # 返回默认参数作为备份
            return self._get_default_params()
            
        logger.info("Successfully retrieved camera parameters from SDK")
        return stCameraParam
    
    def _parse_rgb_intrinsics(self) -> Dict[str, float]:
        """解析RGB相机内参"""
        intrinsics = self.camera_params.stRgbCalibInfo.stIntrinsic.fData
        
        return {
            'fx': intrinsics[0],
            'fy': intrinsics[4],
            'cx': intrinsics[2],
            'cy': intrinsics[5],
            'width': self.camera_params.stRgbCalibInfo.nWidth,
            'height': self.camera_params.stRgbCalibInfo.nHeight
        }
    
    def _parse_depth_intrinsics(self) -> Dict[str, float]:
        """解析深度相机内参"""
        intrinsics = self.camera_params.stDepthCalibInfo.stIntrinsic.fData
        
        return {
            'fx': intrinsics[0],
            'fy': intrinsics[4],
            'cx': intrinsics[2],
            'cy': intrinsics[5],
            'width': self.camera_params.stDepthCalibInfo.nWidth,
            'height': self.camera_params.stDepthCalibInfo.nHeight
        }
    
    def _parse_extrinsics(self) -> np.ndarray:
        """解析外参矩阵（深度到RGB）"""
        extrinsics = self.camera_params.stDepth2RgbExtrinsic.fData
        return np.array(extrinsics).reshape((4, 4))
    
    def _get_default_params(self) -> MV3D_RGBD_CAMERA_PARAM:
        """获取默认参数（备用）"""
        params = MV3D_RGBD_CAMERA_PARAM()
        
        # RGB默认内参
        params.stRgbCalibInfo.nWidth = 640
        params.stRgbCalibInfo.nHeight = 480
        params.stRgbCalibInfo.stIntrinsic.fData[0] = 405.48887567  # fx
        params.stRgbCalibInfo.stIntrinsic.fData[4] = 405.38609241  # fy
        params.stRgbCalibInfo.stIntrinsic.fData[2] = 321.2703974   # cx
        params.stRgbCalibInfo.stIntrinsic.fData[5] = 231.30748631  # cy
        
        # 深度默认内参
        params.stDepthCalibInfo.nWidth = 640
        params.stDepthCalibInfo.nHeight = 480
        params.stDepthCalibInfo.stIntrinsic.fData[0] = 447.11  # fx
        params.stDepthCalibInfo.stIntrinsic.fData[4] = 444.08  # fy
        params.stDepthCalibInfo.stIntrinsic.fData[2] = 319.43  # cx
        params.stDepthCalibInfo.stIntrinsic.fData[5] = 239.41  # cy
        
        logger.warning("Using default camera parameters")
        return params
    
    def update_resolution(self, rgb_width: int, rgb_height: int):
        """更新分辨率并调整内参"""
        base_width = self.rgb_intrinsics['width']
        base_height = self.rgb_intrinsics['height']
        
        if base_width == 0 or base_height == 0:
            logger.warning("Base resolution is invalid, using current values")
            return
            
        scale_x = rgb_width / base_width
        scale_y = rgb_height / base_height
        
        # 更新RGB内参
        self.current_rgb_intrinsics = {
            'fx': self.rgb_intrinsics['fx'] * scale_x,
            'fy': self.rgb_intrinsics['fy'] * scale_y,
            'cx': self.rgb_intrinsics['cx'] * scale_x,
            'cy': self.rgb_intrinsics['cy'] * scale_y,
            'width': rgb_width,
            'height': rgb_height
        }
        
        logger.info(
            f"Updated RGB intrinsics for {rgb_width}x{rgb_height}: "
            f"fx={self.current_rgb_intrinsics['fx']:.2f}, "
            f"fy={self.current_rgb_intrinsics['fy']:.2f}, "
            f"cx={self.current_rgb_intrinsics['cx']:.2f}, "
            f"cy={self.current_rgb_intrinsics['cy']:.2f}"
        )
    
    def get_rgb_intrinsics_dict(self) -> Dict:
        """获取RGB内参字典（兼容格式）"""
        return {
            'fx': self.current_rgb_intrinsics['fx'],
            'fy': self.current_rgb_intrinsics['fy'],
            'cx': self.current_rgb_intrinsics['cx'],
            'cy': self.current_rgb_intrinsics['cy']
        }
    
    def get_depth_intrinsics_dict(self) -> Dict:
        """获取深度内参字典（兼容格式）"""
        return {
            'fx': self.current_depth_intrinsics['fx'],
            'fy': self.current_depth_intrinsics['fy'],
            'cx': self.current_depth_intrinsics['cx'],
            'cy': self.current_depth_intrinsics['cy']
        }
    
    def get_stereo_params_dict(self) -> Dict:
        """获取立体标定参数字典（兼容格式）"""
        extrinsics_4x4 = self.depth_to_rgb_extrinsics
        R = extrinsics_4x4[:3, :3]
        T = extrinsics_4x4[:3, 3:4] * 1000.0  # 转换为mm
        
        return {
            'R': R,
            'T': T
        }
    
    def get_calibration_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                          Optional[np.ndarray], Optional[np.ndarray], 
                                          Optional[np.ndarray]]:
        """获取相机标定参数（从SDK获取）
        
        Returns:
            Tuple containing:
            - ir_camera_intrinsic: IR相机内参矩阵 (3x3)
            - ir_camera_distortion: IR相机畸变系数
            - rgb_camera_intrinsic: RGB相机内参矩阵 (3x3)
            - rgb_camera_distortion: RGB相机畸变系数
            - depth_to_rgb_extrinsic: 深度到RGB的外参矩阵 (4x4)
            
        Raises:
            RuntimeError: 无法从SDK获取相机参数时
        """
        if not self.camera_available or self.camera is None:
            logger.warning("Camera not available, returning None values")
            return None, None, None, None, None
        
        try:
            # 获取传感器标定参数
            stCameraParam = MV3D_RGBD_CAMERA_PARAM()
            ret = self.camera.MV3D_RGBD_GetCameraParam(pointer(stCameraParam))
            
            if ret != MV3D_RGBD_OK:
                raise RuntimeError(
                    f"无法从SDK获取相机参数，错误码: 0x{ret:x}\n"
                    "请确保:\n"
                    "1. 相机已正确连接\n"
                    "2. 相机已正确初始化\n"
                    "3. SDK版本兼容"
                )
            
            # 提取内参和畸变参数
            ir_camera_intrinsic = np.array([*stCameraParam.stDepthCalibInfo.stIntrinsic.fData]).reshape(3, 3)
            ir_camera_distortion = np.array([*stCameraParam.stDepthCalibInfo.stDistortion.fData])
            
            rgb_camera_intrinsic = np.array([*stCameraParam.stRgbCalibInfo.stIntrinsic.fData]).reshape(3, 3)
            rgb_camera_distortion = np.array([*stCameraParam.stRgbCalibInfo.stDistortion.fData])
            
            depth_to_rgb_extrinsic = np.array([*stCameraParam.stDepth2RgbExtrinsic.fData]).reshape(4, 4)
            
            # 验证参数有效性
            if np.all(ir_camera_intrinsic == 0) or np.all(rgb_camera_intrinsic == 0):
                raise ValueError("从SDK获取的相机内参全为零，参数无效")
            
            logger.info("成功从SDK获取相机标定参数")
            logger.debug(f"IR内参: fx={ir_camera_intrinsic[0,0]:.2f}, fy={ir_camera_intrinsic[1,1]:.2f}")
            logger.debug(f"RGB内参: fx={rgb_camera_intrinsic[0,0]:.2f}, fy={rgb_camera_intrinsic[1,1]:.2f}")
            
            return ir_camera_intrinsic, ir_camera_distortion, rgb_camera_intrinsic, rgb_camera_distortion, depth_to_rgb_extrinsic
            
        except RuntimeError:
            raise  # 重新抛出RuntimeError
        except Exception as e:
            raise RuntimeError(f"获取相机标定参数时发生异常: {e}")
    
    def get_undistortion_maps(self, rgb_resolution: Tuple[int, int], 
                             depth_resolution: Tuple[int, int],
                             undistortion_alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, 
                                                                      np.ndarray, np.ndarray]:
        """获取新的相机映射表，用于图像去畸变
        
        Args:
            rgb_resolution: RGB图像分辨率 (width, height)
            depth_resolution: 深度图像分辨率 (width, height)
            undistortion_alpha: 去畸变参数
            
        Returns:
            Tuple containing:
            - mapx_rgb: RGB图像X轴映射表
            - mapy_rgb: RGB图像Y轴映射表
            - mapx_ir: IR图像X轴映射表
            - mapy_ir: IR图像Y轴映射表
        """
        try:
            # 获取标定数据
            ir_intrinsic, ir_distortion, rgb_intrinsic, rgb_distortion, _ = self.get_calibration_data()
            
            if ir_intrinsic is None or rgb_intrinsic is None:
                logger.warning("Calibration data not available, creating default maps")
                # 返回默认的映射表（无畸变校正）
                rgb_width, rgb_height = rgb_resolution
                depth_width, depth_height = depth_resolution
                
                mapx_rgb = np.arange(rgb_width).reshape(1, -1).repeat(rgb_height, axis=0).astype(np.float32)
                mapy_rgb = np.arange(rgb_height).reshape(-1, 1).repeat(rgb_width, axis=1).astype(np.float32)
                mapx_ir = np.arange(depth_width).reshape(1, -1).repeat(depth_height, axis=0).astype(np.float32)
                mapy_ir = np.arange(depth_height).reshape(-1, 1).repeat(depth_width, axis=1).astype(np.float32)
                
                return mapx_rgb, mapy_rgb, mapx_ir, mapy_ir
            
            rgb_width, rgb_height = rgb_resolution
            depth_width, depth_height = depth_resolution
            
            new_mtx_rgb_undistort, roi = cv2.getOptimalNewCameraMatrix(
                rgb_intrinsic, rgb_distortion, 
                (rgb_width, rgb_height), undistortion_alpha, 
                (rgb_width, rgb_height)
            )
            
            mapx_rgb, mapy_rgb = cv2.initUndistortRectifyMap(
                rgb_intrinsic, rgb_distortion, np.eye(3), 
                new_mtx_rgb_undistort, (rgb_width, rgb_height), cv2.CV_16SC2
            )
            
            mapx_ir, mapy_ir = cv2.initUndistortRectifyMap(
                ir_intrinsic, ir_distortion, np.eye(3), 
                ir_intrinsic, (depth_width, depth_height), cv2.CV_16SC2
            )
            
            logger.debug("Camera undistortion maps created successfully")
            return mapx_rgb, mapy_rgb, mapx_ir, mapy_ir
            
        except Exception as e:
            logger.error(f"Error creating undistortion maps: {e}")
            raise