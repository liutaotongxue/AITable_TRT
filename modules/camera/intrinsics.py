"""
相机内参管理器
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any
from ctypes import pointer
from ..core.logger import logger


class CameraIntrinsicsManager:
    """管理相机内参参数"""
    
    def __init__(self, camera, sdk_classes=None):
        """
        初始化内参管理器

        Args:
            camera: TOF相机实例（已通过tof_sdk_loader加载SDK）
            sdk_classes: SDK类字典，包含MV3D_RGBD_CAMERA_PARAM和MV3D_RGBD_OK
        """
        self.camera = camera

        # 保存SDK类引用（用于类型注解和常量）
        if sdk_classes:
            self.MV3D_RGBD_CAMERA_PARAM = sdk_classes.get('MV3D_RGBD_CAMERA_PARAM')
            self.MV3D_RGBD_OK = sdk_classes.get('MV3D_RGBD_OK', 0)
        else:
            # Dummy fallback（不应该发生，但保持健壮性）
            self.MV3D_RGBD_CAMERA_PARAM = type('MV3D_RGBD_CAMERA_PARAM', (), {})
            self.MV3D_RGBD_OK = 0
            logger.warning("SDK classes not provided, using dummy fallback")

        # 获取相机参数
        try:
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
        except Exception as e:
            logger.error(f"Error parsing camera intrinsics: {e}, using default values")
            # 如果解析失败，填充默认值，避免后续访问属性时报错
            self.camera_params = None
            self.rgb_intrinsics = {
                'fx': 405.0, 'fy': 405.0, 'cx': 320.0, 'cy': 240.0,
                'width': 640, 'height': 480
            }
            self.depth_intrinsics = {
                'fx': 447.0, 'fy': 444.0, 'cx': 319.0, 'cy': 239.0,
                'width': 640, 'height': 480
            }
            self.depth_to_rgb_extrinsics = np.eye(4)
            self.current_rgb_intrinsics = self.rgb_intrinsics.copy()
            self.current_depth_intrinsics = self.depth_intrinsics.copy()
        
    def _get_camera_params(self):
        """从SDK获取相机参数"""
        stCameraParam = self.MV3D_RGBD_CAMERA_PARAM()
        ret = self.camera.MV3D_RGBD_GetCameraParam(pointer(stCameraParam))

        if ret != self.MV3D_RGBD_OK:
            logger.warning(f"Failed to get camera params from SDK: {ret:#x}")
            # 返回默认参数作为备份
            return self._get_default_params()

        logger.info("Successfully retrieved camera parameters from SDK")
        return stCameraParam
    
    def _parse_rgb_intrinsics(self) -> Dict[str, float]:
        """解析RGB相机内参"""
        intrinsics = self.camera_params.stRgbCalibInfo.stIntrinsic.fData
        width = self.camera_params.stRgbCalibInfo.nWidth
        height = self.camera_params.stRgbCalibInfo.nHeight

        # 如果 SDK 返回的分辨率为 0，使用默认值（老 SDK 兼容）
        if width == 0 or height == 0:
            logger.warning(f"SDK returned invalid RGB resolution: {width}x{height}, using default 640x480")
            width = 640
            height = 480

        return {
            'fx': intrinsics[0] if intrinsics[0] != 0 else 405.0,
            'fy': intrinsics[4] if intrinsics[4] != 0 else 405.0,
            'cx': intrinsics[2] if intrinsics[2] != 0 else 320.0,
            'cy': intrinsics[5] if intrinsics[5] != 0 else 240.0,
            'width': width,
            'height': height
        }
    
    def _parse_depth_intrinsics(self) -> Dict[str, float]:
        """解析深度相机内参"""
        intrinsics = self.camera_params.stDepthCalibInfo.stIntrinsic.fData
        width = self.camera_params.stDepthCalibInfo.nWidth
        height = self.camera_params.stDepthCalibInfo.nHeight

        # 如果 SDK 返回的分辨率为 0，使用默认值（老 SDK 兼容）
        if width == 0 or height == 0:
            logger.warning(f"SDK returned invalid Depth resolution: {width}x{height}, using default 640x480")
            width = 640
            height = 480

        return {
            'fx': intrinsics[0] if intrinsics[0] != 0 else 447.0,
            'fy': intrinsics[4] if intrinsics[4] != 0 else 444.0,
            'cx': intrinsics[2] if intrinsics[2] != 0 else 319.0,
            'cy': intrinsics[5] if intrinsics[5] != 0 else 239.0,
            'width': width,
            'height': height
        }
    
    def _parse_extrinsics(self) -> np.ndarray:
        """解析外参矩阵（深度到RGB）"""
        extrinsics = self.camera_params.stDepth2RgbExtrinsic.fData
        return np.array(extrinsics).reshape((4, 4))
    
    def _get_default_params(self):
        """获取默认参数（备用）"""
        params = self.MV3D_RGBD_CAMERA_PARAM()
        
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

    def update_resolution_from_frame(self, rgb_frame: Optional[Any]) -> bool:
        """
        从实际帧数据更新分辨率（老 SDK 兼容）

        当 SDK 返回 0x0 分辨率时，可以从第一帧实际数据中获取真实分辨率

        Args:
            rgb_frame: RGB 帧数据（numpy array）

        Returns:
            bool: 更新成功返回 True，失败返回 False
        """
        if rgb_frame is None:
            return False

        try:
            import numpy as np
            if isinstance(rgb_frame, np.ndarray) and len(rgb_frame.shape) >= 2:
                actual_height, actual_width = rgb_frame.shape[:2]

                # 只在当前分辨率无效时才更新
                if self.current_rgb_intrinsics['width'] == 0 or self.current_rgb_intrinsics['height'] == 0:
                    logger.info(f"Detected actual frame resolution: {actual_width}x{actual_height}, updating intrinsics")
                    # 直接设置实际分辨率（不缩放，因为基准分辨率就是实际的）
                    self.rgb_intrinsics['width'] = actual_width
                    self.rgb_intrinsics['height'] = actual_height
                    self.current_rgb_intrinsics['width'] = actual_width
                    self.current_rgb_intrinsics['height'] = actual_height
                    return True
        except Exception as e:
            logger.debug(f"Failed to update resolution from frame: {e}")

        return False

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
        # 若外参为空（SDK 未提供或默认占位），使用单位矩阵防止上层崩溃
        if self.depth_to_rgb_extrinsics is None:
            logger.warning("Depth-to-RGB extrinsics not available, using identity fallback")
            extrinsics_4x4 = np.eye(4, dtype=float)
        else:
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
        if self.camera is None:
            logger.warning("Camera not available, returning None values")
            return None, None, None, None, None

        try:
            # 获取传感器标定参数
            stCameraParam = self.MV3D_RGBD_CAMERA_PARAM()
            ret = self.camera.MV3D_RGBD_GetCameraParam(pointer(stCameraParam))

            if ret != self.MV3D_RGBD_OK:
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
            
            # Validate parameter validity
            if np.all(ir_camera_intrinsic == 0) or np.all(rgb_camera_intrinsic == 0):
                raise ValueError("Camera intrinsics from SDK are all zeros, parameters invalid")

            logger.info("Successfully retrieved camera calibration parameters from SDK")
            logger.debug(f"IR intrinsics: fx={ir_camera_intrinsic[0,0]:.2f}, fy={ir_camera_intrinsic[1,1]:.2f}")
            logger.debug(f"RGB intrinsics: fx={rgb_camera_intrinsic[0,0]:.2f}, fy={rgb_camera_intrinsic[1,1]:.2f}")
            
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
