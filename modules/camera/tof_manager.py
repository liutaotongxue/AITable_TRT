"""
TOF相机管理器
"""
import os
import sys
import ctypes
from ctypes import *
from typing import Optional
from ..core.logger import logger

# 添加camera目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
camera_sdk_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'camera')
if camera_sdk_dir not in sys.path:
    sys.path.insert(0, camera_sdk_dir)

try:
    from Mv3dRgbdImport.Mv3dRgbdDefine import (  # type: ignore
        MV3D_RGBD_DEVICE_INFO_LIST,
        MV3D_RGBD_FRAME_DATA,
        MV3D_RGBD_CAMERA_PARAM,
        MV3D_RGBD_PARAM,
        DeviceType_Ethernet,
        DeviceType_USB,
        DeviceType_Ethernet_Vir,
        DeviceType_USB_Vir,
        ImageType_YUV422,
        ImageType_RGB8_Planar,
        ImageType_Mono8,
        ImageType_Depth,
        ImageType_Rgbd,
        MV3D_RGBD_OK,
        ParamType_Float,
        MV3D_RGBD_FLOAT_Z_UNIT
    )
    from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd  # type: ignore
    TOF_CAMERA_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    logger.warning(f"TOF camera API not available: {e}")
    logger.warning("Camera functionality will be disabled")
    TOF_CAMERA_AVAILABLE = False
    
    # Create dummy classes and constants for graceful degradation
    class MV3D_RGBD_DEVICE_INFO_LIST:
        pass
    
    class MV3D_RGBD_FRAME_DATA:
        pass
    
    class MV3D_RGBD_CAMERA_PARAM:
        pass
    
    class MV3D_RGBD_PARAM:
        pass
    
    class Mv3dRgbd:
        pass
    
    # Dummy constants
    DeviceType_Ethernet = 0
    DeviceType_USB = 1
    DeviceType_Ethernet_Vir = 2
    DeviceType_USB_Vir = 3
    ImageType_YUV422 = 0
    ImageType_RGB8_Planar = 1
    ImageType_Mono8 = 2
    ImageType_Depth = 3
    ImageType_Rgbd = 4
    MV3D_RGBD_OK = 0
    ParamType_Float = 1
    MV3D_RGBD_FLOAT_Z_UNIT = 0


class TOFCameraManager:
    """TOF相机资源管理器"""
    
    def __init__(self):
        self.camera = None
        self.device_list = None
        self.is_initialized = False
        self.intrinsics_manager = None
        self.camera_available = TOF_CAMERA_AVAILABLE
        
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def initialize(self):
        """初始化相机"""
        if not self.camera_available:
            logger.warning("TOF camera not available - skipping initialization")
            return False
            
        try:
            # 获取设备数量
            nDeviceNum = ctypes.c_uint(0)
            ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(
                DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir, 
                byref(nDeviceNum)
            )
            if ret != 0 or nDeviceNum.value == 0:
                raise RuntimeError(f"No TOF camera found (Error: 0x{ret:x})")
            
            # 获取设备列表
            self.device_list = MV3D_RGBD_DEVICE_INFO_LIST()
            ret = Mv3dRgbd.MV3D_RGBD_GetDeviceList(
                DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir,
                pointer(self.device_list.DeviceInfo[0]), 20, byref(nDeviceNum)
            )
            if ret != 0:
                raise RuntimeError(f"Failed to get device list (Error: 0x{ret:x})")
            
            # 打开设备
            self.camera = Mv3dRgbd()
            ret = self.camera.MV3D_RGBD_OpenDevice(pointer(self.device_list.DeviceInfo[0]))
            if ret != 0:
                raise RuntimeError(f"Failed to open device (Error: 0x{ret:x})")
            
            # 设置图像对齐
            try:
                align_value = ctypes.c_int(1)
                ret = self.camera.MV3D_RGBD_SetIntValue("ImageAlign", align_value)
                if ret == 0:
                    logger.info("Image alignment enabled: depth aligned to RGB coordinates")
            except Exception as e:
                logger.warning(f"Error setting image alignment: {e}")
            
            # 设置RGBD输出模式
            try:
                rgbd_output = ctypes.c_int(1)
                ret = self.camera.MV3D_RGBD_SetIntValue("OutputRgbd", rgbd_output)
                if ret == 0:
                    logger.info("RGBD output enabled")
            except Exception as e:
                logger.warning(f"Error setting RGBD output: {e}")
            
            # 开始取流
            ret = self.camera.MV3D_RGBD_Start()
            if ret != 0:
                self.camera.MV3D_RGBD_CloseDevice()
                raise RuntimeError(f"Failed to start camera (Error: 0x{ret:x})")
            
            # 初始化内参管理器
            from .intrinsics import CameraIntrinsicsManager
            self.intrinsics_manager = CameraIntrinsicsManager(self.camera)
            
            self.is_initialized = True
            logger.info("TOF camera initialized successfully with SDK intrinsics")
            
        except RuntimeError as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
            
    def cleanup(self):
        """清理相机资源"""
        if self.camera_available and self.camera and self.is_initialized:
            try:
                self.camera.MV3D_RGBD_Stop()
                self.camera.MV3D_RGBD_CloseDevice()
                logger.info("TOF camera cleaned up")
            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")
    
    def fetch_frame(self, timeout: int = 5000) -> Optional[MV3D_RGBD_FRAME_DATA]:
        """获取一帧数据"""
        if not self.camera_available or not self.is_initialized:
            return None
            
        try:
            frame_data = MV3D_RGBD_FRAME_DATA()
            ret = self.camera.MV3D_RGBD_FetchFrame(pointer(frame_data), timeout)
            if ret == 0:
                return frame_data
            else:
                logger.warning(f"Frame fetch failed: 0x{ret:x}")
                return None
        except Exception as e:
            logger.error(f"Error fetching frame: {e}")
            return None