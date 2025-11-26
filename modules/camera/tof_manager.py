"""
TOF Camera Manager
Implements DepthCameraInterface
"""
import ctypes
from ctypes import *
import time
import threading
from typing import Optional, Dict, Any
from ..core.logger import logger
from .depth_camera_interface import DepthCameraInterface
from .tof_sdk_loader import TOFSDKLoader
from .image_processor import ImageDataProcessor


class TOFCameraManager(DepthCameraInterface):
    """
    TOF Camera Resource Manager
    Implements DepthCameraInterface
    """

    def __init__(self, sdk_python_path: Optional[str] = None,
                 sdk_library_path: Optional[str] = None):
        """
        Initialize TOF Camera Manager

        Args:
            sdk_python_path: SDK Python module path (optional, from system_config.json)
            sdk_library_path: SDK library path (optional, from system_config.json)
        """
        self._camera = None  # Mv3dRgbd instance (internal variable, avoid property conflict with base class)
        self.device_list = None
        self._is_initialized = False
        self._intrinsics_manager = None
        self._camera_fps = 0.0
        self._z_unit = None

        # 状态管理
        self._camera_status = "not_initialized"  # ok, error, soft_restarting, recovering
        self._last_frame_time = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10  # 连续失败阈值，触发软重启
        self._has_ever_initialized = False  # 是否曾经成功初始化（区分首次启动失败和恢复失败）

        # 软重启统计
        self._total_soft_restarts = 0
        self._total_manual_restarts = 0  # 手动重启次数
        self._total_auto_restarts = 0    # 自动重启次数
        self._total_reopens = 0
        self._last_restart_time = None
        self._restart_cooldown = 5.0  # 重启冷却时间（秒）

        # 指数退避策略（区分首次启动和恢复）
        self._consecutive_restart_failures = 0  # 连续重启失败次数
        self._max_restart_failures = 3         # 最大连续失败次数（恢复场景）
        self._max_initial_failures = 5         # 最大连续失败次数（首次启动场景，更宽容）
        self._base_backoff_time = 10.0         # 基础退避时间（秒）
        self._initial_backoff_time = 5.0       # 首次启动退避时间（秒，更短）
        self._max_backoff_time = 300.0         # 最大退避时间（5分钟）
        self._backoff_multiplier = 2.0         # 退避倍数

        # FPS 追踪（用于遥测诊断）
        self._last_successful_frame_time = None  # 最后一次成功获取帧的时间戳
        self._frame_times = []  # 最近帧时间戳列表（用于计算 FPS）
        self._fps_window = 10  # FPS 计算窗口（最近 N 帧）
        
        # 保存SDK路径以便重启时使用
        self.sdk_python_path = sdk_python_path
        self.sdk_library_path = sdk_library_path
        
        # 线程安全锁
        self._lock = threading.RLock()  # 可重入锁，支持同一线程多次获取

        # 使用 SDK 加载器加载 SDK
        self._sdk_loader = TOFSDKLoader(sdk_python_path, sdk_library_path)

        try:
            self._sdk_loader.load_sdk()
            self._camera_available = True
            logger.info("TOF SDK loaded successfully")
        except RuntimeError as e:
            logger.warning(f"TOF camera SDK not available: {e}")
            logger.warning("Camera functionality will be disabled")
            self._camera_available = False

        # 从 SDK 加载器获取类和常量
        if self._camera_available:
            sdk = self._sdk_loader.get_all_sdk_classes()
            self._MV3D_RGBD_DEVICE_INFO_LIST = sdk.get('MV3D_RGBD_DEVICE_INFO_LIST')
            self._MV3D_RGBD_FRAME_DATA = sdk.get('MV3D_RGBD_FRAME_DATA')
            self._MV3D_RGBD_CAMERA_PARAM = sdk.get('MV3D_RGBD_CAMERA_PARAM')
            self._MV3D_RGBD_PARAM = sdk.get('MV3D_RGBD_PARAM')
            self._Mv3dRgbd = sdk.get('Mv3dRgbd')
            self._DeviceType_Ethernet = sdk.get('DeviceType_Ethernet')
            self._DeviceType_USB = sdk.get('DeviceType_USB')
            self._ImageType_YUV422 = sdk.get('ImageType_YUV422')
            self._ImageType_RGB8_Planar = sdk.get('ImageType_RGB8_Planar')
            self._ImageType_Mono8 = sdk.get('ImageType_Mono8')
            self._ImageType_Depth = sdk.get('ImageType_Depth')
            self._ImageType_Rgbd = sdk.get('ImageType_Rgbd')
            self._MV3D_RGBD_OK = sdk.get('MV3D_RGBD_OK')
            self._CoordinateType_Depth = sdk.get('CoordinateType_Depth')
            self._CoordinateType_RGB = sdk.get('CoordinateType_RGB')
            self._StreamType_Depth = sdk.get('StreamType_Depth')
            self._MV3D_RGBD_IMAGE_DATA = sdk.get('MV3D_RGBD_IMAGE_DATA')

            # SDK 常量 (暴露给需要直接访问的组件)
            self.MV3D_RGBD_IMAGE_DATA = self._MV3D_RGBD_IMAGE_DATA
            self.ImageType_Depth = self._ImageType_Depth
            self.MV3D_RGBD_OK = self._MV3D_RGBD_OK
            self.CoordinateType_Depth = self._CoordinateType_Depth
            self.CoordinateType_RGB = self._CoordinateType_RGB
            self.StreamType_Depth = self._StreamType_Depth
        else:
            # Dummy values when SDK not available
            self.MV3D_RGBD_IMAGE_DATA = None
            self.ImageType_Depth = 3
            self.MV3D_RGBD_OK = 0
            self.CoordinateType_Depth = 1
            self.CoordinateType_RGB = 2
            self.StreamType_Depth = 0
        
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def initialize(self) -> bool:
        """Initialize camera (implements DepthCameraInterface)

        Raises:
            RuntimeError: Raised when TOF camera/SDK is unavailable (hard block)
        """
        if not self._camera_available:
            # During soft restart, skip hard block if SDK still unavailable
            if self._camera_status == "recovering":
                logger.warning("SDK still not available during soft restart")
                return False

            # Per config requirements, TOF camera is mandatory, hard block on unavailability (initial startup)
            error_msg = (
                "TOF camera/SDK not available - cannot continue!\n"
                "Possible causes:\n"
                "1. SDK library path incorrect or missing\n"
                "2. Camera hardware not connected\n"
                "3. SDK version incompatible\n"
                "4. Insufficient permissions\n"
                "\n"
                "Please check:\n"
                f"- SDK Python path: {self.sdk_python_path}\n"
                f"- SDK library path: {self.sdk_library_path}\n"
                "\n"
                "Per system_config.json, TOF camera is a required component."
            )
            logger.error(error_msg)
            raise RuntimeError("TOF camera is required but not available")

        # If already initialized, clean up first (for soft restart)
        if self._is_initialized:
            logger.info("Camera already initialized, cleaning up first...")
            self._cleanup_camera()
            
        try:
            # 获取设备数量
            nDeviceNum = ctypes.c_uint(0)
            ret = self._Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(
                self._DeviceType_Ethernet | self._DeviceType_USB,
                byref(nDeviceNum)
            )
            if ret != 0 or nDeviceNum.value == 0:
                raise RuntimeError(f"No TOF camera found (Error: 0x{ret:x})")

            # 获取设备列表
            self.device_list = self._MV3D_RGBD_DEVICE_INFO_LIST()
            ret = self._Mv3dRgbd.MV3D_RGBD_GetDeviceList(
                self._DeviceType_Ethernet | self._DeviceType_USB,
                pointer(self.device_list.DeviceInfo[0]), 20, byref(nDeviceNum)
            )
            if ret != 0:
                raise RuntimeError(f"Failed to get device list (Error: 0x{ret:x})")

            # 打开设备
            self._camera = self._Mv3dRgbd()
            ret = self._camera.MV3D_RGBD_OpenDevice(pointer(self.device_list.DeviceInfo[0]))
            if ret != 0:
                raise RuntimeError(f"Failed to open device (Error: 0x{ret:x})")

            # 注释掉 SDK 对齐参数设置，回退到旧版策略（使用原始深度图）
            # 旧版策略：获取原始深度图，在应用层使用内参进行 3D 坐标转换
            # 老 SDK 不支持 MV3D_RGBD_SetIntValue 接口，强制设置可能导致初始化失败
            #
            # if hasattr(self._camera, "MV3D_RGBD_SetIntValue"):
            #     try:
            #         align_value = ctypes.c_int(1)
            #         ret = self._camera.MV3D_RGBD_SetIntValue("ImageAlign", align_value)
            #         if ret == 0:
            #             logger.info("Image alignment enabled: depth aligned to RGB coordinates")
            #     except Exception as e:
            #         logger.warning(f"Error setting image alignment: {e}")
            # else:
            #     logger.debug("MV3D_RGBD_SetIntValue not available (old SDK version)")
            #
            # if hasattr(self._camera, "MV3D_RGBD_SetIntValue"):
            #     try:
            #         rgbd_output = ctypes.c_int(1)
            #         ret = self._camera.MV3D_RGBD_SetIntValue("OutputRgbd", rgbd_output)
            #         if ret == 0:
            #             logger.info("RGBD output enabled")
            #     except Exception as e:
            #         logger.warning(f"Error setting RGBD output: {e}")
            # else:
            #     logger.debug("RGBD output config not available (old SDK version)")

            logger.info("Using original depth map (no SDK alignment) - will use intrinsics for 3D conversion")

            # 开始取流（带重试机制，与旧版本保持一致）
            for attempt in range(3):  # 尝试 3 次
                ret = self._camera.MV3D_RGBD_Start()
                if ret == 0:
                    logger.info(f"Camera started successfully on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"Start attempt {attempt + 1} failed with error 0x{ret:x}")
                    if attempt < 2:
                        time.sleep(1)  # Wait 1 second before retry
            if ret != 0:
                self._camera.MV3D_RGBD_CloseDevice()
                error_msg = f"Failed to start camera after 3 attempts (Error: 0x{ret:x})"
                if ret == 0x80060005:
                    error_msg += "\nPossible solutions:"
                    error_msg += "\n1. Check if another program is using the camera"
                    error_msg += "\n2. Try running with sudo privileges"
                    error_msg += "\n3. Check if USB connection is stable"
                    error_msg += "\n4. Reconnect the camera"
                raise RuntimeError(error_msg)

            # Initialize intrinsics manager with SDK classes
            from .intrinsics import CameraIntrinsicsManager
            self._intrinsics_manager = CameraIntrinsicsManager(
                camera=self._camera,
                sdk_classes={
                    'MV3D_RGBD_CAMERA_PARAM': self._MV3D_RGBD_CAMERA_PARAM,
                    'MV3D_RGBD_OK': self._MV3D_RGBD_OK
                }
            )

            # Query Z Unit (using old SDK compatible MV3D_RGBD_GetParam method)
            self._query_z_unit()

            with self._lock:
                self._is_initialized = True
                self._camera_status = "ok"  # Initialization success, status set to ok
                self._has_ever_initialized = True  # Mark as successfully initialized at least once
            logger.info("TOF camera initialized successfully with SDK intrinsics")

            # No warmup process, consistent with old version (old version has no warmup)
            # Main loop will automatically skip empty frames, no preheating needed

            return True
            
        except RuntimeError as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
            
    def cleanup(self) -> None:
        """清理相机资源 (实现 DepthCameraInterface 接口)"""
        with self._lock:
            if self._camera_available and self._camera and self._is_initialized:
                try:
                    self._camera.MV3D_RGBD_Stop()
                    self._camera.MV3D_RGBD_CloseDevice()
                    logger.info("TOF camera cleaned up")
                except Exception as e:
                    logger.error(f"Error during camera cleanup: {e}")

            # Clear FPS tracking data to prevent mixing timestamps from different sessions
            self._frame_times.clear()
            self._last_successful_frame_time = None
    
    def fetch_frame(self, timeout: int = 5000) -> Optional[Any]:
        """获取帧数据 (阻塞, 实现 DepthCameraInterface 接口)"""
        with self._lock:
            # 检查是否在软重启中（不计入失败）
            if self._camera_status in ["soft_restarting", "recovering"]:
                logger.debug(f"Camera is {self._camera_status}, skipping frame fetch")
                return None
            
            # 检查相机可用性和初始化状态
            if not self._camera_available:
                logger.warning("Camera not available, counting as failure")
                self._handle_frame_failure()
                return None
                
            if not self._is_initialized:
                logger.warning("Camera not initialized, counting as failure")
                self._handle_frame_failure()
                return None

            try:
                frame_data = self._MV3D_RGBD_FRAME_DATA()
                ret = self._camera.MV3D_RGBD_FetchFrame(pointer(frame_data), timeout)
                if ret == 0:
                    # 成功获取帧，重置失败计数并更新 FPS 追踪
                    current_time = time.time()
                    self._consecutive_failures = 0
                    self._last_frame_time = current_time
                    self._last_successful_frame_time = current_time

                    # 更新 FPS 追踪窗口
                    self._frame_times.append(current_time)
                    if len(self._frame_times) > self._fps_window:
                        self._frame_times.pop(0)  # 保持窗口大小

                    return frame_data
                else:
                    logger.warning(f"Frame fetch failed: 0x{ret:x}")
                    self._handle_frame_failure()
                    return None
            except Exception as e:
                logger.error(f"Error fetching frame: {e}")
                self._handle_frame_failure()
                return None

    # ========== DepthCameraInterface 实现 ==========

    def get_latest_frame(self) -> Optional[Any]:
        """获取最新帧 (实现 DepthCameraInterface 接口)"""
        return self.fetch_frame()

    def get_intrinsics(self) -> Optional['CameraIntrinsicsManager']:
        """获取相机内参管理器 (实现 DepthCameraInterface 接口)"""
        return self._intrinsics_manager

    def get_camera_status(self) -> str:
        """获取相机状态 (实现 DepthCameraInterface 接口)"""
        with self._lock:
            # 优先返回特殊状态（软重启/恢复/错误）
            if self._camera_status in ["soft_restarting", "recovering", "error"]:
                return self._camera_status
            
            # 其次检查 SDK 可用性
            if not self._camera_available:
                return "unavailable"
            
            # 再检查初始化状态
            if not self._is_initialized:
                return "not_initialized"
            
            # 最后返回正常状态
            return self._camera_status

    def get_telemetry(self) -> Dict[str, Any]:
        """获取遥测数据 (实现 DepthCameraInterface 接口)"""
        with self._lock:
            # 计算帧年龄（最后一帧的时间差）
            frame_age_ms = 0.0
            if self._last_frame_time:
                frame_age_ms = (time.time() - self._last_frame_time) * 1000

            # 计算实测 FPS
            measured_fps = self._calculate_fps()

            # 确定当前使用的退避阈值（首次启动 vs 恢复）
            current_max_failures = (
                self._max_initial_failures if not self._has_ever_initialized
                else self._max_restart_failures
            )

            return {
                "camera_available": self._camera_available,
                "is_initialized": self._is_initialized,
                "camera_fps": self._camera_fps,
                "z_unit": self._z_unit,
                "status": self._camera_status,
                "consecutive_failures": self._consecutive_failures,
                "total_soft_restarts": self._total_soft_restarts,
                "total_manual_restarts": self._total_manual_restarts,
                "total_auto_restarts": self._total_auto_restarts,
                "total_reopens": self._total_reopens,
                "frame_age_ms": frame_age_ms,
                "last_successful_frame_time": self._last_successful_frame_time,
                "measured_fps": measured_fps,
                "last_restart_time": self._last_restart_time,
                "consecutive_restart_failures": self._consecutive_restart_failures,
                "max_restart_failures": current_max_failures,
                "backoff_active": self._consecutive_restart_failures >= current_max_failures,
                "has_ever_initialized": self._has_ever_initialized,
            }

    def is_initialized(self) -> bool:
        """检查相机是否已初始化 (实现 DepthCameraInterface 接口)"""
        return self._is_initialized

    def is_available(self) -> bool:
        """检查相机是否可用 (实现 DepthCameraInterface 接口)"""
        return self._camera_available

    def get_sdk_classes(self) -> Optional[Dict[str, Any]]:
        """获取 SDK 相关类和常量 (实现 DepthCameraInterface 接口)"""
        if not self._camera_available:
            return None
        return {
            'MV3D_RGBD_FRAME_DATA': self._MV3D_RGBD_FRAME_DATA,
            # 图像类型常量（完整）
            'ImageType_YUV422': self._ImageType_YUV422,
            'ImageType_RGB8_Planar': self._ImageType_RGB8_Planar,
            'ImageType_Mono8': self._ImageType_Mono8,
            'ImageType_Depth': self._ImageType_Depth,
            'ImageType_Rgbd': self._ImageType_Rgbd,
            'MV3D_RGBD_OK': self._MV3D_RGBD_OK,
        }

    # ========== 属性 (实现 DepthCameraInterface 属性) ==========

    @property
    def camera_fps(self) -> float:
        """相机实际帧率"""
        return self._camera_fps

    @camera_fps.setter
    def camera_fps(self, value: float):
        """设置相机帧率"""
        self._camera_fps = value

    @property
    def camera_available(self) -> bool:
        """相机是否可用"""
        return self._camera_available

    @property
    def intrinsics_manager(self) -> Optional['CameraIntrinsicsManager']:
        """相机内参管理器"""
        return self._intrinsics_manager

    @property
    def z_unit(self) -> float:
        """深度单位"""
        return self._z_unit if self._z_unit is not None else 1.0

    @property
    def camera(self) -> Optional[Any]:
        """原始相机对象（重写基类 property）"""
        return self._camera

    # ========== 内部方法 ==========

    def _calculate_fps(self) -> float:
        """计算实测 FPS（基于最近帧时间窗口）"""
        if len(self._frame_times) < 2:
            return 0.0

        # 计算时间跨度
        time_span = self._frame_times[-1] - self._frame_times[0]
        if time_span <= 0:
            return 0.0

        # FPS = (帧数 - 1) / 时间跨度
        return (len(self._frame_times) - 1) / time_span

    def _query_z_unit(self):
        """查询深度图的 Z 单位（mm）- 使用旧版本兼容的方法"""
        try:
            param = self._MV3D_RGBD_PARAM()
            # 使用 MV3D_RGBD_GetParam 而不是 MV3D_RGBD_GetFloatValue（老 SDK 兼容）
            ret = self._camera.MV3D_RGBD_GetParam(b"ZUnit", pointer(param))

            if ret == self._MV3D_RGBD_OK:
                # 根据 SDK 文档，ZUnit 在 ParamInfo 中
                if hasattr(param, 'ParamInfo') and hasattr(param.ParamInfo, 'stFloatParam'):
                    # fCurValue 是 c_float 类型，需转换为 Python float
                    raw_value = param.ParamInfo.stFloatParam.fCurValue
                    self._z_unit = float(raw_value)
                    logger.info(f"Depth Z-Unit queried from SDK: {self._z_unit} mm")
                elif hasattr(param, 'ParamFloatValue'):
                    # 备用字段，同样转换为 Python float
                    raw_value = param.ParamFloatValue
                    self._z_unit = float(raw_value)
                    logger.info(f"Depth Z-Unit queried from SDK: {self._z_unit} mm")
                else:
                    logger.warning("ZUnit parameter structure unexpected, using default 1.0mm")
                    self._z_unit = 1.0
            else:
                logger.warning(f"Failed to query ZUnit (error: 0x{ret:x}), using default 1.0mm")
                self._z_unit = 1.0

        except Exception as e:
            logger.warning(f"Error querying ZUnit: {e}, using default 1.0mm")
            self._z_unit = 1.0

    def _warmup_camera(self):
        """暖机相机（内部化，从 main_gui.py 迁移）- 简化版本，与旧版本保持一致"""
        if not self._camera_available or not self._is_initialized:
            logger.warning("Camera not available, skipping warmup")
            return

        from .warmup_manager import CameraWarmupManager

        # 使用更短的暖机时间（30 次尝试，~1.5 秒），避免初始化时间过长
        warmup_mgr = CameraWarmupManager(
            camera_manager=self,
            image_processor=self.image_processor,
            max_attempts=30,  # 减少到 30 次
            poll_interval=0.05  # 50ms 间隔
        )

        warmup_mgr.warmup()
    
    # ========== 软重启相关方法 ==========

    def _handle_frame_failure(self):
        """处理帧获取失败"""
        # 注意：调用者已经持有锁，所以这里不需要再获取锁

        # 短路：SDK 不可用时不累加失败，避免日志风暴
        if not self._camera_available:
            logger.debug("SDK unavailable, skipping failure accumulation (no point restarting)")
            return

        self._consecutive_failures += 1

        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.warning(f"Consecutive failures ({self._consecutive_failures}) exceeded threshold, triggering soft restart")
            self._trigger_soft_restart()
    
    def _trigger_soft_restart(self, is_manual=False):
        """触发软重启

        Args:
            is_manual: 是否手动触发
        """
        # 注意：调用者可能已经持有锁，所以只在必要时获取锁

        # 指数退避检查（只对自动重启生效）
        if not is_manual:
            # 区分首次启动和恢复场景，使用不同的阈值和退避时间
            if self._has_ever_initialized:
                # 恢复场景：曾经成功过，现在失败
                max_failures = self._max_restart_failures
                base_backoff = self._base_backoff_time
                scenario = "recovery"
            else:
                # 首次启动场景：从未成功过
                max_failures = self._max_initial_failures
                base_backoff = self._initial_backoff_time
                scenario = "initial startup"

            if self._consecutive_restart_failures >= max_failures:
                backoff_time = min(
                    base_backoff * (self._backoff_multiplier ** self._consecutive_restart_failures),
                    self._max_backoff_time
                )
                logger.warning(
                    f"Too many consecutive restart failures in {scenario} "
                    f"({self._consecutive_restart_failures}/{max_failures}), "
                    f"applying exponential backoff: {backoff_time:.1f}s"
                )
                # 使用 Timer 在后台延迟执行，避免阻塞调用线程
                timer = threading.Timer(backoff_time, self._execute_delayed_restart, args=(is_manual,))
                timer.daemon = True
                timer.start()
                return

        # 立即执行重启检查
        self._execute_restart_logic(is_manual)
    
    def _execute_delayed_restart(self, is_manual=False):
        """延迟执行重启（指数退避后）"""
        logger.info("Exponential backoff completed, proceeding with restart...")
        self._execute_restart_logic(is_manual)
    
    def _execute_restart_logic(self, is_manual=False):
        """执行重启逻辑（检查冷却时间并启动后台线程）"""
        with self._lock:
            # 检查重启冷却时间
            if self._last_restart_time:
                elapsed = time.time() - self._last_restart_time
                if elapsed < self._restart_cooldown:
                    logger.info(f"Soft restart requested but in cooldown ({elapsed:.1f}s < {self._restart_cooldown}s)")
                    return
            
            # 更新状态和统计
            self._camera_status = "soft_restarting"
            self._total_soft_restarts += 1
            
            # 区分重启类型计数
            if is_manual:
                self._total_manual_restarts += 1
                self._total_reopens += 1  # 手动重启等同于重新打开
                restart_type = "manual"
            else:
                self._total_auto_restarts += 1
                restart_type = "automatic"
            
            self._last_restart_time = time.time()
            
            logger.info(f"Starting {restart_type} soft restart #{self._total_soft_restarts}...")
            
            # 在后台线程执行软重启
            restart_thread = threading.Thread(target=self._execute_soft_restart, args=(is_manual,))
            restart_thread.daemon = True
            restart_thread.start()
    
    def _execute_soft_restart(self, is_manual=False):
        """执行软重启（在后台线程中）
        
        Args:
            is_manual: 是否手动触发
        """
        restart_success = False
        
        try:
            # 步骤1: 关闭当前相机
            logger.info("Soft restart step 1/3: Closing camera...")
            with self._lock:
                self._cleanup_camera()
            
            # 步骤2: 等待一小段时间
            time.sleep(2.0)
            
            # 步骤3: 重新初始化相机
            logger.info("Soft restart step 2/3: Reinitializing camera...")
            with self._lock:
                self._camera_status = "recovering"
            
            # 尝试重新初始化
            if self.initialize():
                logger.info("Soft restart step 3/3: Success! Camera reinitialized")
                with self._lock:
                    self._camera_status = "ok"
                    self._consecutive_failures = 0
                    # 重启成功，重置指数退避计数
                    self._consecutive_restart_failures = 0
                restart_success = True
            else:
                logger.error("Soft restart failed: Unable to reinitialize camera")
                with self._lock:
                    self._camera_status = "error"
                    # 重置失败计数，使得系统可以再次尝试软重启
                    self._consecutive_failures = 0
                    # 增加重启失败计数（只对自动重启）
                    if not is_manual:
                        self._consecutive_restart_failures += 1
                    logger.info("Reset failure count to allow future restart attempts")
                
        except Exception as e:
            logger.error(f"Soft restart error: {e}")
            with self._lock:
                self._camera_status = "error"
                self._consecutive_failures = 0  # 重置计数以允许重试
                # 增加重启失败计数（只对自动重启）
                if not is_manual:
                    self._consecutive_restart_failures += 1
        
        # 记录指数退避状态
        if not restart_success and not is_manual:
            logger.warning(
                f"Restart failure #{self._consecutive_restart_failures} "
                f"(max: {self._max_restart_failures})"
            )
    
    def _cleanup_camera(self):
        """清理相机资源（内部使用）"""
        if self._camera:
            try:
                # 停止相机
                ret = self._camera.MV3D_RGBD_Stop()
                if ret != 0:
                    logger.warning(f"Failed to stop camera: 0x{ret:x}")

                # 关闭设备
                ret = self._camera.MV3D_RGBD_CloseDevice()
                if ret != 0:
                    logger.warning(f"Failed to close device: 0x{ret:x}")

                # 释放资源
                ret = self._camera.MV3D_RGBD_Release()
                if ret != 0:
                    logger.warning(f"Failed to release camera: 0x{ret:x}")

                self._camera = None
                self._is_initialized = False
                logger.info("Camera resources cleaned up")

            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")

        # Clear FPS tracking data to prevent mixing timestamps from different sessions
        self._frame_times.clear()
        self._last_successful_frame_time = None

        # Clear depth history cache to prevent cross-session depth drift
        if hasattr(ImageDataProcessor, '_depth_history'):
            ImageDataProcessor._depth_history.clear()
            logger.debug("Cleared depth temporal filter history")
    
    def trigger_manual_restart(self):
        """手动触发软重启（公共接口）"""
        logger.info("Manual soft restart requested")
        with self._lock:
            self._trigger_soft_restart(is_manual=True)