"""
深度相机抽象接口
==============

定义所有深度相机必须实现的统一接口,支持:
- TOF 相机 (Vzense SDK)
- Intel RealSense
- OAK-D
- 模拟相机 (用于测试)

设计原则:
1. 依赖倒置原则 (Dependency Inversion Principle)
2. 开闭原则 (Open-Closed Principle)
3. 里氏替换原则 (Liskov Substitution Principle)
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class DepthCameraInterface(ABC):
    """
    深度相机统一接口

    所有深度相机实现都必须遵守此接口契约
    """

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化相机设备

        Returns:
            bool: 成功返回 True, 失败返回 False

        Raises:
            RuntimeError: 当硬件不可用或初始化失败时
        """
        pass

    @abstractmethod
    def get_latest_frame(self) -> Optional[Any]:
        """
        获取最新帧数据 (非阻塞)

        Returns:
            帧数据对象或 None (相机类型相关的帧数据结构)

        Notes:
            - 非阻塞调用,立即返回最新可用帧
            - 无可用帧时返回 None
            - 恢复模式下返回缓冲区中的最新帧
        """
        pass

    @abstractmethod
    def fetch_frame(self, timeout: int = 5000) -> Optional[Any]:
        """
        获取帧数据 (阻塞, 向后兼容)

        Args:
            timeout: 超时时间 (毫秒)

        Returns:
            帧数据对象或 None
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Optional['CameraIntrinsicsManager']:
        """
        获取相机内参管理器

        Returns:
            CameraIntrinsicsManager 实例或 None
        """
        pass

    @abstractmethod
    def get_camera_status(self) -> str:
        """
        获取相机当前状态

        Returns:
            状态字符串: "ok", "soft_restarting", "reopening",
                        "recovering", "error", "initializing", "stopped"
        """
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """
        获取相机遥测数据

        Returns:
            包含状态、统计信息的字典:
            {
                "status": str,
                "total_soft_restarts": int,
                "total_reopens": int,
                "total_usb_resets": int,
                "consecutive_failures": int,
                "frame_age_ms": float,
                ...
            }
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        清理相机资源

        Notes:
            - 停止采集线程
            - 关闭相机设备
            - 释放硬件资源
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查相机是否已初始化

        Returns:
            bool: 已初始化返回 True, 否则返回 False
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查相机是否可用

        Returns:
            bool: 可用返回 True, 否则返回 False

        Notes:
            - SDK 加载成功 + 设备连接正常 = True
            - 任一条件不满足 = False
        """
        pass

    # ========== 可选接口 (子类可选实现) ==========

    def get_sdk_classes(self) -> Optional[Dict[str, Any]]:
        """
        获取 SDK 相关类和常量 (可选)

        Returns:
            SDK 类字典或 None

        Notes:
            主要用于与特定 SDK 集成 (如 Vzense Mv3dRgbd)
        """
        return None

    def get_cached_pointcloud(self) -> Optional[Any]:
        """
        获取缓存的点云数据 (可选)

        Returns:
            点云图像数据或 None
        """
        return None

    @property
    def camera_fps(self) -> float:
        """
        相机实际帧率 (可选)

        Returns:
            float: 相机 FPS, 未计算则返回 0.0
        """
        return 0.0

    @property
    def camera_available(self) -> bool:
        """
        相机是否可用 (等同于 is_available, 但作为属性访问)

        Returns:
            bool: 可用返回 True, 否则返回 False
        """
        return self.is_available()

    @property
    def intrinsics_manager(self) -> Optional['CameraIntrinsicsManager']:
        """
        相机内参管理器 (等同于 get_intrinsics, 但作为属性访问)

        Returns:
            CameraIntrinsicsManager 实例或 None
        """
        return self.get_intrinsics()

    @property
    def z_unit(self) -> float:
        """
        深度单位 (可选, 主要用于 Vzense SDK)

        Returns:
            float: 深度单位系数, 默认 1.0
        """
        return 1.0

    @property
    def camera(self) -> Optional[Any]:
        """
        原始相机对象 (可选, 用于需要直接访问 SDK API 的场景)

        Returns:
            原始相机对象或 None

        Notes:
            - Vzense: 返回 Mv3dRgbd 实例
            - RealSense: 返回 rs.pipeline 实例
            - 尽量避免直接使用, 优先通过接口方法访问
        """
        return None


class MockDepthCamera(DepthCameraInterface):
    """
    模拟深度相机实现 (用于单元测试)

    特性:
    - 生成合成 RGB 和深度数据
    - 不依赖硬件设备
    - 可配置分辨率和内参
    """

    def __init__(self, width: int = 640, height: int = 480):
        """
        Args:
            width: RGB 宽度
            height: RGB 高度
        """
        self.width = width
        self.height = height
        self._initialized = False
        self._intrinsics = None

    def initialize(self) -> bool:
        """模拟初始化"""
        self._initialized = True
        # 创建模拟内参
        from .intrinsics import CameraIntrinsicsManager
        self._intrinsics = self._create_mock_intrinsics()
        return True

    def _create_mock_intrinsics(self):
        """创建模拟内参 (简化版)"""
        # 这里返回 None, 实际使用时需要创建完整的内参对象
        return None

    def get_latest_frame(self) -> Optional[Any]:
        """返回模拟帧数据"""
        if not self._initialized:
            return None

        # 创建合成帧数据结构
        class MockFrameData:
            def __init__(self, width, height):
                self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
                self.depth = np.full((height, width), 500.0, dtype=np.float32)

        return MockFrameData(self.width, self.height)

    def fetch_frame(self, timeout: int = 5000) -> Optional[Any]:
        """阻塞获取 (直接返回最新帧)"""
        return self.get_latest_frame()

    def get_intrinsics(self):
        """返回模拟内参"""
        return self._intrinsics

    def get_camera_status(self) -> str:
        """返回模拟状态"""
        return "ok" if self._initialized else "stopped"

    def get_telemetry(self) -> Dict[str, Any]:
        """返回空遥测"""
        return {
            "status": self.get_camera_status(),
            "total_soft_restarts": 0,
            "total_reopens": 0,
            "total_usb_resets": 0,
            "consecutive_failures": 0,
            "frame_age_ms": 0.0,
            "nodata_streak": 0
        }

    def cleanup(self) -> None:
        """清理模拟资源"""
        self._initialized = False

    def is_initialized(self) -> bool:
        """返回初始化状态"""
        return self._initialized

    def is_available(self) -> bool:
        """模拟相机始终可用"""
        return True
