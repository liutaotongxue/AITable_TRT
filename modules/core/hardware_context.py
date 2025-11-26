"""
硬件上下文模块 - 管理全局硬件资源和配置
由预检阶段初始化，提供给各模块使用
"""
import torch
from typing import Optional, Dict


class HardwareContext:
    """
    硬件上下文 - 单例模式
    存储预检阶段验证的硬件信息和配置
    """

    _instance: Optional['HardwareContext'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化硬件上下文（单例）"""
        if self._initialized:
            return

        # 硬件状态
        self._device: Optional[torch.device] = None
        self._cuda_available = False
        self._cudnn_available = False

        # GPU信息
        self._gpu_name: Optional[str] = None
        self._gpu_memory: Optional[float] = None
        self._cuda_version: Optional[str] = None

        # 模型路径
        self._model_paths: Dict[str, str] = {}

        # SDK路径
        self._sdk_paths: Dict[str, str] = {}

        # 性能配置
        self._performance_config: Dict = {}

        self._initialized = True

    def initialize_from_preflight(self,
                                  device: torch.device,
                                  cuda_available: bool,
                                  cudnn_available: bool,
                                  gpu_name: str = None,
                                  gpu_memory: float = None,
                                  cuda_version: str = None,
                                  model_paths: Dict[str, str] = None,
                                  sdk_paths: Dict[str, str] = None,
                                  performance_config: Dict = None):
        """
        从预检结果初始化硬件上下文

        Args:
            device: torch.device 对象
            cuda_available: CUDA是否可用
            cudnn_available: CUDNN是否可用
            gpu_name: GPU名称
            gpu_memory: GPU内存(GB)
            cuda_version: CUDA版本
            model_paths: 模型路径字典
            sdk_paths: SDK路径字典
            performance_config: 性能配置
        """
        self._device = device
        self._cuda_available = cuda_available
        self._cudnn_available = cudnn_available
        self._gpu_name = gpu_name
        self._gpu_memory = gpu_memory
        self._cuda_version = cuda_version
        self._model_paths = model_paths or {}
        self._sdk_paths = sdk_paths or {}
        self._performance_config = performance_config or {}

    @property
    def device(self) -> torch.device:
        """获取torch设备对象"""
        if self._device is None:
            raise RuntimeError("HardwareContext not initialized. Run preflight check first.")
        return self._device

    @property
    def cuda_available(self) -> bool:
        """CUDA是否可用"""
        return self._cuda_available

    @property
    def cudnn_available(self) -> bool:
        """CUDNN是否可用"""
        return self._cudnn_available

    @property
    def gpu_name(self) -> str:
        """GPU名称"""
        return self._gpu_name or "Unknown"

    @property
    def gpu_memory(self) -> float:
        """GPU内存(GB)"""
        return self._gpu_memory or 0.0

    @property
    def cuda_version(self) -> str:
        """CUDA版本"""
        return self._cuda_version or "Unknown"

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        获取模型路径

        Args:
            model_name: 模型名称 (如 'yolo_face', 'emonet')

        Returns:
            模型路径，如果未找到返回None
        """
        return self._model_paths.get(model_name)

    def get_sdk_path(self, sdk_name: str) -> Optional[str]:
        """
        获取SDK路径

        Args:
            sdk_name: SDK名称 (如 'python', 'lib_aarch64')

        Returns:
            SDK路径，如果未找到返回None
        """
        return self._sdk_paths.get(sdk_name)

    def get_performance_config(self, key: str, default=None):
        """
        获取性能配置

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self._performance_config.get(key, default)

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._device is not None

    def summary(self) -> str:
        """返回硬件上下文摘要"""
        if not self.is_initialized():
            return "HardwareContext: Not initialized"

        return (
            f"HardwareContext Summary:\n"
            f"  Device: {self._device}\n"
            f"  CUDA Available: {self._cuda_available}\n"
            f"  CUDNN Available: {self._cudnn_available}\n"
            f"  GPU: {self.gpu_name} ({self.gpu_memory:.1f}GB)\n"
            f"  CUDA Version: {self.cuda_version}\n"
            f"  Model Paths: {len(self._model_paths)} configured\n"
            f"  SDK Paths: {len(self._sdk_paths)} configured"
        )

    def reset(self):
        """重置上下文（主要用于测试）"""
        self._device = None
        self._cuda_available = False
        self._cudnn_available = False
        self._gpu_name = None
        self._gpu_memory = None
        self._cuda_version = None
        self._model_paths = {}
        self._sdk_paths = {}
        self._performance_config = {}


# 全局单例实例
hardware_context = HardwareContext()


def get_hardware_context() -> HardwareContext:
    """获取全局硬件上下文实例"""
    return hardware_context
