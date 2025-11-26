"""
TOF SDK 加载器
==============

职责:
- 动态加载 Vzense Mv3dRgbd SDK 模块
- 临时修改环境变量 (sys.path, LD_LIBRARY_PATH)
- 加载完成后恢复环境
- 缓存 SDK 类和常量供后续使用

单一职责: SDK 环境准备和模块导入
"""
import os
import sys
from typing import Optional, Dict, Any
from ..core.logger import logger


class TOFSDKLoader:
    """
    TOF SDK 加载器

    设计特点:
    1. 环境变量隔离 - 加载完成后恢复原始环境
    2. 异常安全 - 失败时自动清理环境修改
    3. 单例模式 - SDK 只需加载一次
    """

    def __init__(self, sdk_python_path: Optional[str] = None,
                 sdk_library_path: Optional[str] = None):
        """
        Args:
            sdk_python_path: SDK Python 模块路径
            sdk_library_path: SDK 动态库路径 (.so 文件)
        """
        self.sdk_python_path = sdk_python_path
        self.sdk_library_path = sdk_library_path

        # SDK 加载状态
        self.sdk_loaded = False

        # 环境变量备份
        self._original_ld_library_path: Optional[str] = None
        self._sys_path_modified = False

        # 缓存 SDK 类和常量
        self.sdk_classes: Dict[str, Any] = {}

    def load_sdk(self) -> bool:
        """
        动态加载 SDK 模块

        Returns:
            bool: 成功返回 True, 失败返回 False

        Raises:
            RuntimeError: SDK 路径不存在或导入失败时
        """
        if self.sdk_loaded:
            logger.debug("SDK already loaded, skipping")
            return True

        if not self.sdk_python_path or not self.sdk_library_path:
            raise RuntimeError(
                "SDK paths not provided\n"
                "Please configure 'sdk_paths' in system_config.json or hardware context"
            )

        try:
            # 验证路径存在
            if not os.path.exists(self.sdk_python_path):
                raise RuntimeError(f"SDK python path not found: {self.sdk_python_path}")
            if not os.path.exists(self.sdk_library_path):
                raise RuntimeError(f"SDK library path not found: {self.sdk_library_path}")

            # 保存原始环境
            self._backup_environment()

            # 临时修改环境变量
            self._modify_environment()

            # 动态导入 SDK 模块
            self._import_sdk_modules()

            # 恢复环境变量 (加载成功后立即恢复)
            self._restore_environment()

            self.sdk_loaded = True
            logger.info(f"SDK modules loaded successfully from {self.sdk_python_path}")
            return True

        except ImportError as e:
            # 失败时也要恢复环境
            self._restore_environment()
            raise RuntimeError(
                f"Failed to import SDK modules from {self.sdk_python_path}\n"
                f"Error: {e}\n"
                f"Please check:\n"
                f"1. SDK Python path exists: {self.sdk_python_path}\n"
                f"2. Mv3dRgbdImport module is in the path\n"
                f"3. SDK library path exists: {self.sdk_library_path}"
            )
        except Exception as e:
            # 失败时也要恢复环境
            self._restore_environment()
            raise RuntimeError(f"Failed to load SDK modules: {e}")

    def _backup_environment(self) -> None:
        """备份原始环境变量"""
        self._original_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

    def _modify_environment(self) -> None:
        """临时修改环境变量"""
        # 临时添加 SDK Python 路径
        if self.sdk_python_path not in sys.path:
            sys.path.insert(0, self.sdk_python_path)
            self._sys_path_modified = True

        # 临时设置动态库路径
        if self.sdk_library_path not in self._original_ld_library_path:
            os.environ['LD_LIBRARY_PATH'] = (
                f"{self.sdk_library_path}:{self._original_ld_library_path}"
            )

    def _import_sdk_modules(self) -> None:
        """
        动态导入 SDK 模块并缓存类/常量

        Notes:
            type: ignore 注释用于忽略 mypy 类型检查
            (SDK 模块是动态加载的, 不在类型检查范围内)
        """
        # 导入 SDK 定义模块
        from Mv3dRgbdImport.Mv3dRgbdDefine import (  # type: ignore
            MV3D_RGBD_DEVICE_INFO_LIST,
            MV3D_RGBD_FRAME_DATA,
            MV3D_RGBD_CAMERA_PARAM,
            MV3D_RGBD_PARAM,
            MV3D_RGBD_IMAGE_DATA,
            DeviceType_Ethernet,
            DeviceType_USB,
            ImageType_YUV422,
            ImageType_RGB8_Planar,
            ImageType_Mono8,
            ImageType_Depth,
            ImageType_Rgbd,
            MV3D_RGBD_OK,
            ParamType_Float,
            ParamType_Int,
            MV3D_RGBD_FLOAT_Z_UNIT,
            StreamType_Depth,
            CoordinateType_Depth,
            CoordinateType_RGB
        )
        from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd  # type: ignore

        # 缓存 SDK 类和常量
        self.sdk_classes = {
            # 数据结构
            'MV3D_RGBD_DEVICE_INFO_LIST': MV3D_RGBD_DEVICE_INFO_LIST,
            'MV3D_RGBD_FRAME_DATA': MV3D_RGBD_FRAME_DATA,
            'MV3D_RGBD_CAMERA_PARAM': MV3D_RGBD_CAMERA_PARAM,
            'MV3D_RGBD_PARAM': MV3D_RGBD_PARAM,
            'MV3D_RGBD_IMAGE_DATA': MV3D_RGBD_IMAGE_DATA,

            # API 类
            'Mv3dRgbd': Mv3dRgbd,

            # 设备类型
            'DeviceType_Ethernet': DeviceType_Ethernet,
            'DeviceType_USB': DeviceType_USB,

            # 图像类型
            'ImageType_YUV422': ImageType_YUV422,
            'ImageType_RGB8_Planar': ImageType_RGB8_Planar,
            'ImageType_Mono8': ImageType_Mono8,
            'ImageType_Depth': ImageType_Depth,
            'ImageType_Rgbd': ImageType_Rgbd,

            # 返回码
            'MV3D_RGBD_OK': MV3D_RGBD_OK,

            # 参数类型
            'ParamType_Float': ParamType_Float,
            'ParamType_Int': ParamType_Int,
            'MV3D_RGBD_FLOAT_Z_UNIT': MV3D_RGBD_FLOAT_Z_UNIT,

            # 坐标系统
            'StreamType_Depth': StreamType_Depth,
            'CoordinateType_Depth': CoordinateType_Depth,
            'CoordinateType_RGB': CoordinateType_RGB
        }

    def _restore_environment(self) -> None:
        """
        恢复原始环境变量

        Notes:
            正常加载成功时环境已在 load_sdk() 中恢复,
            此方法主要用于异常情况下的清理
        """
        # 恢复 sys.path
        if self._sys_path_modified and self.sdk_python_path in sys.path:
            try:
                sys.path.remove(self.sdk_python_path)
                self._sys_path_modified = False
            except ValueError:
                pass

        # 恢复 LD_LIBRARY_PATH
        if self._original_ld_library_path is not None:
            os.environ['LD_LIBRARY_PATH'] = self._original_ld_library_path
            self._original_ld_library_path = None

    def get_sdk_class(self, class_name: str) -> Optional[Any]:
        """
        获取缓存的 SDK 类或常量

        Args:
            class_name: 类名或常量名 (如 'Mv3dRgbd', 'MV3D_RGBD_OK')

        Returns:
            SDK 类/常量或 None

        Example:
            >>> loader = TOFSDKLoader(...)
            >>> loader.load_sdk()
            >>> Mv3dRgbd = loader.get_sdk_class('Mv3dRgbd')
        """
        if not self.sdk_loaded:
            logger.warning(f"SDK not loaded, cannot get class '{class_name}'")
            return None
        return self.sdk_classes.get(class_name)

    def get_all_sdk_classes(self) -> Dict[str, Any]:
        """
        获取所有缓存的 SDK 类和常量

        Returns:
            SDK 类字典 (class_name -> class/constant)
        """
        return self.sdk_classes.copy()

    def is_loaded(self) -> bool:
        """
        检查 SDK 是否已加载

        Returns:
            bool: 已加载返回 True, 否则返回 False
        """
        return self.sdk_loaded

    def cleanup(self) -> None:
        """
        清理 SDK 加载器资源

        Notes:
            主要用于恢复环境变量 (通常已在加载时恢复, 此方法用于保险)
        """
        self._restore_environment()
