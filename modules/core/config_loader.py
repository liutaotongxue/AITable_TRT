"""
统一配置加载器 - 使用 dataclass 实现轻量级配置管理
=====================================================

此模块负责：
1. 从 config/system_config.json 读取并验证所有配置
2. 提供类型安全的配置访问接口
3. 统一管理路径解析逻辑
4. 消除 constants.py、main_gui.py、preflight_check.py 之间的配置重复

配置文件位置：
- 默认: config/system_config.json
- 环境变量: AITABLE_CONFIG=path/to/config.json
- 参数指定: load_config(config_path="path/to/config.json")

使用示例：
```python
from modules.core.config_loader import get_config

config = get_config()  # 单例模式
print(config.hardware.cuda.device_id)
print(config.models["yolo_face"]["primary"])
print(config.algorithm.fatigue.ear_threshold)
```
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


# ============================================================================
# 简化配置类（不依赖 Pydantic）
# ============================================================================

class DictConfig:
    """字典风格的配置基类"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # 递归转换嵌套字典
                setattr(self, key, DictConfig(**value))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # 转换字典列表
                setattr(self, key, [DictConfig(**item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        """字典风格的 get 方法"""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """支持 config["key"] 语法"""
        return getattr(self, key)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"{self.__class__.__name__}({attrs})"


class AlgorithmConfig(DictConfig):
    """算法配置（fatigue, emotion, camera）"""
    pass


class SystemConfig(DictConfig):
    """
    系统配置（顶层）

    属性:
        system: 系统信息
        environment: 环境配置
        paths: 路径配置
        models: 模型配置
        hardware: 硬件配置
        performance: 性能配置
        logging: 日志配置
        pose_angles: 姿态角度配置
        preflight: 预检配置
        algorithm: 算法配置
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 注入特殊方法
        self._config_path: Optional[Path] = None

    def set_config_path(self, path: Path) -> None:
        """设置配置文件路径（用于相对路径解析）"""
        self._config_path = path

    def resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        """解析相对/绝对路径"""
        if not path_str:
            return None

        if not self._config_path:
            raise RuntimeError("配置文件路径未设置，无法解析相对路径")

        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self._config_path.parent / candidate

        return candidate if candidate.exists() else None

    def resolve_model_path(self, model_name: str) -> Optional[Path]:
        """
        解析模型路径（优先级：primary > fallback）

        特殊校验：
        - yolo_pose: 商业版本强制要求 .engine 文件，不支持 .pt 回退

        Args:
            model_name: 模型名称（yolo_face, emonet, emonet_trt, yolo_pose）

        Returns:
            Optional[Path]: 解析后的模型路径，若未找到或校验失败则返回 None

        Raises:
            RuntimeError: 当 yolo_pose 模型不是 .engine 文件时
        """
        model_cfg = self.models.get(model_name)
        if not model_cfg:
            return None

        # 尝试 primary
        primary = getattr(model_cfg, "primary", None)
        if primary:
            resolved = self.resolve_path(primary)
            if resolved:
                # 特殊校验：yolo_pose 必须是 .engine 文件
                if model_name == "yolo_pose" and not str(resolved).endswith(".engine"):
                    raise RuntimeError(
                        f"姿态检测模型必须是 TensorRT .engine 文件，当前: {resolved}\n"
                        f"商业版本不支持 .pt 回退，请更新 system_config.json:\n"
                        f'  "yolo_pose": {{"primary": "models/yolov8n-pose_fp16.engine", "required": true}}'
                    )
                return resolved

        # 尝试 fallback
        fallbacks = getattr(model_cfg, "fallback", [])
        for fallback in fallbacks:
            resolved = self.resolve_path(fallback)
            if resolved:
                # 同样的校验逻辑
                if model_name == "yolo_pose" and not str(resolved).endswith(".engine"):
                    raise RuntimeError(
                        f"姿态检测模型必须是 TensorRT .engine 文件，当前: {resolved}\n"
                        f"商业版本不支持 .pt 回退"
                    )
                return resolved

        return None

    def resolve_sdk_path(self, sdk_key: str) -> Optional[Path]:
        """解析SDK路径"""
        attr_map = {
            "python": "sdk_python_path",
            "lib_aarch64": "sdk_lib_path_aarch64",
            "lib_linux64": "sdk_lib_path_linux64",
        }

        attr_name = attr_map.get(sdk_key)
        if not attr_name:
            return None

        path_str = getattr(self.paths, attr_name, None)
        return self.resolve_path(path_str)


# ============================================================================
# 配置加载器（单例模式）
# ============================================================================

_config_instance: Optional[SystemConfig] = None


def load_config(config_path: Optional[str | Path] = None) -> SystemConfig:
    """
    Load configuration from JSON file

    Args:
        config_path: Configuration file path (default: auto-detect)
                    Search order: parameter > env AITABLE_CONFIG > root/system_config.json > config/system_config.json

    Returns:
        SystemConfig: Configuration object

    Raises:
        FileNotFoundError: Configuration file does not exist
        ValueError: Configuration file format error
    """
    if config_path is None:
        # Search order: env var > root dir > config/ subdir
        # Supports environment variable AITABLE_CONFIG override
        config_env = os.getenv("AITABLE_CONFIG")
        if config_env:
            config_path = Path(config_env)
        else:
            # Try root directory first, then config/ subdirectory
            root_dir = Path(__file__).parent.parent.parent
            root_config = root_dir / "system_config.json"
            config_config = root_dir / "config" / "system_config.json"

            if root_config.exists():
                config_path = root_config
            elif config_config.exists():
                config_path = config_config
            else:
                # Default to root path (will raise FileNotFoundError below)
                config_path = root_config
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Configuration file JSON parsing failed: {e}") from e

    # 确保 algorithm 存在（如果 JSON 中未定义）
    if "algorithm" not in raw_data:
        raw_data["algorithm"] = {
            "fatigue": {
                "ear_threshold": 0.15,
                "perclos_window": 30,
                "perclos_normal": 20,
                "perclos_mild": 40,
                "fps": 3.7,
            },
            "emotion": {
                "confidence_threshold": 0.3,
                "smoothing_window": 15,
            },
            "camera": {
                "rgb_resolution": [1280, 1024],
                "depth_resolution": [1280, 1024],
                "min_valid_depth": 200,
                "max_valid_depth": 1500,
                "color_exposure_mode": 1,
                "color_brightness": 110,
                "color_gamma": 0.6,
            },
        }

    # 确保 target_person 存在并有 enabled 字段（默认 False）
    if "target_person" not in raw_data:
        raw_data["target_person"] = {"enabled": False}
    elif "enabled" not in raw_data["target_person"]:
        raw_data["target_person"]["enabled"] = False

    # 确保 debug 存在（如果 JSON 中未定义）
    if "debug" not in raw_data:
        raw_data["debug"] = {"target_overlay": False}

    config = SystemConfig(**raw_data)
    config.set_config_path(config_path)
    return config


def get_config(config_path: Optional[str | Path] = None, reload: bool = False) -> SystemConfig:
    """
    获取配置单例（懒加载）

    Args:
        config_path: 配置文件路径（仅首次加载时有效）
        reload: 是否强制重新加载配置

    Returns:
        SystemConfig: 配置单例
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = load_config(config_path)

    return _config_instance


# ============================================================================
# 环境变量覆盖支持
# ============================================================================

def apply_env_overrides(config: SystemConfig) -> SystemConfig:
    """
    应用环境变量覆盖（优先级：ENV > system_config.json > 默认值）

    支持的环境变量：
    - AITABLE_LOG_LEVEL: 日志级别
    - AITABLE_POSE_INTERVAL: 姿态检测间隔
    - AITABLE_DEBUG_DEPTH: 深度调试模式
    - AITABLE_USE_DEBUG_CONFIG: 使用调试配置

    Args:
        config: 原始配置对象

    Returns:
        应用环境变量后的配置对象
    """
    # 日志级别
    if log_level := os.getenv("AITABLE_LOG_LEVEL"):
        config.logging.level = log_level.upper()

    # 姿态检测间隔
    if pose_interval := os.getenv("AITABLE_POSE_INTERVAL"):
        try:
            config.pose_angles.pose_interval = int(pose_interval)
        except ValueError:
            pass

    return config


# ============================================================================
# 向后兼容接口
# ============================================================================

def get_algorithm_constants() -> Dict[str, Any]:
    """
    获取算法常量字典（兼容旧代码）

    Returns:
        包含所有算法常量的字典，可用于替换 Constants 类
    """
    config = get_config()

    return {
        # 疲劳检测
        "EAR_THRESHOLD": config.algorithm.fatigue.ear_threshold,
        "PERCLOS_WINDOW": config.algorithm.fatigue.perclos_window,
        "PERCLOS_NORMAL": config.algorithm.fatigue.perclos_normal,
        "PERCLOS_MILD": config.algorithm.fatigue.perclos_mild,
        "FPS": config.algorithm.fatigue.fps,

        # 情绪识别
        "EMOTION_CONFIDENCE_THRESHOLD": config.algorithm.emotion.confidence_threshold,
        "EMOTION_SMOOTHING_WINDOW": config.algorithm.emotion.smoothing_window,

        # 相机
        "RGB_RESOLUTION": tuple(config.algorithm.camera.rgb_resolution),
        "DEPTH_RESOLUTION": tuple(config.algorithm.camera.depth_resolution),
        "MIN_VALID_DEPTH": config.algorithm.camera.min_valid_depth,
        "MAX_VALID_DEPTH": config.algorithm.camera.max_valid_depth,

        # 日志
        "LOG_LEVEL": config.logging.level,
    }


if __name__ == "__main__":
    # 测试配置加载
    import sys

    try:
        config = get_config()
        print("Config loaded successfully")
        print(f"  System version: {config.system.version}")
        print(f"  Platform: {config.system.platform}")
        print(f"  CUDA device ID: {config.hardware.cuda.device_id}")
        print(f"  Log level: {config.logging.level}")
        print(f"  EAR threshold: {config.algorithm.fatigue.ear_threshold}")
        print(f"  PERCLOS window: {config.algorithm.fatigue.perclos_window}s")

        # 测试路径解析
        yolo_path = config.resolve_model_path("yolo_face")
        print(f"  YOLO model path: {yolo_path}")

    except Exception as e:
        print(f"Config loading failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
