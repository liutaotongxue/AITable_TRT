"""
日志配置模块 - 从 system_config.json 读取配置
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = 'AITable',
    level: Optional[int] = None,
    log_dir: Optional[str] = None,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None,
    max_size_mb: int = 100,
    file_rotation: str = 'daily'
) -> logging.Logger:
    """
    设置日志配置（从 system_config.json 读取参数）

    优先级：
    1. 环境变量 AITABLE_LOG_LEVEL
    2. 参数传入
    3. system_config.json
    4. 默认值

    Args:
        name: 日志名称
        level: 日志级别（None 时从配置读取）
        log_dir: 日志目录（None 时从配置读取）
        enable_console: 是否启用控制台（None 时从配置读取）
        enable_file: 是否启用文件（None 时从配置读取）
        max_size_mb: 最大文件大小（MB）
        file_rotation: 文件轮转策略（'daily' 或 'size'）

    Returns:
        logger: 配置好的日志对象
    """
    # 尝试从 system_config.json 读取配置
    config = None
    try:
        # 延迟导入避免循环依赖
        from modules.core.config_loader import get_config

        # 尝试多个配置文件路径
        config_paths = [
            Path.cwd() / "system_config.json",           # 当前目录
            Path(__file__).parent.parent.parent / "system_config.json",  # 项目根目录
            Path(__file__).parent.parent.parent / "config" / "system_config.json",  # config/目录
        ]

        for config_path in config_paths:
            if config_path.exists():
                config = get_config(config_path=config_path)
                break
    except Exception as e:
        # 配置加载失败时使用默认值（不影响日志系统启动）
        print(f"Warning: 无法加载 system_config.json，使用默认日志配置: {e}")

    # 应用配置（优先级：参数 > 环境变量 > 配置文件 > 默认值）
    if level is None:
        # 环境变量优先级最高
        env_level = os.getenv("AITABLE_LOG_LEVEL")
        if env_level:
            level = getattr(logging, env_level.upper(), logging.INFO)
        elif config and hasattr(config, 'logging'):
            level_str = getattr(config.logging, 'level', 'INFO')
            level = getattr(logging, level_str.upper(), logging.INFO)
        else:
            level = logging.INFO

    if log_dir is None:
        if config and hasattr(config, 'paths'):
            log_dir = getattr(config.paths, 'logs_dir', 'logs')
        else:
            log_dir = 'logs'

    if enable_console is None:
        if config and hasattr(config, 'logging'):
            enable_console = getattr(config.logging, 'enable_console', True)
        else:
            enable_console = True

    if enable_file is None:
        if config and hasattr(config, 'logging'):
            enable_file = getattr(config.logging, 'enable_file', True)
        else:
            enable_file = True

    # 从配置读取轮转参数
    if config and hasattr(config, 'logging'):
        file_rotation = getattr(config.logging, 'file_rotation', 'daily')
        max_size_mb = getattr(config.logging, 'max_size_mb', 100)

    # 创建日志目录
    log_path = Path(log_dir)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 如果logger已有handler，则不重复添加
    if logger.handlers:
        return logger

    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件handler
    if enable_file:
        if file_rotation == 'daily':
            # 每日轮转（基于文件名）
            log_file = log_path / f'{name}_{datetime.now().strftime("%Y%m%d")}.log'
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        else:
            # 大小轮转
            log_file = log_path / f'{name}.log'
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 创建默认logger（懒加载，从配置文件读取参数）
logger = setup_logger()
