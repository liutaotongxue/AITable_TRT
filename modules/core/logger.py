"""
日志配置模块
"""
import logging
from datetime import datetime
import os

def setup_logger(name='AITable', level=logging.INFO, log_dir='logs'):
    """
    设置日志配置
    
    Args:
        name: 日志名称
        level: 日志级别
        log_dir: 日志目录
    
    Returns:
        logger: 配置好的日志对象
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 创建默认logger
logger = setup_logger()