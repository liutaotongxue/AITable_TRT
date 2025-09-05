"""
系统常量配置
"""

class Constants:
    """系统常量"""
    # 距离配置
    OPTIMAL_DISTANCE_RANGE = (0.3, 0.5)  # 最佳距离范围（米）
    WARNING_DISTANCE_MIN = 0.2  # 最小警告距离
    WARNING_DISTANCE_MAX = 0.8  # 最大警告距离
    
    # 检测配置
    FACE_CONFIDENCE_THRESHOLD = 0.5  # 人脸检测置信度阈值
    SMOOTHING_WINDOW = 10  # 平滑窗口大小
    
    # 深度配置
    DEPTH_WINDOW_SIZES = [3, 5, 7]  # 深度采样窗口大小
    MIN_VALID_DEPTH_RATIO = 0.3  # 最小有效深度比例
    DEPTH_STD_THRESHOLD = 10  # 深度标准差阈值（mm）
    
    # 显示配置
    DISPLAY_UPDATE_INTERVAL = 0.1  # 显示更新间隔
    HISTORY_DISPLAY_LENGTH = 50  # 历史记录显示长度
    
    # 情绪识别配置
    EMOTION_CONFIDENCE_THRESHOLD = 0.3  # 情绪识别置信度阈值
    EMOTION_SMOOTHING_WINDOW = 15  # 情绪平滑窗口大小
    
    # 疲劳检测配置
    EAR_THRESHOLD = 0.21  # 眼部纵横比阈值
    PERCLOS_THRESHOLD = 0.4  # PERCLOS阈值
    CONSECUTIVE_FRAMES = 15  # 连续帧数阈值
    FATIGUE_WINDOW_SIZE = 30  # 疲劳检测窗口大小