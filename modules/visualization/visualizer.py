"""
增强可视化模块
用于人脸检测、距离测量、情绪识别和疲劳检测的可视化显示
"""
import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional

# 处理相对导入问题
try:
    from ..core.constants import Constants
    from ..core.logger import logger
except ImportError:
    # 如果相对导入失败，尝试创建默认值或绝对导入
    import sys
    import os
    # 将父目录添加到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from modules.core.constants import Constants
        from modules.core.logger import logger
    except ImportError:
        # 如果还是失败，创建默认的Constants类
        class Constants:
            HISTORY_DISPLAY_LENGTH = 100
            OPTIMAL_DISTANCE_RANGE = (0.4, 0.6)  # 默认最佳距离范围：40-60厘米
        
        # 创建简单的logger
        class Logger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        logger = Logger()


class EnhancedVisualizer:
    """增强可视化器 - 结合原始显示效果与新功能"""
    
    def __init__(self):
        # 初始化距离历史记录队列
        self.distance_history = deque(maxlen=Constants.HISTORY_DISPLAY_LENGTH)
        # 初始化时间历史记录队列
        self.time_history = deque(maxlen=Constants.HISTORY_DISPLAY_LENGTH)
        
    def draw_visualization(self, image: np.ndarray, results: Dict, model_info: str = "YOLOv8n-Face") -> np.ndarray:
        """
        绘制可视化效果 ，增强显示
        
        参数:
            image: 输入图像
            results: 检测结果字典
            model_info: 模型信息字符串
        
        返回:
            带有可视化效果的图像
        """
        # 如果没有检测结果，显示无检测画面
        if not results or not results.get('detection'):
            return self.draw_no_detection(image, model_info)
        
        # 提取各项检测结果
        detection = results['detection']
        distance = results.get('stable_distance')  # 稳定距离
        raw_distance = results.get('raw_distance')  # 原始距离
        depth_available = results.get('depth_available', False)  # 深度数据是否可用
        emotion = results.get('emotion')  # 情绪识别结果
        emotion_enabled = results.get('emotion_enabled', False)  # 情绪识别是否启用
        fatigue = results.get('fatigue')  # 疲劳检测结果
        fatigue_enabled = results.get('fatigue_enabled', False)  # 疲劳检测是否启用
        
        # 1. 绘制人脸边框 
        bbox = detection['bbox']
        # 深度可用时为绿色，否则为橙色
        color = (0, 255, 0) if depth_available else (0, 165, 255)
        cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
        
        # 2. 绘制眼睛位置标记 
        # 左眼：深度可用时为蓝色，否则为灰色
        left_eye_color = (255, 0, 0) if depth_available else (128, 128, 128)
        # 右眼：深度可用时为绿色，否则为灰色
        right_eye_color = (0, 255, 0) if depth_available else (128, 128, 128)
        
        # 绘制眼睛圆点
        cv2.circle(image, detection['left_eye'], 8, left_eye_color, -1)
        cv2.circle(image, detection['right_eye'], 8, right_eye_color, -1)
        
        # 眼睛标签（L表示左眼，R表示右眼）
        cv2.putText(image, "L", (detection['left_eye'][0]-15, detection['left_eye'][1]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_eye_color, 2)
        cv2.putText(image, "R", (detection['right_eye'][0]-15, detection['right_eye'][1]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_eye_color, 2)
        
        # 3. 主要距离显示 
        if distance and depth_available:
            self.draw_main_distance(image, distance)
        else:
            self.draw_no_depth_warning(image)
        
        # 4.英文界面信息面板
        self.draw_english_info_panel(image, results, model_info, raw_distance)
        
        # 5. 情绪识别结果显示
        if emotion_enabled and emotion:
            self.draw_emotion_info(image, emotion, detection)
        
        # 6. 疲劳检测结果显示
        if fatigue_enabled and fatigue and fatigue.get('enabled', False):
            self.draw_fatigue_info(image, fatigue, detection)
        
        # 7. 性能信息显示 - 保持原始风格
        self.draw_performance_info(image, results)
        
        return image
    
    def draw_main_distance(self, image: np.ndarray, distance: float):
        """
        绘制主要距离显示 - 保持原始风格
        
        参数:
            image: 输入图像
            distance: 距离值（米）
        """
        h, w = image.shape[:2]
        
        # 转换为厘米单位
        distance_cm = distance * 100
        distance_text = f"{distance_cm:.1f} cm"
        
        # 根据距离判断状态 - 保持原始逻辑
        if Constants.OPTIMAL_DISTANCE_RANGE[0] <= distance <= Constants.OPTIMAL_DISTANCE_RANGE[1]:
            color = (0, 255, 0)  # 绿色 - 最佳距离
            status = "OPTIMAL"
        elif distance < Constants.OPTIMAL_DISTANCE_RANGE[0]:
            color = (0, 165, 255)  # 橙色 - 太近
            status = "TOO CLOSE!"
        elif distance > Constants.OPTIMAL_DISTANCE_RANGE[1]:
            color = (0, 165, 255)  # 橙色 - 太远
            status = "TOO FAR!"
        else:
            color = (0, 0, 255)  # 红色 - 警告
            status = "CAUTION"
        
        # 设置字体参数
        font_scale = 3.0
        thickness = 4
        (text_width, text_height), _ = cv2.getTextSize(
            distance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 计算文本位置（居中）
        text_x = (w - text_width) // 2
        text_y = 80
        
        # 绘制背景矩形（黑色）
        cv2.rectangle(image, (text_x - 15, text_y - text_height - 15),
                     (text_x + text_width + 15, text_y + 15), (0, 0, 0), -1)
        
        # 绘制距离文本
        cv2.putText(image, distance_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 绘制状态文本
        status_text = f"[{status}]"
        cv2.putText(image, status_text, (text_x + text_width//2 - 50, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    def draw_english_info_panel(self, image: np.ndarray, results: Dict, model_info: str, raw_distance: Optional[float]):
        """
        绘制英文信息面板 - 新功能
        
        参数:
            image: 输入图像
            results: 检测结果
            model_info: 模型信息
            raw_distance: 原始深度距离
        """
        h, w = image.shape[:2]
        
        # 右上角信息面板 - 增大尺寸
        panel_x = w - 350
        panel_y = 20
        panel_width = 330
        panel_height = 200
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 面板标题 
        cv2.putText(image, "System Information", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 模型信息 
        cv2.putText(image, f"Model: {model_info}", (panel_x + 10, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # SDK信息
        cv2.putText(image, "SDK: Mv3dRgbd", (panel_x + 10, panel_y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # 坐标系信息
        cv2.putText(image, "Coordinate: RGB Aligned", (panel_x + 10, panel_y + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # 系统状态
        status_color = (0, 255, 0) if results.get('depth_available') else (0, 0, 255)
        status_text = "Status: RUNNING" if results.get('depth_available') else "Status: NO DEPTH"
        cv2.putText(image, status_text, (panel_x + 10, panel_y + 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        
        # 左下角深度信息 
        if raw_distance:
            depth_info_y = h - 80
            cv2.putText(image, f"Raw Depth: {raw_distance:.3f}m", (20, depth_info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, "Depth Status: DEPTH OK", (20, depth_info_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            depth_info_y = h - 50
            cv2.putText(image, "Depth Status: NO DEPTH", (20, depth_info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    def draw_no_depth_warning(self, image: np.ndarray):
        """
        绘制深度数据丢失警告
        
        参数:
            image: 输入图像
        """
        h, w = image.shape[:2]
        
        # 警告文本
        warning_text = "DEPTH DATA LOST"
        font_scale = 2.0
        thickness = 3
        
        # 计算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(
            warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 计算文本位置（居中）
        text_x = (w - text_width) // 2
        text_y = 80
        
        # 绘制红色背景
        cv2.rectangle(image, (text_x - 15, text_y - text_height - 15),
                     (text_x + text_width + 15, text_y + 15), (0, 0, 128), -1)
        
        # 绘制警告文本
        cv2.putText(image, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        # 提示信息
        hint_text = "Check lighting/position"
        cv2.putText(image, hint_text, (text_x - 50, text_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def draw_performance_info(self, image: np.ndarray, results: Dict):
        """
        绘制性能信息 - 增强可见性
        
        参数:
            image: 输入图像
            results: 检测结果
        """
        fps = results.get('fps', 0)
        detection = results.get('detection', {})
        
        # 第一行：FPS和检测方法 
        info_text = f"FPS: {fps:.1f} | Detector: {detection.get('method', 'Unknown')}"
        cv2.putText(image, info_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 第二行：检测置信度 
        if 'confidence' in detection:
            conf_text = f"Face Detection: {detection['confidence']:.3f}"
            cv2.putText(image, conf_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    
    def draw_emotion_info(self, image: np.ndarray, emotion: Dict, detection: Dict):
        """
        绘制情绪识别结果
        
        参数:
            image: 输入图像
            emotion: 情绪识别结果字典
            detection: 人脸检测结果字典
        """
        h, w = image.shape[:2]
        
        # 情绪信息面板位置 - 左上角 
        panel_x = 20
        panel_y = 120
        panel_width = 330
        panel_height = 180
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 面板标题 
        cv2.putText(image, "Emotion Analysis", (panel_x + 10, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 获取情绪结果
        emotion_name = emotion.get('emotion', 'Unknown')
        valence = emotion.get('valence', 0.0)  # 情绪效价（正负情绪）
        arousal = emotion.get('arousal', 0.0)  # 情绪唤醒度（激动程度）
        
        # 情绪颜色映射
        emotion_colors = {
            'Happy': (0, 255, 0),      # 绿色 - 开心
            'happy': (0, 255, 0),
            'Sad': (255, 0, 0),        # 蓝色 - 悲伤
            'sad': (255, 0, 0),
            'Anger': (0, 0, 255),      # 红色 - 愤怒
            'anger': (0, 0, 255),
            'Surprise': (0, 255, 255), # 黄色 - 惊讶
            'surprise': (0, 255, 255),
            'Fear': (128, 0, 128),     # 紫色 - 恐惧
            'fear': (128, 0, 128),
            'Disgust': (0, 128, 255),  # 橙色 - 厌恶
            'disgust': (0, 128, 255),
            'Contempt': (128, 128, 128), # 灰色 - 轻蔑
            'contempt': (128, 128, 128),
            'Neutral': (255, 255, 255), # 白色 - 中性
            'neutral': (255, 255, 255)
        }
        
        # 获取对应的颜色
        emotion_color = emotion_colors.get(emotion_name, (255, 255, 255))
        
        # 显示情绪类型 
        cv2.putText(image, f"Emotion: {emotion_name}", (panel_x + 10, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # 显示效价和唤醒度 
        cv2.putText(image, f"Valence: {valence:.2f}", (panel_x + 10, panel_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Arousal: {arousal:.2f}", (panel_x + 10, panel_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def draw_no_detection(self, image: np.ndarray, model_info: str) -> np.ndarray:
        """
        显示无人脸检测到的画面
        
        参数:
            image: 输入图像
            model_info: 模型信息
        
        返回:
            带有提示信息的图像
        """
        h, w = image.shape[:2]
        
        # 主要提示文本
        text = "No Face Detected"
        font_scale = 1.5
        thickness = 3
        
        # 计算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 计算文本位置（居中）
        text_x = (w - text_width) // 2
        text_y = h // 2
        
        # 绘制主要文本
        cv2.putText(image, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        
        # 提示信息
        hint = "Please face the camera"
        cv2.putText(image, hint, (text_x - 30, text_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 显示模型信息
        model_text = f"Model: {model_info}"
        cv2.putText(image, model_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return image
    
    def draw_fatigue_info(self, image: np.ndarray, fatigue: Dict, detection: Dict):
        """
        绘制疲劳检测结果信息面板
        
        参数:
            image: 输入图像
            fatigue: 疲劳检测结果字典
            detection: 人脸检测结果字典
        """
        h, w = image.shape[:2]
        
        # 疲劳信息面板位置 - 右下角 - 加宽以优化布局
        panel_x = w - 550
        panel_y = h - 280
        panel_width = 530
        panel_height = 260
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # 面板标题 - 大字体
        cv2.putText(image, "Fatigue Detection", (panel_x + 10, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 分隔线
        cv2.line(image, (panel_x, panel_y + 45), (panel_x + panel_width, panel_y + 45), (60, 60, 60), 1)
        
        # EAR（眼睛纵横比）信息
        ear_avg = fatigue.get('ear_avg', 0)
        eye_closed = fatigue.get('eye_closed', False)
        drowsy_alert = fatigue.get('drowsy_alert', False)
        
        # 显示EAR值
        ear_text = f"EAR: {ear_avg:.3f}"
        cv2.putText(image, ear_text, (panel_x + 15, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 眼睛状态
        eye_status = "CLOSED" if eye_closed else "OPEN"
        eye_color = (0, 0, 255) if eye_closed else (0, 255, 0)
        cv2.putText(image, f"Eyes: {eye_status}", (panel_x + 220, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
        
        # 立即疲劳警告
        if drowsy_alert:
            cv2.putText(image, "DROWSY ALERT!", (panel_x + 15, panel_y + 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # PERCLOS（闭眼时间百分比）信息
        perclos_valid = fatigue.get('perclos_valid', False)
        if perclos_valid:
            perclos = fatigue.get('perclos', 0)
            fatigue_level = fatigue.get('fatigue_level', 'Unknown')
            fatigue_color = fatigue.get('fatigue_color', (255, 255, 255))
            
            # PERCLOS数值 
            perclos_text = f"PERCLOS: {perclos:.1f}%"
            cv2.putText(image, perclos_text, (panel_x + 15, panel_y + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 绘制PERCLOS进度条 
            bar_x = panel_x + 15
            bar_y = panel_y + 170
            bar_width = 380
            bar_height = 25
            
            # 背景条
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            
            # 进度条
            progress = min(perclos / 60 * bar_width, bar_width)  # 60%作为最大显示值
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(progress), bar_y + bar_height), fatigue_color, -1)
            
            # 标记阈值线（20%，40%）
            normal_line = int(20 / 60 * bar_width)  # 正常阈值线
            mild_line = int(40 / 60 * bar_width)    # 轻度疲劳阈值线
            cv2.line(image, (bar_x + normal_line, bar_y), (bar_x + normal_line, bar_y + bar_height), (255, 255, 255), 1)
            cv2.line(image, (bar_x + mild_line, bar_y), (bar_x + mild_line, bar_y + bar_height), (255, 255, 255), 1)
            
            # 疲劳等级 - 大字体
            cv2.putText(image, fatigue_level, (panel_x + 410, panel_y + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, fatigue_color, 2)
            
            # 警告框（如果疲劳等级达到严重）
            if fatigue.get('fatigue_level_code', 0) >= 2:
                # 绘制红色边框
                cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 5)
                # 显示警告文字
                cv2.putText(image, "FATIGUE WARNING!", (w//2 - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            # 数据收集进度提示
            window_progress = fatigue.get('window_progress', 0)
            cv2.putText(image, f"Collecting data... {window_progress:.0f}%",
                       (panel_x + 15, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 0), 2)
        
        # 底部图例 - 较大字体
        cv2.putText(image, "Normal(<20%) | Mild(20-40%) | Severe(>40%)",
                   (panel_x + 15, panel_y + 235), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (180, 180, 180), 2)
        
        # FPS信息
        fps = fatigue.get('fps', 0)
        cv2.putText(image, f"FPS: {fps:.1f}", (panel_x + 420, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    
    def draw_combined_visualization(self, frame: np.ndarray, results: Dict, fatigue_detector=None) -> np.ndarray:
        """
        绘制完整的可视化界面
        
        参数:
            frame: 输入图像
            results: 处理结果字典
            fatigue_detector: 疲劳检测器实例
        
        返回:
            完整的可视化图像（仅包含带叠加信息的视频）
        """
        # 1. 首先绘制基础可视化（不含疲劳信息）
        visualization = self.draw_visualization(frame, results)
        
        # 2. 如果疲劳检测已启用且有检测器实例
        fatigue = results.get('fatigue')
        if fatigue and fatigue_detector and fatigue.get('enabled', False):
            
            # 3. 绘制严重疲劳警告框（在画面上）
            if fatigue.get('perclos_valid', False) and fatigue.get('fatigue_level_code', 0) >= 2:
                # 绘制红色边框
                cv2.rectangle(visualization, (0, 0), (visualization.shape[1], visualization.shape[0]), 
                            (0, 0, 255), 5)
                # 显示警告文字
                cv2.putText(visualization, "FATIGUE WARNING!", 
                           (visualization.shape[1]//2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        
        return visualization


# 测试代码（直接运行此文件时执行）
if __name__ == "__main__":
    print("增强可视化模块 - 测试模式")
    
    # 创建可视化器实例
    visualizer = EnhancedVisualizer()
    
    # 创建测试图像
    test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 创建测试数据
    test_results = {
        'detection': {
            'bbox': {'x1': 400, 'y1': 200, 'x2': 600, 'y2': 400},
            'left_eye': (450, 280),
            'right_eye': (550, 280),
            'confidence': 0.95,
            'method': 'YOLOv8'
        },
        'stable_distance': 0.5,
        'raw_distance': 0.48,
        'depth_available': True,
        'fps': 30.0,
        'emotion': {
            'emotion': 'Happy',
            'valence': 0.8,
            'arousal': 0.6
        },
        'emotion_enabled': True,
        'fatigue': {
            'enabled': True,
            'ear_avg': 0.25,
            'eye_closed': False,
            'drowsy_alert': False,
            'perclos_valid': True,
            'perclos': 15.0,
            'fatigue_level': 'Normal',
            'fatigue_level_code': 0,
            'fatigue_color': (0, 255, 0),
            'window_progress': 100,
            'fps': 30.0
        },
        'fatigue_enabled': True
    }
    
    # 绘制可视化
    result_image = visualizer.draw_visualization(test_image, test_results)
    
    # 显示结果
    cv2.imshow('Enhanced Visualizer Test', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("测试完成！")