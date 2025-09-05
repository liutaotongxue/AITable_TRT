"""
增强版疲劳检测器模块 - 集成EAR和PERCLOS双重标准
包含卡内基梅隆研究所PERCLOS黄金标准实现
适配TOF相机的RGB数据流
"""
import cv2
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
from typing import Dict, Optional, Tuple
from ..core.constants import Constants
from ..core.logger import logger

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available for fatigue detection")
    MEDIAPIPE_AVAILABLE = False


class FatigueDetector:
    """
    增强版疲劳检测器 - 集成EAR和PERCLOS算法
    专门适配TOF相机的RGB数据流
    """
    
    def __init__(self, perclos_window=30, fps=30):
        """
        初始化检测器
        Args:
            perclos_window: PERCLOS计算时间窗口（秒），默认30秒
            fps: 预期帧率，用于计算窗口大小
        """
        self.enabled = MEDIAPIPE_AVAILABLE
        
        if not self.enabled:
            logger.warning("MediaPipe不可用，疲劳检测功能将被禁用")
            return
            
        # MediaPipe初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 眼部关键点 - 使用quick_start.py中验证的索引
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # EAR参数 - 使用更准确的阈值
        self.EAR_THRESHOLD = 0.15  # 眼睛闭合阈值（对应80%闭合）
        self.EAR_CONSEC_FRAMES = 10  # 连续帧数阈值
        
        # PERCLOS参数
        self.PERCLOS_WINDOW = perclos_window  # 时间窗口（秒）
        self.FPS = fps  # 预期帧率
        self.WINDOW_SIZE = self.PERCLOS_WINDOW * self.FPS  # 窗口帧数
        
        # PERCLOS疲劳等级阈值（卡内基梅隆标准）
        self.PERCLOS_NORMAL = 20      # 正常阈值 < 20%
        self.PERCLOS_MILD = 40        # 轻度疲劳 20%-40%
        # 严重疲劳 > 40%
        
        # 状态追踪
        self.frame_count = 0
        self.drowsy_frames = 0
        self.eye_closed_frames = 0
        
        # PERCLOS计算缓冲区（使用滑动窗口）
        self.eye_state_buffer = deque(maxlen=self.WINDOW_SIZE)
        self.timestamp_buffer = deque(maxlen=self.WINDOW_SIZE)
        
        # 统计数据
        self.perclos_history = deque(maxlen=100)  # 保存历史PERCLOS值
        self.ear_history = deque(maxlen=100)       # 保存历史EAR值
        
        # 初始化时间
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        logger.info(f"Enhanced Fatigue detector initialized: PERCLOS window={perclos_window}s")
        logger.info(f"PERCLOS阈值: Normal<{self.PERCLOS_NORMAL}%, Mild:{self.PERCLOS_NORMAL}-{self.PERCLOS_MILD}%, Severe>{self.PERCLOS_MILD}%")
    
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        计算眼睛纵横比（EAR）
        使用与quick_start.py相同的算法
        """
        if len(eye_points) < 6:
            return 0.0
            
        # 垂直距离
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        # 水平距离
        C = distance.euclidean(eye_points[0], eye_points[3])
        # EAR计算
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def calculate_perclos(self) -> Tuple[float, bool]:
        """
        计算PERCLOS值
        Returns:
            perclos: 眼睛闭合百分比 (0-100)
            valid: 是否有足够的数据进行计算
        """
        if len(self.eye_state_buffer) < self.FPS * 5:  # 至少需要5秒数据
            return 0, False
        
        # 计算闭眼帧数占比
        closed_count = sum(self.eye_state_buffer)
        total_count = len(self.eye_state_buffer)
        perclos = (closed_count / total_count) * 100 if total_count > 0 else 0
        
        return perclos, True
    
    def get_fatigue_level(self, perclos: float) -> Tuple[int, str, Tuple[int, int, int]]:
        """
        根据PERCLOS值判定疲劳等级
        Args:
            perclos: PERCLOS百分比值
        Returns:
            level: 疲劳等级 (0:正常, 1:轻度, 2:严重)
            description: 等级描述
            color: 显示颜色 (BGR)
        """
        if perclos < self.PERCLOS_NORMAL:
            return 0, "Normal", (0, 255, 0)  # 绿色
        elif perclos < self.PERCLOS_MILD:
            return 1, "Mild Fatigue", (0, 165, 255)  # 橙色
        else:
            return 2, "Severe Fatigue", (0, 0, 255)  # 红色
    
    def detect_fatigue(self, frame: np.ndarray) -> Dict:
        """
        检测疲劳状态 - 使用TOF相机的RGB数据
        Args:
            frame: TOF相机的RGB图像帧
        Returns:
            status: 包含检测结果的字典
        """
        current_time = time.time()
        
        # 初始化返回状态
        status = {
            'fatigue_level': "Calculating...",
            'fatigue_level_code': 0,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'perclos': 0.0,
            'perclos_valid': False,
            'eye_closed': False,
            'drowsy_alert': False,
            'alarm': False,
            'enabled': self.enabled,
            'face_detected': False,
            'fps': 0.0,
            'buffer_size': len(self.eye_state_buffer),
            'window_progress': 0.0
        }
        
        if not self.enabled:
            status['fatigue_level'] = "MediaPipe unavailable"
            return status
        
        # 计算实际FPS
        if self.last_frame_time > 0:
            status['fps'] = 1.0 / max(current_time - self.last_frame_time, 0.001)
        self.last_frame_time = current_time
        
        try:
            # TOF相机的RGB数据已经是BGR格式，需要转换为RGB给MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                status['face_detected'] = True
                landmarks = results.multi_face_landmarks[0]
                
                # 获取图像尺寸
                h, w = frame.shape[:2]
                
                # 提取眼部坐标
                left_eye = []
                right_eye = []
                
                for idx in self.LEFT_EYE:
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    left_eye.append([x, y])
                    
                for idx in self.RIGHT_EYE:
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    right_eye.append([x, y])
                
                # 计算EAR
                left_ear = self.calculate_ear(np.array(left_eye))
                right_ear = self.calculate_ear(np.array(right_eye))
                ear_avg = (left_ear + right_ear) / 2.0
                
                status['ear_left'] = left_ear
                status['ear_right'] = right_ear
                status['ear_avg'] = ear_avg
                
                # 保存EAR历史
                self.ear_history.append(ear_avg)
                
                # 判断眼睛是否闭合（用于PERCLOS）
                eye_closed = ear_avg < self.EAR_THRESHOLD
                status['eye_closed'] = eye_closed
                
                # 更新PERCLOS缓冲区
                self.eye_state_buffer.append(1 if eye_closed else 0)
                self.timestamp_buffer.append(current_time)
                
                # 计算窗口进度
                status['window_progress'] = (len(self.eye_state_buffer) / self.WINDOW_SIZE) * 100
                
                # 计算PERCLOS
                perclos, valid = self.calculate_perclos()
                status['perclos'] = perclos
                status['perclos_valid'] = valid
                
                if valid:
                    # 保存PERCLOS历史
                    self.perclos_history.append(perclos)
                    
                    # 获取疲劳等级
                    level, desc, color = self.get_fatigue_level(perclos)
                    status['fatigue_level'] = desc
                    status['fatigue_level_code'] = level
                    status['fatigue_color'] = color
                    
                    # 设置警报
                    if level >= 2:  # 重度疲劳
                        status['alarm'] = True
                else:
                    status['fatigue_level'] = "Collecting data..."
                
                # EAR短期疲劳判断（快速响应）
                if eye_closed:
                    self.eye_closed_frames += 1
                    if self.eye_closed_frames >= self.EAR_CONSEC_FRAMES:
                        status['drowsy_alert'] = True
                else:
                    self.eye_closed_frames = 0
                
                # 绘制眼部轮廓（直接在原始帧上绘制，与quick_start.py保持一致）
                eye_color = (0, 0, 255) if eye_closed else (0, 255, 0)
                cv2.polylines(frame, [np.array(left_eye)], True, eye_color, 2)
                cv2.polylines(frame, [np.array(right_eye)], True, eye_color, 2)
                
                # 绘制眼部中心点
                left_center = np.mean(left_eye, axis=0).astype(int)
                right_center = np.mean(right_eye, axis=0).astype(int)
                cv2.circle(frame, tuple(left_center), 3, eye_color, -1)
                cv2.circle(frame, tuple(right_center), 3, eye_color, -1)
            
            else:
                # 没有检测到人脸，重置连续闭眼计数
                self.eye_closed_frames = 0
                status['fatigue_level'] = "No face detected"
        
        except Exception as e:
            logger.error(f"TOF疲劳检测错误: {e}")
            status['fatigue_level'] = f"Detection error: {str(e)[:20]}..."
        
        self.frame_count += 1
        return status
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        Returns:
            stats: 统计数据字典
        """
        stats = {
            'total_frames': self.frame_count,
            'window_size': self.WINDOW_SIZE,
            'current_buffer_size': len(self.eye_state_buffer),
            'avg_ear': 0.0,
            'avg_perclos': 0.0,
            'max_perclos': 0.0,
            'min_perclos': 0.0,
            'runtime_seconds': time.time() - self.start_time
        }
        
        if len(self.ear_history) > 0:
            stats['avg_ear'] = float(np.mean(list(self.ear_history)))
            
        if len(self.perclos_history) > 0:
            perclos_array = np.array(list(self.perclos_history))
            stats['avg_perclos'] = float(np.mean(perclos_array))
            stats['max_perclos'] = float(np.max(perclos_array))
            stats['min_perclos'] = float(np.min(perclos_array))
        
        return stats
    
    def reset(self):
        """
        重置检测器状态
        """
        self.frame_count = 0
        self.drowsy_frames = 0
        self.eye_closed_frames = 0
        
        # 清空缓冲区
        self.eye_state_buffer.clear()
        self.timestamp_buffer.clear()
        self.perclos_history.clear()
        self.ear_history.clear()
        
        # 重置时间
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        logger.info("疲劳检测器已重置")
    
    def is_camera_required(self) -> bool:
        """
        检查是否需要相机数据
        Returns:
            bool: 如果需要TOF相机的RGB数据则返回True
        """
        return self.enabled
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        验证输入帧是否有效
        Args:
            frame: 输入的图像帧
        Returns:
            bool: 帧是否有效
        """
        if frame is None:
            logger.warning("疲劳检测: 输入帧为None")
            return False
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.warning(f"疲劳检测: 无效的帧格式 {frame.shape}，需要3通道RGB图像")
            return False
            
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            logger.warning(f"疲劳检测: 图像尺寸过小 {frame.shape[:2]}")
            return False
            
        return True
    

    
    def draw_status_panel(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """
        绘制状态面板（基于quick_start.py的可视化）
        Args:
            frame: 输入图像
            status: 状态字典
        Returns:
            panel: 状态面板图像
        """
        panel_height = 180
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # 绘制分隔线
        cv2.line(panel, (0, 90), (panel.shape[1], 90), (60, 60, 60), 1)
        
        # 上半部分 - EAR和即时状态
        y_offset = 25
        
        # EAR值
        ear_text = f"EAR: {status.get('ear_avg', 0):.3f}"
        cv2.putText(panel, ear_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 眼睛状态
        eye_status = "CLOSED" if status.get('eye_closed', False) else "OPEN"
        eye_color = (0, 0, 255) if status.get('eye_closed', False) else (0, 255, 0)
        cv2.putText(panel, f"Eyes: {eye_status}", (180, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        
        # 即时疲劳警告
        if status.get('drowsy_alert', False):
            cv2.putText(panel, "DROWSY ALERT!", (350, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS
        fps_text = f"FPS: {status.get('fps', 0):.1f}"
        cv2.putText(panel, fps_text, (panel.shape[1] - 100, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 下半部分 - PERCLOS和疲劳等级
        y_offset = 65
        
        # PERCLOS值和进度条
        if status.get('perclos_valid', False):
            # PERCLOS数值
            perclos_text = f"PERCLOS: {status.get('perclos', 0):.1f}%"
            cv2.putText(panel, perclos_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 绘制PERCLOS进度条
            bar_x = 180
            bar_y = y_offset - 15
            bar_width = 200
            bar_height = 20
            
            # 背景条
            cv2.rectangle(panel, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (60, 60, 60), -1)
            
            # 进度条
            progress = min(status.get('perclos', 0) / 60 * bar_width, bar_width)  # 60%为最大显示
            bar_color = status.get('fatigue_color', (128, 128, 128))
            cv2.rectangle(panel, (bar_x, bar_y), 
                         (bar_x + int(progress), bar_y + bar_height), 
                         bar_color, -1)
            
            # 标记阈值线
            normal_line = int(self.PERCLOS_NORMAL / 60 * bar_width)
            mild_line = int(self.PERCLOS_MILD / 60 * bar_width)
            cv2.line(panel, (bar_x + normal_line, bar_y), 
                    (bar_x + normal_line, bar_y + bar_height), 
                    (255, 255, 255), 1)
            cv2.line(panel, (bar_x + mild_line, bar_y), 
                    (bar_x + mild_line, bar_y + bar_height), 
                    (255, 255, 255), 1)
            
            # 疲劳等级
            cv2.putText(panel, status.get('fatigue_level', 'Unknown'), (bar_x + bar_width + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
        else:
            # 数据收集中提示
            window_progress = status.get('window_progress', 0)
            cv2.putText(panel, f"Collecting data... {window_progress:.0f}%", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 1)
        
        # 第三行 - 统计信息
        y_offset = 120
        
        # 窗口时间
        window_text = f"Window: {self.PERCLOS_WINDOW}s"
        cv2.putText(panel, window_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # 平均EAR
        if len(self.ear_history) > 0:
            avg_ear = np.mean(list(self.ear_history))
            cv2.putText(panel, f"Avg EAR: {avg_ear:.3f}", (150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # 平均PERCLOS
        if len(self.perclos_history) > 0:
            avg_perclos = np.mean(list(self.perclos_history))
            cv2.putText(panel, f"Avg PERCLOS: {avg_perclos:.1f}%", (300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # 运行时间
        runtime = time.time() - self.start_time
        runtime_text = f"Runtime: {int(runtime)}s"
        cv2.putText(panel, runtime_text, (panel.shape[1] - 150, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # 第四行 - 图例
        y_offset = 155
        cv2.putText(panel, "Levels:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(panel, "Normal(<20%)", (80, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel, "Mild(20-40%)", (200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(panel, "Severe(>40%)", (320, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return panel
    
    def draw_fatigue_warning(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """
        在帧上绘制疲劳警告（重度疲劳时显示红色边框）
        Args:
            frame: 输入图像帧
            status: 疲劳检测状态
        Returns:
            frame: 绘制后的图像帧
        """
        # 绘制警告框（如果疲劳等级严重）
        if status.get('perclos_valid', False) and status.get('fatigue_level_code', 0) >= 2:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                        (0, 0, 255), 5)
            cv2.putText(frame, "FATIGUE WARNING!", 
                       (frame.shape[1]//2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame