"""
YOLO人脸检测器模块 - 仅使用YOLO检测
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict
from ultralytics import YOLO
from ..core.logger import logger
from ..core.constants import Constants


class SimpleFaceDetector:
    """YOLO人脸检测器 - 仅使用YOLO模型"""
    
    def __init__(self, model_path='yolov8n-face.pt', confidence_threshold=None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold or Constants.FACE_CONFIDENCE_THRESHOLD
        self.model = None
        
        # 初始化YOLO模型
        success = self.init_yolo()
        
        if not success:
            raise RuntimeError("Failed to initialize YOLO face detection model")
        
        logger.info("YOLO face detector initialized successfully")
    
    def init_yolo(self) -> bool:
        """初始化YOLO模型"""
        try:
            if not Path(self.model_path).exists():
                alternative_paths = [
                    Path("models") / "yolov8n-face.pt",
                    Path("yolov8n.pt"),
                    "yolov8n-face.pt",  # 可能在当前目录
                    "yolov8n.pt",      # 通用YOLO模型
                ]
                
                for alt_path in alternative_paths:
                    if Path(alt_path).exists():
                        self.model_path = str(alt_path)
                        break
                else:
                    logger.error(f"YOLO model not found. Tried paths:")
                    for path in [self.model_path] + [str(p) for p in alternative_paths]:
                        logger.error(f"  - {path}")
                    return False
            
            # 加载模型
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置推理参数
            self.model.overrides['conf'] = self.confidence_threshold
            self.model.overrides['iou'] = 0.35
            self.model.overrides['verbose'] = False
            
            # 使用GPU加速（如果可用）
            if torch.cuda.is_available():
                self.model.to('cuda')
                logger.info("Using GPU acceleration for YOLO")
            else:
                logger.info("Using CPU for YOLO inference")
            
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    def detect_face(self, rgb_frame: np.ndarray) -> Optional[Dict]:
        """YOLO人脸检测"""
        try:
            # YOLO推理
            results = self.model(rgb_frame, verbose=False)
            
            if not results or not results[0].boxes:
                return None
            
            boxes = results[0].boxes
            
            # 过滤置信度过低的检测结果
            valid_boxes = []
            for i, box in enumerate(boxes):
                conf = float(box.conf[0].cpu().numpy())
                if conf >= self.confidence_threshold:
                    valid_boxes.append((i, box, conf))
            
            if not valid_boxes:
                return None
            
            # 选择最大面积且置信度最高的人脸
            if len(valid_boxes) > 1:
                best_box = max(valid_boxes, key=lambda x: (
                    # 先按面积排序
                    (x[1].xyxy[0][2] - x[1].xyxy[0][0]) * (x[1].xyxy[0][3] - x[1].xyxy[0][1]),
                    # 再按置信度排序
                    x[2]
                ))
                best_idx, box, conf = best_box
            else:
                best_idx, box, conf = valid_boxes[0]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 尝试获取关键点（如果模型支持）
            keypoints = None
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                if results[0].keypoints.xy.shape[0] > best_idx:
                    keypoints = results[0].keypoints.xy[best_idx].cpu().numpy()
            
            # 提取眼睛位置
            if keypoints is not None and keypoints.shape[0] >= 2:
                # 使用YOLO关键点
                left_eye = tuple(keypoints[0].astype(int))
                right_eye = tuple(keypoints[1].astype(int))
            else:
                # 估算眼睛位置
                left_eye, right_eye = self.estimate_eye_positions(x1, y1, x2, y2)
            
            return {
                'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                'left_eye': left_eye,
                'right_eye': right_eye,
                'confidence': conf,
                'method': 'YOLO'
            }
            
        except Exception as e:
            logger.error(f"YOLO face detection failed: {e}")
            return None
    
    def estimate_eye_positions(self, x1, y1, x2, y2):
        """基于人脸框估计眼睛位置"""
        face_width = x2 - x1
        face_height = y2 - y1
        
        # 根据人脸比例估算眼睛位置
        # 左眼约在人脸宽度的33%位置，高度的38%位置
        left_eye = (int(x1 + face_width * 0.33), int(y1 + face_height * 0.38))
        # 右眼约在人脸宽度的67%位置，高度的38%位置  
        right_eye = (int(x1 + face_width * 0.67), int(y1 + face_height * 0.38))
        
        return left_eye, right_eye
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {'status': 'Not initialized'}
        
        return {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'status': 'Ready'
        }
    
    def set_confidence_threshold(self, threshold: float):
        """动态设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        if self.model is not None:
            self.model.overrides['conf'] = self.confidence_threshold
        logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")