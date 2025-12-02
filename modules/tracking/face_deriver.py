"""
FaceDeriver - 从 Pose keypoints 推导人脸边界框

COCO 17 关键点索引：
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    ...
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DerivedFace:
    """推导出的人脸信息"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)
    left_eye: Optional[Tuple[float, float]]  # (x, y)
    right_eye: Optional[Tuple[float, float]]  # (x, y)
    nose: Optional[Tuple[float, float]]  # (x, y)
    valid: bool  # 是否有效（关键点置信度足够）
    confidence: float  # 综合置信度


class FaceDeriver:
    """
    从 Pose keypoints 推导人脸边界框
    
    策略：
    1. 使用眼睛和鼻子关键点构建人脸框
    2. 如果眼睛缺失，使用鼻子 + 肩膀估算
    3. 返回有效性标志，供上层决定是否回退到 FaceDetector
    """
    
    # COCO 关键点索引
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    
    def __init__(
        self,
        min_keypoint_confidence: float = 0.3,
        face_expand_ratio: float = 1.5,
        min_face_size: int = 20,
    ):
        """
        Args:
            min_keypoint_confidence: 关键点最低置信度阈值
            face_expand_ratio: 人脸框扩展比例（基于眼间距）
            min_face_size: 最小人脸尺寸（像素）
        """
        self.min_keypoint_confidence = min_keypoint_confidence
        self.face_expand_ratio = face_expand_ratio
        self.min_face_size = min_face_size
    
    def derive_face(
        self,
        keypoints: List[Dict],
        person_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> DerivedFace:
        """
        From keypoints derive face bounding box.

        Args:
            keypoints: list of 17 keypoints with fields {index, x, y, confidence}
            person_bbox: person bbox (x1, y1, x2, y2); kept for API compatibility, not used for fallback in this version

        Returns:
            DerivedFace result
        """
        # 提取关键点
        kps = self._extract_keypoints(keypoints)
        
        nose = kps.get(self.NOSE)
        left_eye = kps.get(self.LEFT_EYE)
        right_eye = kps.get(self.RIGHT_EYE)
        left_ear = kps.get(self.LEFT_EAR)
        right_ear = kps.get(self.RIGHT_EAR)
        left_shoulder = kps.get(self.LEFT_SHOULDER)
        right_shoulder = kps.get(self.RIGHT_SHOULDER)
        
        # 策略 1：双眼 + 鼻子（最佳情况）
        if left_eye and right_eye and nose:
            return self._derive_from_eyes_nose(
                left_eye, right_eye, nose, 
                left_ear, right_ear
            )
        
        # 策略 2：单眼 + 鼻子
        if (left_eye or right_eye) and nose:
            eye = left_eye or right_eye
            return self._derive_from_single_eye_nose(eye, nose, left_ear, right_ear)
        
        # 策略 3：鼻子 + 肩膀（估算）
        if nose and (left_shoulder or right_shoulder):
            return self._derive_from_nose_shoulders(
                nose, left_shoulder, right_shoulder
            )
        
        # 无法推导
        return DerivedFace(
            bbox=(0, 0, 0, 0),
            center=(0, 0),
            left_eye=None,
            right_eye=None,
            nose=None,
            valid=False,
            confidence=0.0,
        )
    
    def _extract_keypoints(
        self, 
        keypoints: List[Dict]
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        提取有效的关键点
        
        Returns:
            Dict[index -> (x, y, confidence)]，只包含置信度足够的关键点
        """
        result = {}
        for kp in keypoints:
            idx = kp.get('index', -1)
            x = kp.get('x', 0)
            y = kp.get('y', 0)
            conf = kp.get('confidence', 0)
            
            if conf >= self.min_keypoint_confidence and x > 0 and y > 0:
                result[idx] = (x, y, conf)
        
        return result
    
    def _derive_from_eyes_nose(
        self,
        left_eye: Tuple[float, float, float],
        right_eye: Tuple[float, float, float],
        nose: Tuple[float, float, float],
        left_ear: Optional[Tuple[float, float, float]] = None,
        right_ear: Optional[Tuple[float, float, float]] = None,
    ) -> DerivedFace:
        """从双眼和鼻子推导人脸框"""
        lx, ly, lc = left_eye
        rx, ry, rc = right_eye
        nx, ny, nc = nose
        
        # 眼间距
        eye_dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
        
        # 人脸中心（眼睛中点和鼻子的加权平均）
        eye_center_x = (lx + rx) / 2
        eye_center_y = (ly + ry) / 2
        face_center_x = (eye_center_x + nx) / 2
        face_center_y = (eye_center_y * 0.6 + ny * 0.4)  # 偏向眼睛
        
        # 估算人脸尺寸
        # 典型人脸：眼间距约占脸宽的 1/3
        face_width = eye_dist * 3.0 * self.face_expand_ratio
        face_height = face_width * 1.3  # 人脸高宽比约 1.3
        
        # 如果有耳朵，可以更准确估算宽度
        if left_ear and right_ear:
            ear_dist = np.sqrt(
                (left_ear[0] - right_ear[0]) ** 2 + 
                (left_ear[1] - right_ear[1]) ** 2
            )
            face_width = max(face_width, ear_dist * 1.2)
        
        # 确保最小尺寸
        face_width = max(face_width, self.min_face_size)
        face_height = max(face_height, self.min_face_size)
        
        # 计算边界框
        x1 = face_center_x - face_width / 2
        y1 = eye_center_y - face_height * 0.4  # 眼睛在脸的上 40%
        x2 = face_center_x + face_width / 2
        y2 = y1 + face_height
        
        # 综合置信度
        confidence = (lc + rc + nc) / 3
        
        return DerivedFace(
            bbox=(x1, y1, x2, y2),
            center=(face_center_x, face_center_y),
            left_eye=(lx, ly),
            right_eye=(rx, ry),
            nose=(nx, ny),
            valid=True,
            confidence=confidence,
        )
    
    def _derive_from_single_eye_nose(
        self,
        eye: Tuple[float, float, float],
        nose: Tuple[float, float, float],
        left_ear: Optional[Tuple[float, float, float]] = None,
        right_ear: Optional[Tuple[float, float, float]] = None,
    ) -> DerivedFace:
        """从单眼和鼻子推导人脸框（侧脸情况）"""
        ex, ey, ec = eye
        nx, ny, nc = nose
        
        # 眼鼻距离
        eye_nose_dist = np.sqrt((ex - nx) ** 2 + (ey - ny) ** 2)
        
        # 估算人脸尺寸（眼鼻距离约占脸高的 1/4）
        face_height = eye_nose_dist * 4.0 * self.face_expand_ratio
        face_width = face_height / 1.3
        
        # 人脸中心（偏向眼睛）
        face_center_x = (ex + nx) / 2
        face_center_y = (ey + ny) / 2
        
        # 确保最小尺寸
        face_width = max(face_width, self.min_face_size)
        face_height = max(face_height, self.min_face_size)
        
        # 计算边界框
        x1 = face_center_x - face_width / 2
        y1 = face_center_y - face_height / 2
        x2 = face_center_x + face_width / 2
        y2 = face_center_y + face_height / 2
        
        # 置信度降低（侧脸）
        confidence = (ec + nc) / 2 * 0.8
        
        # 判断是左眼还是右眼
        is_left_eye = ex < nx  # 眼睛在鼻子左边
        
        return DerivedFace(
            bbox=(x1, y1, x2, y2),
            center=(face_center_x, face_center_y),
            left_eye=(ex, ey) if is_left_eye else None,
            right_eye=None if is_left_eye else (ex, ey),
            nose=(nx, ny),
            valid=True,
            confidence=confidence,
        )
    
    def _derive_from_nose_shoulders(
        self,
        nose: Tuple[float, float, float],
        left_shoulder: Optional[Tuple[float, float, float]],
        right_shoulder: Optional[Tuple[float, float, float]],
    ) -> DerivedFace:
        """从鼻子和肩膀推导人脸框（低头/遮挡情况）"""
        nx, ny, nc = nose
        
        # 肩宽
        if left_shoulder and right_shoulder:
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder:
            shoulder_width = abs(left_shoulder[0] - nx) * 2
            shoulder_center_y = left_shoulder[1]
        else:
            shoulder_width = abs(right_shoulder[0] - nx) * 2
            shoulder_center_y = right_shoulder[1]
        
        # 头部约占肩宽的 1/2
        face_width = shoulder_width * 0.5 * self.face_expand_ratio
        face_height = face_width * 1.3
        
        # 确保最小尺寸
        face_width = max(face_width, self.min_face_size)
        face_height = max(face_height, self.min_face_size)
        
        # 人脸中心（鼻子上方）
        face_center_x = nx
        face_center_y = ny - face_height * 0.2
        
        # 计算边界框
        x1 = face_center_x - face_width / 2
        y1 = face_center_y - face_height / 2
        x2 = face_center_x + face_width / 2
        y2 = face_center_y + face_height / 2
        
        # 置信度较低（推测成分大）
        confidence = nc * 0.6
        
        return DerivedFace(
            bbox=(x1, y1, x2, y2),
            center=(face_center_x, face_center_y),
            left_eye=None,
            right_eye=None,
            nose=(nx, ny),
            valid=True,
            confidence=confidence,
        )
    
    def _derive_from_person_bbox(
        self,
        person_bbox: Tuple[float, float, float, float],
    ) -> DerivedFace:
        """Derived face from person bbox (fallback)

        Note: derive_face() no longer calls this method; fallback handled in orchestrator.
        """
        x1, y1, x2, y2 = person_bbox
        
        person_width = x2 - x1
        person_height = y2 - y1
        
        # 假设头部在人体框的上 1/6
        head_height = person_height / 6
        head_width = person_width * 0.6
        
        face_center_x = (x1 + x2) / 2
        face_y1 = y1
        face_y2 = y1 + head_height
        
        face_x1 = face_center_x - head_width / 2
        face_x2 = face_center_x + head_width / 2
        
        return DerivedFace(
            bbox=(face_x1, face_y1, face_x2, face_y2),
            center=(face_center_x, (face_y1 + face_y2) / 2),
            left_eye=None,
            right_eye=None,
            nose=None,
            valid=False,  # 标记为不可靠
            confidence=0.3,
        )
    
    def get_eye_positions(
        self,
        keypoints: List[Dict],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        直接获取眼睛位置（用于眼距计算）
        
        Returns:
            (left_eye, right_eye)，每个是 (x, y) 或 None
        """
        kps = self._extract_keypoints(keypoints)
        
        left_eye = kps.get(self.LEFT_EYE)
        right_eye = kps.get(self.RIGHT_EYE)
        
        left = (left_eye[0], left_eye[1]) if left_eye else None
        right = (right_eye[0], right_eye[1]) if right_eye else None
        
        return left, right