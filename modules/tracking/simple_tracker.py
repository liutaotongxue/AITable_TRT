"""
SimpleTracker - 基于 IoU 的简单多目标跟踪器

特性：
- 单 YOLO-Pose 检测器作为唯一人员来源
- IoU 轨迹关联（无需 Kalman/ReID）
- 深度选人确定主目标
- 从 keypoints 推导 face_bbox
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .face_deriver import FaceDeriver, DerivedFace
from .depth_selector import DepthSelector, DepthSample


@dataclass
class TrackedPerson:
    """跟踪的人员对象"""
    track_id: int
    
    # 边界框 (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]
    
    # Pose 关键点（原始格式）
    keypoints: List[Dict]
    
    # Pose 检测置信度
    confidence: float
    
    # 推导的人脸信息
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    face_valid: bool = False
    face_confidence: float = 0.0
    
    # 眼睛位置（用于眼距计算）
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    nose: Optional[Tuple[float, float]] = None
    
    # 深度信息
    depth_mm: float = 0.0
    depth_valid: bool = False
    
    # 跟踪状态
    frames_since_update: int = 0
    total_frames: int = 1
    is_primary: bool = False
    
    # 时间戳
    last_update_time: float = field(default_factory=time.time)


class SimpleTracker:
    """
    基于 IoU 的简单多目标跟踪器
    
    使用方法：
        tracker = SimpleTracker()
        
        # 每帧调用
        tracks = tracker.update(pose_detections, depth_frame)
        target = tracker.get_primary_target()
        
        if target:
            face_bbox = target.face_bbox
            keypoints = target.keypoints
    """
    
    def __init__(
        self,
        # 深度选人参数
        depth_range_mm: Tuple[float, float] = (200, 1500),
        switch_depth_delta_mm: float = 100,
        
        # 跟踪参数
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        min_keep_frames: int = 15,
        
        # 人脸推导参数
        face_keypoint_confidence: float = 0.3,
        face_expand_ratio: float = 1.5,
    ):
        """
        Args:
            depth_range_mm: 有效深度范围 (min, max)
            switch_depth_delta_mm: 切换主目标需要的深度差
            iou_threshold: IoU 匹配阈值
            max_lost_frames: 丢失多少帧后删除轨迹
            min_keep_frames: 最少保持帧数（防止频繁切换）
            face_keypoint_confidence: 人脸关键点最低置信度
            face_expand_ratio: 人脸框扩展比例
        """
        # 初始化组件
        self.face_deriver = FaceDeriver(
            min_keypoint_confidence=face_keypoint_confidence,
            face_expand_ratio=face_expand_ratio,
        )
        
        self.depth_selector = DepthSelector(
            depth_range_mm=depth_range_mm,
            switch_depth_delta_mm=switch_depth_delta_mm,
        )
        
        # 跟踪参数
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_keep_frames = min_keep_frames
        
        # 轨迹存储
        self._tracks: Dict[int, TrackedPerson] = {}
        self._next_id = 1
        
        # 主目标
        self._primary_target: Optional[TrackedPerson] = None
    
    def update(
        self,
        detections: List[Dict],
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[TrackedPerson]:
        """
        更新跟踪器
        
        Args:
            detections: 检测结果列表，每个元素格式：
                {
                    'bbox': {'x1': float, 'y1': float, 'x2': float, 'y2': float},
                    'confidence': float,
                    'keypoints': [{'index': int, 'x': float, 'y': float, 'confidence': float}, ...]
                }
            depth_frame: 深度图（可选）
            
        Returns:
            当前帧的所有跟踪目标列表
        """
        # 1. 转换检测格式
        converted_dets = self._convert_detections(detections)
        
        # 2. IoU 匹配
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            converted_dets
        )
        
        # 3. 更新匹配的轨迹
        for det_idx, track_id in matched:
            det = converted_dets[det_idx]
            self._update_track(track_id, det, depth_frame)
        
        # 4. 创建新轨迹
        for det_idx in unmatched_dets:
            det = converted_dets[det_idx]
            self._create_track(det, depth_frame)
        
        # 5. 处理未匹配的轨迹
        for track_id in unmatched_tracks:
            self._tracks[track_id].frames_since_update += 1
        
        # 6. 删除丢失的轨迹
        self._remove_lost_tracks()
        
        # 7. 选择主目标
        self._select_primary_target(depth_frame)
        
        return list(self._tracks.values())
    
    def _convert_detections(self, detections: List[Dict]) -> List[Dict]:
        """转换检测格式为内部格式"""
        converted = []
        
        for det in detections:
            # 处理 bbox（支持 dict 和 tuple 格式）
            bbox = det.get('bbox', {})
            if isinstance(bbox, dict):
                bbox_tuple = (
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('x2', 0),
                    bbox.get('y2', 0),
                )
            else:
                bbox_tuple = tuple(bbox)
            
            # 获取关键点
            keypoints = det.get('keypoints', [])
            
            converted.append({
                'bbox': bbox_tuple,
                'confidence': det.get('confidence', 0),
                'keypoints': keypoints,
            })
        
        return converted
    
    def _match_detections(
        self,
        detections: List[Dict],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        IoU 匹配检测和轨迹
        
        Returns:
            (matched, unmatched_dets, unmatched_tracks)
            - matched: [(det_idx, track_id), ...]
            - unmatched_dets: [det_idx, ...]
            - unmatched_tracks: [track_id, ...]
        """
        if not detections or not self._tracks:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self._tracks.keys())
            return [], unmatched_dets, unmatched_tracks
        
        # 计算 IoU 矩阵
        track_ids = list(self._tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for d_idx, det in enumerate(detections):
            det_bbox = det['bbox']
            for t_idx, track_id in enumerate(track_ids):
                track_bbox = self._tracks[track_id].bbox
                iou_matrix[d_idx, t_idx] = self._compute_iou(det_bbox, track_bbox)
        
        # 贪婪匹配
        matched = []
        used_dets = set()
        used_tracks = set()
        
        # 按 IoU 降序匹配
        while True:
            if iou_matrix.size == 0:
                break
                
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            d_idx, t_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            matched.append((d_idx, track_ids[t_idx]))
            used_dets.add(d_idx)
            used_tracks.add(track_ids[t_idx])
            
            # 标记已使用
            iou_matrix[d_idx, :] = -1
            iou_matrix[:, t_idx] = -1
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """计算两个边界框的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _update_track(
        self,
        track_id: int,
        detection: Dict,
        depth_frame: Optional[np.ndarray],
    ):
        """更新已有轨迹"""
        track = self._tracks[track_id]
        
        # 更新基本信息
        track.bbox = detection['bbox']
        track.keypoints = detection['keypoints']
        track.confidence = detection['confidence']
        track.frames_since_update = 0
        track.total_frames += 1
        track.last_update_time = time.time()
        
        # 推导人脸
        derived_face = self.face_deriver.derive_face(
            detection['keypoints'],
            detection['bbox'],
        )
        
        track.face_bbox = derived_face.bbox if derived_face.valid else None
        track.face_valid = derived_face.valid
        track.face_confidence = derived_face.confidence
        track.left_eye = derived_face.left_eye
        track.right_eye = derived_face.right_eye
        track.nose = derived_face.nose
        
        # 更新深度
        if depth_frame is not None:
            sample = self.depth_selector.sample_depth(depth_frame, track.bbox)
            track.depth_mm = sample.depth_mm
            track.depth_valid = sample.valid
    
    def _create_track(
        self,
        detection: Dict,
        depth_frame: Optional[np.ndarray],
    ):
        """创建新轨迹"""
        track_id = self._next_id
        self._next_id += 1
        
        # 推导人脸
        derived_face = self.face_deriver.derive_face(
            detection['keypoints'],
            detection['bbox'],
        )
        
        # 采样深度
        depth_mm = 0.0
        depth_valid = False
        if depth_frame is not None:
            sample = self.depth_selector.sample_depth(depth_frame, detection['bbox'])
            depth_mm = sample.depth_mm
            depth_valid = sample.valid
        
        track = TrackedPerson(
            track_id=track_id,
            bbox=detection['bbox'],
            keypoints=detection['keypoints'],
            confidence=detection['confidence'],
            face_bbox=derived_face.bbox if derived_face.valid else None,
            face_valid=derived_face.valid,
            face_confidence=derived_face.confidence,
            left_eye=derived_face.left_eye,
            right_eye=derived_face.right_eye,
            nose=derived_face.nose,
            depth_mm=depth_mm,
            depth_valid=depth_valid,
        )
        
        self._tracks[track_id] = track
    
    def _remove_lost_tracks(self):
        """删除丢失的轨迹"""
        to_remove = []
        
        for track_id, track in self._tracks.items():
            if track.frames_since_update > self.max_lost_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracks[track_id]
    
    def _select_primary_target(self, depth_frame: Optional[np.ndarray]):
        """选择主目标"""
        if not self._tracks:
            self._primary_target = None
            return
        
        # 重置所有轨迹的 is_primary
        for track in self._tracks.values():
            track.is_primary = False
        
        # 构建候选列表
        candidates = [
            {
                'track_id': track.track_id,
                'bbox': track.bbox,
            }
            for track in self._tracks.values()
            if track.frames_since_update == 0  # 只考虑当前帧有检测的
        ]
        
        if not candidates:
            # 没有当前帧检测的候选，保持之前的主目标
            if self._primary_target and self._primary_target.track_id in self._tracks:
                self._primary_target = self._tracks[self._primary_target.track_id]
                self._primary_target.is_primary = True
            else:
                self._primary_target = None
            return
        
        # 使用深度选择器选择主目标
        if depth_frame is not None:
            primary_id = self.depth_selector.select_primary(candidates, depth_frame)
        else:
            # 没有深度图，选择置信度最高的
            primary_id = max(
                candidates,
                key=lambda c: self._tracks[c['track_id']].confidence
            )['track_id']
        
        if primary_id and primary_id in self._tracks:
            self._primary_target = self._tracks[primary_id]
            self._primary_target.is_primary = True
        else:
            self._primary_target = None
    
    def get_primary_target(self) -> Optional[TrackedPerson]:
        """获取主目标"""
        return self._primary_target
    
    def get_all_tracks(self) -> List[TrackedPerson]:
        """获取所有轨迹"""
        return list(self._tracks.values())
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedPerson]:
        """根据 ID 获取轨迹"""
        return self._tracks.get(track_id)
    
    def reset(self):
        """重置跟踪器"""
        self._tracks.clear()
        self._next_id = 1
        self._primary_target = None
        self.depth_selector.reset()
