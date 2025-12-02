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

    # 深度信息（bbox 区域采样，保留兼容）
    depth_mm: float = 0.0
    depth_valid: bool = False

    # 3D 肩部信息（用于选人）
    left_shoulder_3d: Optional[Tuple[float, float, float]] = None
    right_shoulder_3d: Optional[Tuple[float, float, float]] = None
    shoulder_distance_m: float = field(default=float('inf'))  # 到相机原点距离

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
        # 深度选人参数（保留兼容，但不再用于选人）
        depth_range_mm: Tuple[float, float] = (200, 1500),
        switch_depth_delta_mm: float = 100,

        # 跟踪参数
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        min_keep_frames: int = 15,

        # 人脸推导参数
        face_keypoint_confidence: float = 0.3,
        face_expand_ratio: float = 1.5,

        # 相机内参（用于 pixel_to_3d）
        camera_intrinsics: Optional[Dict[str, float]] = None,

        # 3D 选人参数
        distance_range_m: Tuple[float, float] = (0.3, 1.5),
        switch_distance_delta_m: float = 0.05,
        switch_hold_frames: int = 8,
    ):
        """
        Args:
            depth_range_mm: 有效深度范围 (min, max)（保留兼容）
            switch_depth_delta_mm: 切换主目标需要的深度差（保留兼容）
            iou_threshold: IoU 匹配阈值
            max_lost_frames: 丢失多少帧后删除轨迹
            min_keep_frames: 最少保持帧数（防止频繁切换）
            face_keypoint_confidence: 人脸关键点最低置信度
            face_expand_ratio: 人脸框扩展比例
            camera_intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            distance_range_m: 3D 距离有效范围（米）
            switch_distance_delta_m: 切换主目标需要的 3D 距离差（米）
            switch_hold_frames: 连续多少帧更近才切换
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

        # 相机内参（用于 3D 转换）
        self.camera_intrinsics = camera_intrinsics

        # 3D 选人参数
        self.distance_range_m = distance_range_m
        self.switch_distance_delta_m = switch_distance_delta_m
        self.switch_hold_frames = switch_hold_frames

        # 切换候选跟踪（时间滞后）
        self._switch_candidate_id: Optional[int] = None
        self._switch_candidate_frames: int = 0

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

    def _pixel_to_3d_simple(
        self,
        x: float,
        y: float,
        depth_frame: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """
        轻量版 pixel_to_3d，只转换单个点

        Args:
            x, y: 像素坐标
            depth_frame: 深度图（单位：mm）

        Returns:
            (x_m, y_m, z_m) 3D 坐标（米），或 None
        """
        if self.camera_intrinsics is None:
            return None

        if depth_frame is None or x < 0 or y < 0:
            return None

        h, w = depth_frame.shape[:2]
        ix, iy = int(x), int(y)
        if ix >= w or iy >= h:
            return None

        z_mm = depth_frame[iy, ix]
        if z_mm <= 0 or z_mm > 2000:  # 无效深度
            return None

        z_m = z_mm / 1000.0
        fx = self.camera_intrinsics.get('fx', 500)
        fy = self.camera_intrinsics.get('fy', 500)
        cx = self.camera_intrinsics.get('cx', w / 2)
        cy = self.camera_intrinsics.get('cy', h / 2)

        x_m = (x - cx) * z_m / fx
        y_m = (y - cy) * z_m / fy

        return (x_m, y_m, z_m)

    def _compute_shoulder_3d(
        self,
        track: TrackedPerson,
        depth_frame: Optional[np.ndarray],
    ):
        """
        计算肩部 3D 坐标和距离

        COCO 关键点索引：5=左肩，6=右肩
        """
        LEFT_SHOULDER_IDX = 5
        RIGHT_SHOULDER_IDX = 6

        left_kp = next(
            (kp for kp in track.keypoints if kp.get('index') == LEFT_SHOULDER_IDX),
            None
        )
        right_kp = next(
            (kp for kp in track.keypoints if kp.get('index') == RIGHT_SHOULDER_IDX),
            None
        )

        left_3d = None
        right_3d = None

        if depth_frame is not None:
            if left_kp and left_kp.get('confidence', 0) > 0.3:
                left_3d = self._pixel_to_3d_simple(
                    left_kp['x'], left_kp['y'], depth_frame
                )

            if right_kp and right_kp.get('confidence', 0) > 0.3:
                right_3d = self._pixel_to_3d_simple(
                    right_kp['x'], right_kp['y'], depth_frame
                )

        track.left_shoulder_3d = left_3d
        track.right_shoulder_3d = right_3d

        # 计算肩部中点到原点距离
        if left_3d and right_3d:
            midpoint = np.array([
                (left_3d[0] + right_3d[0]) / 2,
                (left_3d[1] + right_3d[1]) / 2,
                (left_3d[2] + right_3d[2]) / 2,
            ])
            track.shoulder_distance_m = float(np.linalg.norm(midpoint))
        elif left_3d:
            track.shoulder_distance_m = float(np.linalg.norm(left_3d))
        elif right_3d:
            track.shoulder_distance_m = float(np.linalg.norm(right_3d))
        else:
            track.shoulder_distance_m = float('inf')

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

        # 更新深度（bbox 区域采样，保留兼容）
        if depth_frame is not None:
            sample = self.depth_selector.sample_depth(depth_frame, track.bbox)
            track.depth_mm = sample.depth_mm
            track.depth_valid = sample.valid

        # 计算 3D 肩部（用于选人）
        self._compute_shoulder_3d(track, depth_frame)
    
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

        # 采样深度（bbox 区域，保留兼容）
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

        # 计算 3D 肩部（用于选人）
        self._compute_shoulder_3d(track, depth_frame)
    
    def _remove_lost_tracks(self):
        """删除丢失的轨迹"""
        to_remove = []
        
        for track_id, track in self._tracks.items():
            if track.frames_since_update > self.max_lost_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracks[track_id]
    
    def _select_primary_target(self, depth_frame: Optional[np.ndarray]):
        """
        选择主目标（使用 3D 肩部距离 + 时间滞后）

        逻辑：
        1. 只考虑当前帧有检测、3D 距离有效、在有效范围内的候选
        2. 找最近的候选
        3. 如果最近的不是当前目标，需要连续 N 帧更近才切换
        """
        if not self._tracks:
            self._primary_target = None
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
            return

        # 重置所有轨迹的 is_primary
        for track in self._tracks.values():
            track.is_primary = False

        # 构建候选列表：当前帧有检测、3D 距离有效、在有效范围内
        min_dist, max_dist = self.distance_range_m
        candidates = [
            track for track in self._tracks.values()
            if track.frames_since_update == 0
            and track.shoulder_distance_m < float('inf')
            and min_dist < track.shoulder_distance_m < max_dist
        ]

        if not candidates:
            # 没有有效候选，保持之前的主目标
            if self._primary_target and self._primary_target.track_id in self._tracks:
                self._primary_target = self._tracks[self._primary_target.track_id]
                self._primary_target.is_primary = True
            return

        # 找最近的候选
        nearest = min(candidates, key=lambda t: t.shoulder_distance_m)

        # 首次选择
        if self._primary_target is None:
            self._primary_target = nearest
            self._primary_target.is_primary = True
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
            return

        current_id = self._primary_target.track_id

        # 当前目标仍是最近的
        if nearest.track_id == current_id:
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
            self._primary_target = nearest
            self._primary_target.is_primary = True
            return

        # 当前目标还在候选中吗？
        current_track = self._tracks.get(current_id)
        if current_track is None or current_track.shoulder_distance_m == float('inf'):
            # 当前目标丢失或无有效 3D，立即切换
            self._primary_target = nearest
            self._primary_target.is_primary = True
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
            return

        # 新目标是否显著更近？
        delta = current_track.shoulder_distance_m - nearest.shoulder_distance_m
        if delta < self.switch_distance_delta_m:
            # 差距不够，保持当前目标
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
            self._primary_target = current_track
            self._primary_target.is_primary = True
            return

        # 时间滞后：连续 N 帧更近才切换
        if nearest.track_id == self._switch_candidate_id:
            self._switch_candidate_frames += 1
        else:
            self._switch_candidate_id = nearest.track_id
            self._switch_candidate_frames = 1

        if self._switch_candidate_frames >= self.switch_hold_frames:
            # 切换！
            self._primary_target = nearest
            self._primary_target.is_primary = True
            self._switch_candidate_id = None
            self._switch_candidate_frames = 0
        else:
            # 继续等待，保持当前目标
            self._primary_target = current_track
            self._primary_target.is_primary = True
    
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
        self._switch_candidate_id = None
        self._switch_candidate_frames = 0
        self.depth_selector.reset()
