"""
姿态检测引擎
===========

封装姿态检测推理逻辑，包括：
- 降频推理（可配置间隔）
- 结果缓存与超时管理
- 3D 坐标转换（depth_frame -> 3D points）
- 头部方向过滤（HeadOrientationFilter）
- One-Euro 滤波角度计算
- 性能监控（EMA 延迟统计）
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..compat import np
from ..core.logger import logger
from ..pose import (
    CalculatePostureAngles,
    QUALITY_FLAG_LEFT_EYE_FALLBACK,
    QUALITY_FLAG_RIGHT_EYE_FALLBACK,
)


@dataclass
class PoseResult:
    """
    姿态检测结果

    Attributes:
        data: 姿态数据字典（包含 keypoints_2d, angles, quality_flags 等）
        latency_ms: EMA 平滑后的推理延迟（毫秒）
        speed_fps: 理论推理速度（FPS）
        source: 数据来源 ("inference", "cache", "cache_expired", "no_detection", "error")
        timestamp: 推理时间戳
        fresh: 是否为新鲜数据（True=本帧推理，False=缓存）
        pose_age_ms: 数据年龄（距离上次推理的时间，毫秒）
    """
    data: Optional[Dict[str, Any]]
    latency_ms: float
    speed_fps: float
    source: str
    timestamp: float
    fresh: bool
    pose_age_ms: float


class PoseEngine:
    """
    姿态检测引擎

    职责:
    - 降频推理（每 N 帧执行一次，减少计算开销）
    - 结果缓存（非推理帧使用缓存结果）
    - 缓存超时管理（过期清空，避免展示陈旧数据）
    - 3D 坐标转换（使用 depth_frame 和 system）
    - 头部方向过滤（眼-鼻融合）
    - One-Euro 滤波角度计算（平滑输出）
    - 性能监控（EMA 延迟统计、性能分解）

    使用示例:
        pose_engine = PoseEngine(
            pose_detector=pose_detector,
            head_orientation_filter=head_orientation_filter,
            system=system,
            interval_frames=10,
            cache_timeout=1.0,
            latency_smoothing=0.7,
            system_config=system_config
        )

        # 主循环中调用
        pose_result = pose_engine.maybe_infer(
            rgb_frame=rgb_frame,
            depth_frame=depth_frame,
            face_present=current_face_detected,
            global_frame_interval=global_frame_interval,
            frame_count=system.frame_count,
            global_fps=global_fps
        )

        # pose_result.data 包含完整的姿态数据
        # pose_result.source 指示数据来源（inference/cache）
    """

    def __init__(
        self,
        pose_detector,  # GetPose3dCoords instance
        head_orientation_filter,  # HeadOrientationFilter instance
        system,  # EyeDistanceSystem instance (用于深度转换)
        interval_frames: int = 10,
        cache_timeout: float = 1.0,
        latency_smoothing: float = 0.7,
        system_config: Optional[Dict[str, Any]] = None,
        include_full_3d_coords: bool = True
    ):
        """
        初始化姿态检测引擎

        Args:
            pose_detector: GetPose3dCoords 实例
            head_orientation_filter: HeadOrientationFilter 实例（可选）
            system: EyeDistanceSystem 实例（用于深度转换）
            interval_frames: 推理间隔（帧数），默认 10
            cache_timeout: 缓存超时时间（秒），默认 1.0
            latency_smoothing: 延迟 EMA 平滑系数，默认 0.7
            system_config: 系统配置字典（可选，用于读取 One-Euro 参数）
            include_full_3d_coords: 是否包含完整 3D 坐标（默认 True）
        """
        self.pose_detector = pose_detector
        self.head_orientation_filter = head_orientation_filter
        self.system = system
        self.interval_frames = interval_frames
        self.cache_timeout = cache_timeout
        self.latency_smoothing = latency_smoothing
        self.system_config = system_config or {}
        self.include_full_3d_coords = include_full_3d_coords

        # 状态变量
        self._frame_count = 0
        self._last_result = None
        self._last_timestamp = time.monotonic()
        self._latency_ms = 0.0
        self._pose_angle_calculator = None  # 延迟初始化（One-Euro 滤波器状态）

        # 日志配置
        self._log_interval_seconds = max(0.1, float(os.getenv("AITABLE_LOG_INTERVAL", "1.0")))
        self._min_log_interval_frames = max(1, int(os.getenv("AITABLE_MIN_LOG_FRAMES", "10")))

        logger.info(f"PoseEngine 已初始化 (interval={interval_frames}, timeout={cache_timeout}s)")

    def should_infer(
        self,
        face_present: bool,
        frame_count: int,
        global_frame_interval: float
    ) -> tuple[bool, str]:
        """
        判断是否需要执行推理（不改变状态）

        Args:
            face_present: 当前是否检测到人脸
            frame_count: 当前帧计数
            global_frame_interval: 全局帧间隔（秒）

        Returns:
            tuple[bool, str]: (是否需要推理, 原因说明)
        """
        # 引擎禁用
        if not self.pose_detector:
            return (False, "engine_disabled")

        # 姿态检测不依赖人脸（可以检测全身姿态）
        # 但如果没有人脸，可能降低检测优先级（这里保持原逻辑）

        # 计算数据年龄
        current_time = time.monotonic()
        pose_age_ms = (current_time - self._last_timestamp) * 1000.0

        # 判断是否需要推理
        if self._last_result is None:
            return (True, "first_inference")
        elif pose_age_ms > self.cache_timeout * 1000:
            return (True, "cache_expired")
        elif (self._frame_count + 1) % self.interval_frames == 0:
            return (True, "interval_reached")
        else:
            return (False, "use_cache")

    def maybe_infer(
        self,
        rgb_frame,
        depth_frame,
        face_present: bool,
        global_frame_interval: float,
        frame_count: int,
        global_fps: float,
        person_roi=None,
        person_bbox: Optional[Dict[str, int]] = None
    ) -> PoseResult:
        """
        执行姿态检测（降频 + 缓存）

        Args:
            rgb_frame: RGB 图像帧
            depth_frame: 深度图像帧
            face_present: 当前是否检测到人脸
            global_frame_interval: 全局帧间隔（秒），用于 One-Euro dt
            frame_count: 当前帧计数（用于日志）
            global_fps: 全局 FPS（用于日志）
            person_roi: 可选，外部提供的 person ROI（RegionROI 对象）
            person_bbox: 可选，外部提供的 person bbox（字典：x1, y1, x2, y2）

        Returns:
            PoseResult: 姿态检测结果
        """
        #  修改：如果提供了 person_roi，即使 face_present=False 也继续（BODY_ONLY 模式）
        if not self.pose_detector:
            return PoseResult(
                data=None,
                latency_ms=0.0,
                speed_fps=0.0,
                source="disabled",
                timestamp=time.time(),
                fresh=False,
                pose_age_ms=0.0
            )

        # 如果没有人脸且没有person_roi，则跳过
        if not face_present and person_roi is None:
            return PoseResult(
                data=None,
                latency_ms=0.0,
                speed_fps=0.0,
                source="no_face_no_person",
                timestamp=time.time(),
                fresh=False,
                pose_age_ms=0.0
            )

        # 递增帧计数（取模防止溢出）
        self._frame_count = (self._frame_count + 1) % (self.interval_frames * 1000)

        # 计算数据年龄（距离上次推理的时间）
        current_time = time.monotonic()
        pose_age_ms = (current_time - self._last_timestamp) * 1000.0

        # 判断是否需要执行推理
        should_infer = (
            (self._frame_count % self.interval_frames == 0) or
            (self._last_result is None) or
            (pose_age_ms > self.cache_timeout * 1000)
        )

        if should_infer:
            # 执行推理
            return self._run_inference(
                rgb_frame=rgb_frame,
                depth_frame=depth_frame,
                pose_age_ms=pose_age_ms,
                global_frame_interval=global_frame_interval,
                frame_count=frame_count,
                global_fps=global_fps,
                person_roi=person_roi,
                person_bbox=person_bbox
            )
        else:
            # 使用缓存
            return self._use_cache(pose_age_ms)

    def _run_inference(
        self,
        rgb_frame,
        depth_frame,
        pose_age_ms: float,
        global_frame_interval: float,
        frame_count: int,
        global_fps: float,
        person_roi=None,
        person_bbox: Optional[Dict[str, int]] = None
    ) -> PoseResult:
        """
        执行姿态推理（完整流程）

        支持两种模式：
        1. 使用外部提供的 person_roi（Face-Person 关联场景）
        2. 在全帧上独立检测（传统模式）
        """
        try:
            # 【性能监控】记录推理开始时间
            pose_start = time.time()
            perf_timestamps = {"start": pose_start}

            # 1. 检测 2D 关键点
            #  关键修改：只在有 person_roi 时才执行检测，确保"所有模块分析同一主角"
            if person_roi is not None and hasattr(person_roi, 'roi_rgb'):
                # 使用外部提供的 person ROI（确保分析同一主角）
                detection_frame = person_roi.roi_rgb
                use_roi_mode = True
                roi_offset_x = person_bbox.get('x1', 0) if person_bbox else 0
                roi_offset_y = person_bbox.get('y1', 0) if person_bbox else 0
                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug(f"[Pose] 使用外部 person_roi (offset_x={roi_offset_x}, offset_y={roi_offset_y})")
            else:
                # 缺失 person_roi：返回缓存，避免在整帧重跑可能检测到其他人
                #  保持与 EmotionEngine/FatigueEngine 一致的行为
                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug("[Pose] 缺失 person_roi，返回缓存（避免整帧检测可能检测到其他人）")

                return PoseResult(
                    data=self._last_result if self._last_result else {'status': 'no_person_roi'},
                    latency_ms=self._latency_ms,
                    speed_fps=self._get_speed_fps(),
                    source="cache_no_person_roi",
                    timestamp=self._last_timestamp if self._last_result else time.time(),
                    fresh=False,
                    pose_age_ms=pose_age_ms
                )

            raw_pose = self.pose_detector.detect_posture_with_yolo(detection_frame)
            perf_timestamps["yolo_inference"] = time.time()

            if not raw_pose:
                # 未检测到姿态
                self._last_timestamp = time.monotonic()
                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug("[Pose] frame result: no_detection")

                return PoseResult(
                    data={'status': 'no_detection'},
                    latency_ms=self._latency_ms,
                    speed_fps=self._get_speed_fps(),
                    source="no_detection",
                    timestamp=time.time(),
                    fresh=False,
                    pose_age_ms=pose_age_ms
                )

            # 2. 提取关键点和质量标志
            keypoints_2d = self.pose_detector.filter_keypoints_only(raw_pose)
            quality_flags = self.pose_detector.extract_quality_flags(raw_pose)

            #  坐标变换：如果使用 ROI 模式，需要将坐标转换回全帧坐标系
            if use_roi_mode and (roi_offset_x != 0 or roi_offset_y != 0):
                keypoints_2d = {
                    kp_name: (x + roi_offset_x, y + roi_offset_y)
                    for kp_name, (x, y) in keypoints_2d.items()
                }
                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug(f"[Pose] 关键点坐标已转换到全帧坐标系 (offset: +{roi_offset_x}, +{roi_offset_y})")

            # 3. 转换为 3D 坐标（使用深度数据）
            keypoints_3d = {}
            depth_available_count = 0

            for kp_name, (x, y) in keypoints_2d.items():
                depth_value_mm = self.system.get_robust_depth(depth_frame, x, y)

                if depth_value_mm is not None and depth_value_mm > 0:
                    point_3d = self.system.pixel_to_3d(x, y, depth_value_mm)
                    if point_3d is not None:
                        keypoints_3d[kp_name] = point_3d
                        depth_available_count += 1
                else:
                    keypoints_3d[kp_name] = None

            perf_timestamps["depth_conversion"] = time.time()

            # 4. 使用头部方向过滤器（统一眼鼻坐标系）
            orientation_result = None
            if self.head_orientation_filter:
                orientation_result = self.head_orientation_filter.update(
                    keypoints_3d,
                    keypoints_conf=quality_flags
                )
                perf_timestamps["head_orientation_filter"] = time.time()

                if orientation_result and "rotation" in orientation_result:
                    # 将头部方向注入 keypoints_3d，供角度计算器使用
                    keypoints_3d["_head_orientation"] = orientation_result

                    # 日志记录模式切换及质量信息
                    mode = orientation_result.get('mode', 'unknown')
                    state = orientation_result.get('state', 'unknown')
                    quality = orientation_result.get('quality', {})
                    eye_nose_angle = quality.get('eye_nose_angle')

                    if quality.get('nose_used'):
                        logger.debug(
                            "头部方向融合偏向鼻子 (state=%s, eye-nose=%.1f°)",
                            state,
                            eye_nose_angle if eye_nose_angle is not None else float('nan')
                        )
                    elif state in ('blend', 'agree_blend', 'mild_conflict_blend'):
                        logger.debug(
                            "头部方向融合模式 (state=%s, eye-nose=%.1f°)",
                            state,
                            eye_nose_angle if eye_nose_angle is not None else float('nan')
                        )

            # 5. 计算姿态角度（One-Euro 滤波）
            angles, angles_fallback_status = self._calculate_angles(
                keypoints_3d,
                global_frame_interval,
                perf_timestamps,
                frame_count,
                global_fps
            )

            # 6. 计算推理延迟（EMA 平滑）
            pose_end = time.time()
            instant_latency = (pose_end - pose_start) * 1000.0
            self._latency_ms = (
                self.latency_smoothing * self._latency_ms +
                (1 - self.latency_smoothing) * instant_latency
            )

            # 7. 性能监控日志
            self._log_performance(perf_timestamps, instant_latency, global_frame_interval, global_fps, frame_count)

            # 8. 组装姿态结果
            pose_data = {
                'keypoints_2d': keypoints_2d,
                'angles': angles,
                'quality_flags': quality_flags,
                'latency_ms': self._latency_ms,
                'speed_fps': self._get_speed_fps(),
                'depth_available_count': depth_available_count,
                'total_keypoints': len(keypoints_2d),
                'status': 'ok' if angles else 'partial',
                'angles_fallback_status': angles_fallback_status,
                'orientation': orientation_result,
                'pose_age_ms': 0.0,
                'pose_fresh': True
            }

            # 可选：包含完整 3D 坐标
            if self.include_full_3d_coords:
                pose_data['keypoints_3d'] = keypoints_3d
            else:
                pose_data['keypoints_3d_count'] = depth_available_count

            # 记录质量警告
            if quality_flags.get(QUALITY_FLAG_LEFT_EYE_FALLBACK):
                logger.debug("姿态检测: 左眼使用耳朵回退")
            if quality_flags.get(QUALITY_FLAG_RIGHT_EYE_FALLBACK):
                logger.debug("姿态检测: 右眼使用耳朵回退")

            # 姿态模块调试日志
            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                kp3d_count = pose_data.get('keypoints_3d_count', depth_available_count)
                logger.debug(
                    f"[Pose] frame result: keypoints_2d={len(keypoints_2d)}, "
                    f"keypoints_3d={kp3d_count}, "
                    f"has_shoulders={'left_shoulder' in keypoints_2d and 'right_shoulder' in keypoints_2d}, "
                    f"has_eyes={'left_eye_center' in keypoints_2d and 'right_eye_center' in keypoints_2d}, "
                    f"status={pose_data.get('status', 'unknown')}"
                )

            # 更新缓存
            self._last_result = pose_data
            self._last_timestamp = time.monotonic()

            return PoseResult(
                data=pose_data,
                latency_ms=self._latency_ms,
                speed_fps=self._get_speed_fps(),
                source="inference",
                timestamp=time.time(),
                fresh=True,
                pose_age_ms=0.0
            )

        except Exception as e:
            logger.error(f"姿态检测失败: {e}")
            self._last_timestamp = time.monotonic()
            return PoseResult(
                data={'status': 'error', 'error': str(e)},
                latency_ms=self._latency_ms,
                speed_fps=self._get_speed_fps(),
                source="error",
                timestamp=time.time(),
                fresh=False,
                pose_age_ms=pose_age_ms
            )

    def _calculate_angles(
        self,
        keypoints_3d: Dict,
        global_frame_interval: float,
        perf_timestamps: Dict,
        frame_count: int,
        global_fps: float
    ):
        """计算姿态角度（One-Euro 滤波）"""
        # 定义优先级关键点列表（用于 3D 充足性检查）
        priority_keys = ['left_shoulder', 'right_shoulder', 'left_eye_center', 'right_eye_center', 'nose']

        available_3d_priority_count = sum(
            1 for k in priority_keys
            if k in keypoints_3d and keypoints_3d[k] is not None
        )

        angles = None
        angles_fallback_status = {}

        # 入口条件：检查必需的 3D 点是否可用
        if available_3d_priority_count >= 2:
            try:
                # 持久化姿态角度计算器（首次初始化或更新关键点）
                if self._pose_angle_calculator is None:
                    # 首次初始化：可选从 system_config.json 读取 One-Euro 参数
                    oneeuro_params = self.system_config.get("pose_angles", {}).get("oneeuro_params", None)
                    self._pose_angle_calculator = CalculatePostureAngles(
                        keypoints_3d,
                        smoothing_factor=0.9,
                        oneeuro_params=oneeuro_params
                    )
                else:
                    # 后续帧：仅更新关键点，保持滤波器状态
                    self._pose_angle_calculator.key_points_3d = keypoints_3d

                # 计算 dt：限制在合理范围内（避免异常值）
                dt = max(0.001, min(global_frame_interval, 1.0))
                angles = self._pose_angle_calculator.get_posture_angles(dt)
                perf_timestamps["oneeuro_filter"] = time.time()

                # 记录哪些角度是 fallback（部分点缺失时计算的）
                if available_3d_priority_count < 4:
                    missing_keys = [k for k in priority_keys
                                   if k not in keypoints_3d or keypoints_3d[k] is None]
                    angles_fallback_status['mode'] = 'partial'
                    angles_fallback_status['missing_keys'] = missing_keys
                    # 检查哪些角度可能受影响
                    if 'left_shoulder' in missing_keys or 'right_shoulder' in missing_keys:
                        angles_fallback_status['affected_angles'] = angles_fallback_status.get('affected_angles', []) + ['shoulder_tilt']
                    if 'left_eye_center' in missing_keys or 'right_eye_center' in missing_keys:
                        angles_fallback_status['affected_angles'] = angles_fallback_status.get('affected_angles', []) + ['head_roll', 'head_forward']
                else:
                    angles_fallback_status['mode'] = 'full'

                # 定期输出元数据日志
                if global_fps > 0:
                    log_interval_frames = max(
                        self._min_log_interval_frames,
                        int(global_fps * self._log_interval_seconds)
                    )
                else:
                    log_interval_frames = self._min_log_interval_frames

                if frame_count % log_interval_frames == 0:
                    metadata = self._pose_angle_calculator.get_metadata()
                    total_3d = metadata.get('total_3d_points', 0)
                    missing_3d = metadata.get('missing_3d_points', [])
                    head_tilt_3d = metadata.get('head_tilt_used_3d', False)
                    smoothing = metadata.get('smoothing_factor', 0.9)
                    calc_errors = metadata.get('calculation_errors', [])

                    log_msg = (f"姿态3D数据: 有效={total_3d}, 缺失={len(missing_3d)}, "
                              f"头倾使用3D={head_tilt_3d}, 平滑系数={smoothing}, "
                              f"模式={angles_fallback_status.get('mode', 'unknown')} "
                              f"(日志间隔={log_interval_frames}帧)")
                    if calc_errors:
                        log_msg += f" 警告: {len(calc_errors)}个计算错误"
                    logger.debug(log_msg)

            except Exception as e:
                logger.warning(f"姿态角度计算失败: {e}")
                angles_fallback_status['mode'] = 'error'
                angles_fallback_status['error'] = str(e)

        return angles, angles_fallback_status

    def _log_performance(
        self,
        perf_timestamps: Dict,
        instant_latency: float,
        global_frame_interval: float,
        global_fps: float,
        frame_count: int
    ):
        """输出性能分析日志"""
        # 计算各环节耗时（毫秒）
        perf_breakdown = {}
        if "yolo_inference" in perf_timestamps:
            perf_breakdown["yolo_inference"] = (perf_timestamps["yolo_inference"] - perf_timestamps["start"]) * 1000
        if "depth_conversion" in perf_timestamps:
            perf_breakdown["depth_conversion"] = (perf_timestamps["depth_conversion"] - perf_timestamps["yolo_inference"]) * 1000
        if "head_orientation_filter" in perf_timestamps:
            perf_breakdown["head_orientation_filter"] = (perf_timestamps["head_orientation_filter"] - perf_timestamps["depth_conversion"]) * 1000
        if "oneeuro_filter" in perf_timestamps:
            perf_breakdown["oneeuro_filter"] = (perf_timestamps["oneeuro_filter"] - perf_timestamps.get("head_orientation_filter", perf_timestamps["depth_conversion"])) * 1000

        # 定期输出性能分析日志（每 10 秒一次）
        if global_fps > 0 and frame_count % max(1, int(global_fps * 10)) == 0:
            logger.info(
                "【姿态性能分析】总耗时=%.1fms | YOLO推理=%.1fms | 深度转换=%.1fms | HeadFilter=%.1fms | OneEuro=%.1fms | dt=%.3fs | FPS=%.1f",
                instant_latency,
                perf_breakdown.get("yolo_inference", 0),
                perf_breakdown.get("depth_conversion", 0),
                perf_breakdown.get("head_orientation_filter", 0),
                perf_breakdown.get("oneeuro_filter", 0),
                global_frame_interval,
                global_fps
            )

    def _use_cache(self, pose_age_ms: float) -> PoseResult:
        """使用缓存结果"""
        if self._last_result is not None:
            # 检查缓存是否超时
            if pose_age_ms > self.cache_timeout * 1000:
                # 缓存超时，清空数据
                self._last_result = None
                return PoseResult(
                    data={'status': 'cache_expired'},
                    latency_ms=self._latency_ms,
                    speed_fps=self._get_speed_fps(),
                    source="cache_expired",
                    timestamp=time.time(),
                    fresh=False,
                    pose_age_ms=pose_age_ms
                )
            else:
                # 使用缓存，更新年龄信息
                cached_data = self._last_result.copy()
                cached_data['pose_age_ms'] = pose_age_ms
                cached_data['pose_fresh'] = False

                return PoseResult(
                    data=cached_data,
                    latency_ms=self._latency_ms,
                    speed_fps=self._get_speed_fps(),
                    source="cache",
                    timestamp=time.time(),
                    fresh=False,
                    pose_age_ms=pose_age_ms
                )
        else:
            # 启动前几帧，没有缓存数据
            return PoseResult(
                data={'status': 'waiting_first_inference'},
                latency_ms=0.0,
                speed_fps=0.0,
                source="waiting",
                timestamp=time.time(),
                fresh=False,
                pose_age_ms=0.0
            )

    def _get_speed_fps(self) -> float:
        """计算理论推理速度（FPS）"""
        return 1000.0 / self._latency_ms if self._latency_ms > 0 else 0.0

    def reset(self):
        """重置引擎状态"""
        self._frame_count = 0
        self._last_result = None
        self._last_timestamp = time.monotonic()
        self._latency_ms = 0.0
        self._pose_angle_calculator = None
        logger.info("[PoseEngine] 状态已重置")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息

        Returns:
            统计字典
        """
        return {
            'latency_ms': self._latency_ms,
            'speed_fps': self._get_speed_fps(),
            'frame_count': self._frame_count,
            'has_cache': self._last_result is not None,
            'interval_frames': self.interval_frames,
            'cache_timeout': self.cache_timeout
        }
