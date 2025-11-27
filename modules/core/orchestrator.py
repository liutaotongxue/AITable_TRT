"""
主循环协调器
===========

完整封装主循环逻辑，协调所有检测引擎和系统组件。
"""
import time
import signal
import os
from datetime import datetime
from typing import Optional, Dict, Any
from ..compat import np
from ..core.logger import logger
from ..engines import EmotionEngine, FatigueEngine, PoseEngine
from ..core.telemetry import TelemetryBuilder
from ..ui import WindowManager
from ..monitoring import FrameRateMonitor
from ..monitoring.resource import ResourceMonitor
from ..core.inference_scheduler import InferenceScheduler, SchedulerConfig
from ..detection import ROIManager

# SimpleTracker（可选导入，模块不存在时禁用追踪模式）
try:
    from ..tracking import SimpleTracker
    _TRACKER_AVAILABLE = True
except ImportError:
    SimpleTracker = None
    _TRACKER_AVAILABLE = False


class DetectionOrchestrator:
    """
    检测协调器 - 完整主循环实现

    职责:
    - 封装整个主循环逻辑（原 run_main_loop 函数）
    - 协调所有检测引擎（情绪、疲劳、姿态）
    - 管理窗口、遥测、状态跟踪
    - 信号处理和优雅退出
    - 提供简洁的 API 供主程序调用

    使用示例:
        orchestrator = DetectionOrchestrator(
            system=system,
            emotion_engine=emotion_engine,
            fatigue_engine=fatigue_engine,
            pose_detector=pose_detector,
            head_orientation_filter=head_orientation_filter,
            window_manager=window_manager,
            telemetry_builder=telemetry_builder,
            system_config=system_config
        )

        # 运行主循环
        orchestrator.run()

        # 清理资源
        orchestrator.cleanup()
    """

    def __init__(
        self,
        system,  # EyeDistanceSystem instance
        emotion_engine: Optional[EmotionEngine] = None,
        fatigue_engine: Optional[FatigueEngine] = None,
        pose_engine: Optional[PoseEngine] = None,
        pose_detector: Optional[Any] = None,  # TRTPoseDetector instance（用于 person 检测）
        window_manager: Optional[WindowManager] = None,
        telemetry_builder: Optional[TelemetryBuilder] = None,
        system_config: Optional[Dict[str, Any]] = None,
        config_manager: Optional['ConfigManager'] = None,
        visual_switch: Optional['RuntimeSwitch'] = None
    ):
        """
        初始化检测协调器

        Args:
            system: EyeDistanceSystem 实例（必需）
            emotion_engine: 情绪检测引擎（可选）
            fatigue_engine: 疲劳检测引擎（可选）
            pose_engine: 姿态检测引擎（可选）
            window_manager: 窗口管理器（可选，None 表示无头模式）
            telemetry_builder: 遥测构建器（可选）
            system_config: 系统配置字典（可选）
            config_manager: 配置管理器（可选）
            visual_switch: 可视化运行时开关（可选，用于热切换 UI）
        """
        # 核心组件
        self.system = system
        self.emotion_engine = emotion_engine
        self.fatigue_engine = fatigue_engine
        self.pose_engine = pose_engine
        self.pose_detector = pose_detector  # 用于检测 person bbox（姿态检测）
        self.window_manager = window_manager
        self.telemetry_builder = telemetry_builder
        self.system_config = system_config or {}
        self.config_manager = config_manager

        # 可视化热切换支持
        self.visual_switch = visual_switch
        self._unsubscribe = None

        # 订阅可视化开关变化
        if self.visual_switch:
            self._unsubscribe = self.visual_switch.subscribe(self._on_visual_change)
            logger.info("Orchestrator 已订阅可视化开关")

        # 信号处理标志
        self.shutdown_requested = False

        # 循环控制
        self.paused = False
        self.visualization = None

        # 人脸检测状态跟踪
        self.previous_face_detected = False

        # ROI 管理器（统一的人脸裁剪逻辑）
        self.roi_manager = ROIManager(margin=0.0, min_size=20)

        # ========== SimpleTracker 初始化 ==========
        self.tracker = None
        self.tracking_enabled = False

        tracking_cfg = self.system_config.get('tracking', {})
        _tracking_cfg_enabled = tracking_cfg.get('enabled', False)

        if _tracking_cfg_enabled and _TRACKER_AVAILABLE and self.pose_detector:
            try:
                self.tracker = SimpleTracker(
                    depth_range_mm=(
                        tracking_cfg.get('depth_range_min', 200),
                        tracking_cfg.get('depth_range_max', 1500)
                    ),
                    switch_depth_delta_mm=tracking_cfg.get('switch_depth_delta_mm', 100),
                    min_keep_frames=tracking_cfg.get('min_keep_frames', 15),
                )
                self.tracking_enabled = True
                logger.info("SimpleTracker 初始化成功")
            except Exception as e:
                logger.warning(f"SimpleTracker 初始化失败: {e}")
                self.tracker = None
                self.tracking_enabled = False
        elif _tracking_cfg_enabled and not _TRACKER_AVAILABLE:
            logger.warning("tracking 模块不可用，使用原 Face 检测模式")
        elif _tracking_cfg_enabled and not self.pose_detector:
            logger.warning("pose_detector 未提供，无法启用追踪模式")
        # ==========================================

        # 帧率监控器（替代手动 FPS 计算）
        self.fps_monitor = FrameRateMonitor(smoothing=0.9)
        self.global_fps = 0.0  # 缓存值（从 fps_monitor 获取）
        self.global_frame_interval = 0.0  # 缓存值（从 fps_monitor 获取）

        # 心跳计数
        self.heartbeat = 0

        # ========== 异步推理调度器 ==========
        self.inference_scheduler = None
        self.async_enabled = False

        # 尝试从配置读取异步推理设置
        async_config = self.system_config.get('performance', {}).get('async_inference', {})
        async_enabled = async_config.get('enabled', False)

        if async_enabled and all([emotion_engine, fatigue_engine, pose_engine]):
            scheduler_config = SchedulerConfig(
                enabled=True,
                max_queue_size=async_config.get('max_queue_size', 1),
                enable_detailed_logging=async_config.get('enable_detailed_logging', False)
            )

            self.inference_scheduler = InferenceScheduler(
                emotion_engine=emotion_engine,
                fatigue_engine=fatigue_engine,
                pose_engine=pose_engine,
                config=scheduler_config
            )
            self.async_enabled = True
            logger.info("DetectionOrchestrator: 异步推理已启用")
        else:
            if async_enabled:
                logger.warning("DetectionOrchestrator: 异步推理配置为启用，但缺少必需的引擎，回退到同步模式")
            logger.info("DetectionOrchestrator: 使用同步推理模式")

        # ========== 配置参数 ==========
        # 使用配置管理器（如果可用）
        if self.config_manager:
            self.UI_ENABLED = self.config_manager.get_bool('ui.enabled')
            self.STAY_OPEN = self.config_manager.get_bool('ui.stay_open')
            self.HEARTBEAT_INTERVAL = self.config_manager.get_int('system.heartbeat_interval')
        else:
            # 回退到环境变量
            self.UI_ENABLED = os.getenv("AITABLE_ENABLE_UI", "1") == "1"
            self.STAY_OPEN = os.getenv("AITABLE_STAY_OPEN", "1") == "1"
            self.HEARTBEAT_INTERVAL = 150

        # 注意：去抖阈值、空帧计数等已移至 WindowManager 管理

        logger.info("DetectionOrchestrator: 已初始化")
        logger.info(f"  - 追踪模式: {'启用' if self.tracking_enabled else '禁用'}")
        logger.info(f"  - 情绪检测: {'启用' if emotion_engine else '禁用'}")
        logger.info(f"  - 疲劳检测: {'启用' if fatigue_engine else '禁用'}")
        logger.info(f"  - 姿态检测: {'启用' if pose_engine else '禁用'}")
        logger.info(f"  - 窗口管理: {'启用' if window_manager else '禁用（无头模式）'}")
        logger.info(f"  - 遥测构建: {'启用' if telemetry_builder else '禁用'}")

    def _signal_handler(self, signum, frame):
        """信号处理器：捕获 SIGINT/SIGTERM 用于优雅退出"""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM" if signum == signal.SIGTERM else f"Signal {signum}"
        logger.info(f"退出原因: 收到信号 {sig_name}")
        self.shutdown_requested = True

    def run(self):
        """
        运行主循环（原 run_main_loop 的完整实现）
        """
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 启动异步推理调度器（如果启用）
        if self.inference_scheduler:
            self.inference_scheduler.start()
            logger.info("异步推理调度器已启动")

        logger.info("=" * 60)
        logger.info("DetectionOrchestrator: 开始主循环")
        logger.info("=" * 60)

        try:
            import cv2  # 导入在需要时

            while True:
                # 检查信号退出标志（优先级最高）
                if self.shutdown_requested:
                    break

                # 全局帧率计算（使用 FrameRateMonitor）
                self.fps_monitor.update()
                self.global_fps = self.fps_monitor.get_fps()
                self.global_frame_interval = self.fps_monitor.get_interval()

                if not self.paused:
                    # 取流（使用新的非阻塞接口）
                    frame_data = self.system.camera.get_latest_frame()

                    # 获取相机状态（用于恢复提示）
                    camera_status = self.system.camera.get_camera_status()

                    if frame_data is None:
                        # 记录空帧（使用 WindowManager）
                        if self.window_manager:
                            self.window_manager.on_empty_frame()
                            streak = self.window_manager.empty_frame_streak
                        else:
                            streak = 0  # 无头模式不跟踪

                        if streak % 30 == 0 and streak > 0:
                            # 增强的断流日志（使用 ResourceMonitor 的静态方法）
                            rss_mb, vms_mb = ResourceMonitor.read_process_memory(os.getpid())
                            mem_info = f"RAM: RSS={rss_mb:.1f}MB VMS={vms_mb:.1f}MB" if rss_mb else "RAM: N/A"
                            logger.warning(
                                f"[FrameFetch] 相机返回空帧（连续 {streak} 次）| "
                                f"相机状态: {camera_status} | "
                                f"{mem_info} | 时间: {datetime.now().strftime('%H:%M:%S')}"
                            )

                        # 根据相机状态生成提示信息
                        if camera_status in ("soft_restarting", "reopening", "recovering"):
                            overlay_text = f"Camera recovering ({camera_status})..."
                        else:
                            overlay_text = "Waiting for camera..."

                        # 根据可视化开关决定是否显示
                        if self.window_manager and self.window_manager.enabled:
                            self.window_manager.show(self.visualization, overlay_text=overlay_text)
                            # 轮询键盘（检查退出请求）
                            status = self.window_manager.poll_keys_and_health()
                            if status == "quit":
                                break
                        else:
                            time.sleep(0.001)  # 避免忙等
                        # 注意：空帧退出检查已移至 WindowManager.check_exit_conditions()
                        continue
                    else:
                        # 记录有效帧（重置空帧计数）
                        if self.window_manager:
                            self.window_manager.on_valid_frame()

                    # 解码
                    rgb_frame, depth_frame = self.system.image_processor.process_frame_data(frame_data)
                    if rgb_frame is None or depth_frame is None:
                        # 记录无效帧
                        if self.window_manager:
                            self.window_manager.on_invalid_frame()
                            streak = self.window_manager.invalid_frame_streak
                        else:
                            streak = 0

                        if streak % 30 == 0 and streak > 0:
                            frame_status = f"RGB={'OK' if rgb_frame is not None else 'FAIL'} Depth={'OK' if depth_frame is not None else 'FAIL'}"
                            logger.warning(
                                f"[FrameDecode] 图像数据不完整（连续 {streak} 次）| "
                                f"{frame_status} | 时间: {datetime.now().strftime('%H:%M:%S')}"
                            )
                        # 根据可视化开关决定是否显示
                        if self.window_manager and self.window_manager.enabled:
                            self.window_manager.show(self.visualization, overlay_text="Waiting for valid frame...")
                            # 轮询键盘（检查退出请求）
                            status = self.window_manager.poll_keys_and_health()
                            if status == "quit":
                                break
                        else:
                            time.sleep(0.001)  # 避免忙等
                        continue
                    # 帧解码成功（无效帧计数可选重置，这里不重置因为WindowManager已管理）

                    # ========== 单帧检测流程 ==========
                    results = None
                    vis_from_system = None
                    current_face_detected = False
                    face_bbox = None
                    face_roi = None
                    face_appeared = False

                    # ========== SimpleTracker 模式 vs Face 检测模式 ==========
                    if self.tracking_enabled and self.tracker:
                        # --- SimpleTracker 模式 ---
                        pose_raw = self.pose_detector.detect_keypoints(rgb_frame)

                        if pose_raw and pose_raw.get('detections'):
                            self.tracker.update(pose_raw['detections'], depth_frame)
                        else:
                            self.tracker.update([], depth_frame)

                        target = self.tracker.get_primary_target()

                        if target:
                            current_face_detected = True
                            face_appeared = not self.previous_face_detected

                            results, vis_from_system = self._process_with_tracker_target(
                                target, rgb_frame, depth_frame
                            )

                            if target.face_valid and target.face_bbox:
                                x1, y1, x2, y2 = target.face_bbox
                                face_bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                            elif target.bbox:
                                x1, y1, x2, y2 = target.bbox
                                head_h = (y2 - y1) / 6
                                face_bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y1 + head_h)}

                            if face_bbox:
                                rois = self.roi_manager.extract_dual(rgb_frame, face_bbox=face_bbox, person_bbox=None)
                                face_roi = rois.get('face_roi')
                        else:
                            vis_from_system = self.system.visualizer.draw_no_detection(rgb_frame.copy(), "Tracker")
                            self.previous_face_detected = False

                    else:
                        # --- 原 Face 检测模式 ---
                        detection = self.system.face_detector.detect_face(rgb_frame)

                        if detection:
                            current_face_detected = True
                            face_bbox = detection['bbox']
                            face_appeared = not self.previous_face_detected
                            results, vis_from_system = self.system.process_frame(rgb_frame, depth_frame)

                            if face_bbox:
                                rois = self.roi_manager.extract_dual(rgb_frame, face_bbox=face_bbox, person_bbox=None)
                                face_roi = rois.get('face_roi')
                        else:
                            vis_from_system = self.system.visualizer.draw_no_detection(rgb_frame.copy(), "YOLO")
                            self.previous_face_detected = False

                    # ========== 推理执行（同步/异步分支）==========
                    if self.async_enabled and self.inference_scheduler:
                        # 异步模式：提交任务到调度器
                        self.inference_scheduler.submit(
                            rgb_frame=rgb_frame,
                            depth_frame=depth_frame,
                            face_bbox=face_bbox,
                            face_roi=face_roi,
                            face_present=current_face_detected,
                            face_just_appeared=face_appeared,
                            frame_count=self.system.frame_count,
                            global_frame_interval=self.global_frame_interval,
                            global_fps=self.global_fps,
                            person_roi=None,
                            person_bbox=None
                        )

                        # 收集最新结果
                        async_results = self.inference_scheduler.collect_results()

                        # 提取情绪结果
                        emotion_results = None
                        emotion_source = None
                        emotion_latency_ms = 0.0
                        emotion_speed_fps = 0.0
                        if async_results.get('emotion'):
                            emotion_wrapper = async_results['emotion']
                            if emotion_wrapper.result:
                                emotion_results = emotion_wrapper.result.data
                                emotion_source = f"{emotion_wrapper.source}"
                                emotion_latency_ms = emotion_wrapper.result.latency_ms
                                emotion_speed_fps = emotion_wrapper.result.speed_fps

                        # 提取疲劳结果
                        fatigue_results = None
                        if async_results.get('fatigue'):
                            fatigue_wrapper = async_results['fatigue']
                            if fatigue_wrapper.result:
                                fatigue_results = fatigue_wrapper.result.data

                        # 提取姿态结果
                        pose_results = None
                        if async_results.get('pose'):
                            pose_wrapper = async_results['pose']
                            if pose_wrapper.result:
                                pose_results = pose_wrapper.result.data

                    else:
                        # 同步模式：直接调用引擎
                        # 情绪识别
                        emotion_results = None
                        emotion_source = None
                        emotion_latency_ms = 0.0
                        emotion_speed_fps = 0.0
                        if self.emotion_engine:
                            emotion_result = self.emotion_engine.maybe_infer(
                                rgb_frame=rgb_frame,
                                face_bbox=face_bbox,
                                face_present=current_face_detected,
                                face_just_appeared=face_appeared,
                                face_roi=face_roi  # 传递预提取的 ROI
                            )
                            emotion_results = emotion_result.data
                            emotion_source = emotion_result.source
                            emotion_latency_ms = emotion_result.latency_ms
                            emotion_speed_fps = emotion_result.speed_fps

                        # 疲劳检测
                        fatigue_results = None
                        if self.fatigue_engine:
                            fatigue_result = self.fatigue_engine.infer(
                                rgb_frame=rgb_frame,
                                face_bbox=face_bbox,
                                face_present=current_face_detected,
                                face_roi=face_roi  # 传递预提取的 ROI
                            )
                            fatigue_results = fatigue_result.data

                        # 姿态检测（使用 PoseEngine）
                        pose_results = None
                        if self.pose_engine:
                            pose_result = self.pose_engine.maybe_infer(
                                rgb_frame=rgb_frame,
                                depth_frame=depth_frame,
                                face_present=current_face_detected,
                                global_frame_interval=self.global_frame_interval,
                                frame_count=self.system.frame_count,
                                global_fps=self.global_fps,
                                person_roi=None,
                                person_bbox=None
                            )
                            pose_results = pose_result.data

                    # 更新人脸检测状态（用于下一帧）
                    self.previous_face_detected = current_face_detected

                    # 合并结果
                    if results:
                        results['emotion'] = emotion_results
                        results['emotion_enabled'] = self.emotion_engine is not None
                        results['emotion_source'] = emotion_source if emotion_results else None
                        results['emotion_latency_ms'] = emotion_latency_ms if emotion_latency_ms > 0 else None
                        results['emotion_speed_fps'] = emotion_speed_fps if emotion_speed_fps > 0 else None
                        results['fatigue'] = fatigue_results
                        results['fatigue_enabled'] = self.fatigue_engine is not None
                        results['pose'] = pose_results
                        results['pose_enabled'] = self.pose_engine is not None

                        # 添加全局和相机 FPS
                        results['global_fps'] = self.global_fps if self.global_fps > 0 else None
                        results['camera_fps'] = self.system.camera.camera_fps if self.system.camera.camera_fps > 0 else None

                        # ========== 创建 telemetry ==========
                        stable_dist = results.get("stable_distance")

                        # 计算肩部中点
                        left_shoulder = results.get("pose", {}).get("keypoints_3d", {}).get("left_shoulder")
                        right_shoulder = results.get("pose", {}).get("keypoints_3d", {}).get("right_shoulder")
                        shoulder_midpoint = None
                        shoulder_midpoint_to_desk = None

                        if left_shoulder is not None and right_shoulder is not None:
                            midpoint_vec = (np.asarray(left_shoulder, dtype=float) + np.asarray(right_shoulder, dtype=float)) / 2.0
                            shoulder_midpoint = midpoint_vec.tolist()
                            shoulder_midpoint_to_desk = self.system.calculate_distance_to_plane(midpoint_vec)

                        # 使用 TelemetryBuilder 构建遥测
                        if self.telemetry_builder:
                            # 获取异步统计（如果启用）
                            async_stats = None
                            if self.inference_scheduler:
                                async_stats = self.inference_scheduler.get_stats()

                            # 获取相机恢复统计
                            camera_telemetry = self.system.camera.get_telemetry()

                            telemetry = self.telemetry_builder.build(
                                results=results,
                                stable_dist=stable_dist,
                                shoulder_midpoint=shoulder_midpoint,
                                left_shoulder=left_shoulder,
                                right_shoulder=right_shoulder,
                                shoulder_midpoint_to_desk=shoulder_midpoint_to_desk,
                                frame_count=self.system.frame_count,
                                global_fps=self.global_fps,
                                async_stats=async_stats,
                                camera_telemetry=camera_telemetry
                            )
                            results['telemetry'] = telemetry

                        # 可视化
                        if (emotion_results is not None) or (fatigue_results is not None and fatigue_results.get('enabled', False)):
                            if self.fatigue_engine and fatigue_results and fatigue_results.get('enabled', False):
                                self.visualization = self.system.visualizer.draw_combined_visualization(
                                    rgb_frame.copy(), results, self.fatigue_engine.detector
                                )
                            else:
                                self.visualization = self.system.visualizer.draw_visualization(
                                    rgb_frame.copy(), results, "YOLO Face Model"
                                )
                        else:
                            self.visualization = vis_from_system
                        # 有结果（no_result_streak 重置由 WindowManager 管理）
                    else:
                        # 无检测结果
                        if self.window_manager:
                            self.window_manager.on_no_result()
                            streak = self.window_manager.no_result_streak
                        else:
                            streak = 0

                        if streak % 30 == 0 and streak > 0:
                            logger.warning(f"处理结果为空（连续 {streak} 次）")

                        # 创建空 telemetry
                        if self.telemetry_builder:
                            # 获取相机恢复统计
                            camera_telemetry = self.system.camera.get_telemetry()

                            self.telemetry_builder.build_empty(
                                frame_count=self.system.frame_count,
                                global_fps=self.global_fps,
                                reuse_last=True,
                                camera_telemetry=camera_telemetry
                            )

                        # 重置人脸状态
                        self.previous_face_detected = False

                        # 根据可视化开关决定是否显示
                        if self.window_manager and self.window_manager.enabled:
                            self.window_manager.show(vis_from_system)
                            # 轮询键盘（检查退出请求）
                            status = self.window_manager.poll_keys_and_health()
                            if status == "quit":
                                break
                        else:
                            time.sleep(0.001)  # 避免忙等
                        continue

                    # 显示 (根据可视化开关状态)
                    if self.window_manager and self.window_manager.enabled:
                        self.window_manager.show(self.visualization)

                # ===== 键盘/窗口逻辑 (仅在 UI 启用时处理) =====
                if self.window_manager and self.window_manager.enabled:
                    # 定义回调函数
                    def on_reset():
                        """重置系统和所有引擎"""
                        self.system.reset()
                        if self.emotion_engine:
                            self.emotion_engine.reset()
                        if self.fatigue_engine:
                            self.fatigue_engine.reset()

                    def on_screenshot():
                        """保存当前帧截图"""
                        if self.visualization is not None:
                            self.window_manager.save_screenshot(self.visualization)

                    # 轮询键盘和窗口健康状态
                    status = self.window_manager.poll_keys_and_health(
                        on_reset_callback=on_reset,
                        on_screenshot_callback=on_screenshot
                    )

                    # 更新暂停状态
                    self.paused = self.window_manager.is_paused()

                    # 检查退出状态
                    if status == "quit":
                        break

                # 无头模式休眠
                if not self.UI_ENABLED:
                    time.sleep(0.001)

                # 心跳日志（每 ~5s 打一次，使用配置的间隔）
                self.heartbeat += 1
                if self.heartbeat % self.HEARTBEAT_INTERVAL == 0:
                    # 心跳日志（帧计数由 WindowManager 管理）
                    if self.window_manager:
                        stats = self.window_manager.get_frame_stats()
                        logger.info(f"[KeepAlive] "
                                    f"empty={stats['empty_frame_streak']} "
                                    f"invalid={stats['invalid_frame_streak']}")
        finally:
            self.cleanup()

    def _on_visual_change(self, enabled: bool):
        """
        可视化状态变化回调

        Args:
            enabled: 新状态 (True=启用, False=禁用)
        """
        if not self.window_manager:
            return

        if enabled:
            logger.info("可视化已启用: 创建窗口")
            self.window_manager.enable()
        else:
            logger.info("可视化已禁用: 销毁窗口 (推理继续)")
            self.window_manager.disable()

    def _process_with_tracker_target(self, target, rgb_frame, depth_frame):
        """使用 SimpleTracker 的目标计算眼距"""
        if target is None:
            return None, self.system.visualizer.draw_no_detection(rgb_frame.copy(), "Tracker")

        self.system.frame_count += 1
        start_time = time.time()
        self.system.update_camera_resolution(rgb_frame)

        # 从 target 提取眼睛位置
        left_eye_pos = (int(target.left_eye[0]), int(target.left_eye[1])) if target.left_eye else None
        right_eye_pos = (int(target.right_eye[0]), int(target.right_eye[1])) if target.right_eye else None

        # 计算距离
        distances = []
        eye_results = {}
        depth_available = False

        for eye_name, eye_pos in [('left', left_eye_pos), ('right', right_eye_pos)]:
            if eye_pos is None:
                eye_results[eye_name] = {'position': None, 'depth': None, 'coord_3d': None, 'distance': None}
                continue

            depth_value = self.system.get_robust_depth(depth_frame, eye_pos[0], eye_pos[1])

            if depth_value:
                coord_3d = self.system.pixel_to_3d(eye_pos[0], eye_pos[1], depth_value)
                distance = self.system.calculate_distance_to_plane(coord_3d)
                distances.append(distance)
                eye_results[eye_name] = {
                    'position': eye_pos, 'depth': depth_value,
                    'coord_3d': coord_3d, 'distance': distance
                }
                depth_available = True
            else:
                eye_results[eye_name] = {'position': eye_pos, 'depth': None, 'coord_3d': None, 'distance': None}

        raw_distance = np.mean(distances) if distances else None
        stable_distance = self.system.distance_processor.add_measurement(raw_distance) if raw_distance else None
        stability_score = self.system.distance_processor.get_stability_score()

        process_time = time.time() - start_time
        self.system.processing_times.append(process_time)

        # 构建 detection
        detection = None
        if target.face_bbox:
            x1, y1, x2, y2 = target.face_bbox
            detection = {
                'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                'confidence': target.face_confidence,
                'left_eye': left_eye_pos, 'right_eye': right_eye_pos,
            }
        elif target.bbox:
            x1, y1, x2, y2 = target.bbox
            detection = {
                'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                'confidence': target.confidence,
                'left_eye': left_eye_pos, 'right_eye': right_eye_pos,
            }

        results = {
            'frame': self.system.frame_count,
            'raw_distance': raw_distance,
            'stable_distance': stable_distance,
            'stability_score': stability_score,
            'eye_results': eye_results,
            'detection': detection,
            'process_time': process_time,
            'fps': 1.0 / np.mean(self.system.processing_times) if self.system.processing_times else 0,
            'depth_available': depth_available,
            'track_id': target.track_id,
        }

        visualization = self.system.visualizer.draw_visualization(rgb_frame.copy(), results, "Pose Tracker")
        return results, visualization

    def cleanup(self):
        """清理资源"""
        # 停止异步推理调度器
        if self.inference_scheduler:
            logger.info("正在停止异步推理调度器...")
            self.inference_scheduler.stop(timeout=5.0)
            # 检查是否有错误
            errors = self.inference_scheduler.check_errors()
            if errors:
                logger.warning(f"异步推理调度器退出时有错误: {errors}")

        # 取消订阅可视化开关
        if self._unsubscribe:
            self._unsubscribe()
            logger.debug("已取消订阅可视化开关")

        # 清理窗口资源
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        logger.info("DetectionOrchestrator: 资源清理完成")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取所有引擎的统计信息

        Returns:
            统计字典
        """
        stats = {}

        if self.emotion_engine:
            stats['emotion'] = self.emotion_engine.get_stats()

        if self.fatigue_engine:
            stats['fatigue'] = self.fatigue_engine.get_stats()

        if self.pose_engine:
            stats['pose'] = self.pose_engine.get_stats()

        stats['global_fps'] = self.global_fps

        # 添加异步推理统计
        if self.inference_scheduler:
            stats['async_inference'] = self.inference_scheduler.get_stats()
            stats['async_enabled'] = True
        else:
            stats['async_enabled'] = False

        return stats
