"""
推理调度器
统一管理多个异步推理引擎，协调任务提交和结果收集
"""

import time
import numpy as np
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

from modules.core.async_engine import AsyncEngineRunner, InferenceTask, InferenceResult
from modules.core.logger import logger


@dataclass
class SchedulerConfig:
    """调度器配置"""
    enabled: bool = True
    max_queue_size: int = 1
    enable_detailed_logging: bool = False


class InferenceScheduler:
    """
    推理调度器

    职责:
    - 管理多个 AsyncEngineRunner (emotion/fatigue/pose)
    - 根据配置决定是否提交推理任务
    - 收集各引擎的最新结果
    - 提供统一的生命周期管理
    """

    def __init__(
        self,
        emotion_engine: Any,
        fatigue_engine: Any,
        pose_engine: Any,
        config: SchedulerConfig
    ):
        """
        初始化推理调度器

        Args:
            emotion_engine: EmotionEngine 实例
            fatigue_engine: FatigueEngine 实例
            pose_engine: PoseEngine 实例
            config: 调度器配置
        """
        self.config = config
        self.enabled = config.enabled

        # 保存引擎引用（用于同步模式）
        self.emotion_engine = emotion_engine
        self.fatigue_engine = fatigue_engine
        self.pose_engine = pose_engine

        # 任务计数器
        self._task_counter = 0

        # 独立的帧计数器（每个引擎维护，避免引擎内部计数器死锁）
        # 关键修复：调度器每帧递增，不依赖 maybe_infer 是否执行
        self._emotion_frame_count = 0
        self._pose_frame_count = 0

        if not self.enabled:
            logger.info("[InferenceScheduler] Async inference disabled, using sync mode")
            self.runners = {}
            return

        logger.info("[InferenceScheduler] Initializing async inference schedulers")

        # 创建异步执行器
        self.runners: Dict[str, AsyncEngineRunner] = {}

        # Emotion Runner
        self.runners['emotion'] = AsyncEngineRunner(
            engine=emotion_engine,
            engine_name='EmotionEngine',
            should_infer_func=self._should_infer_emotion,
            infer_func=self._infer_emotion,
            max_queue_size=config.max_queue_size,
            enable_logging=config.enable_detailed_logging
        )

        # Fatigue Runner
        self.runners['fatigue'] = AsyncEngineRunner(
            engine=fatigue_engine,
            engine_name='FatigueEngine',
            should_infer_func=self._should_infer_fatigue,
            infer_func=self._infer_fatigue,
            max_queue_size=config.max_queue_size,
            enable_logging=config.enable_detailed_logging
        )

        # Pose Runner
        self.runners['pose'] = AsyncEngineRunner(
            engine=pose_engine,
            engine_name='PoseEngine',
            should_infer_func=self._should_infer_pose,
            infer_func=self._infer_pose,
            max_queue_size=config.max_queue_size,
            enable_logging=config.enable_detailed_logging
        )

        logger.info("[InferenceScheduler] All async runners initialized")

    def start(self):
        """启动所有异步执行器"""
        if not self.enabled:
            return

        logger.info("[InferenceScheduler] Starting all async workers")
        for name, runner in self.runners.items():
            runner.start()
        logger.info("[InferenceScheduler] All async workers started")

    def stop(self, timeout: float = 5.0):
        """停止所有异步执行器"""
        if not self.enabled:
            return

        logger.info("[InferenceScheduler] Stopping all async workers")
        for name, runner in self.runners.items():
            runner.stop(timeout=timeout)
        logger.info("[InferenceScheduler] All async workers stopped")

    def submit(
        self,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray],
        face_bbox: Optional[Dict],
        face_roi: Optional[Any],
        face_present: bool,
        face_just_appeared: bool,
        frame_count: int,
        global_frame_interval: float,
        global_fps: float,
        person_roi: Optional[Any] = None,
        person_bbox: Optional[Dict] = None
    ):
        """
        提交当前帧到各推理引擎

        调度器会根据引擎配置决定是否实际提交任务

        Args:
            rgb_frame: RGB图像
            depth_frame: 深度图像
            face_bbox: 人脸边界框 {"x_min", "y_min", "x_max", "y_max"}
            face_roi: 预提取的人脸 ROI（RegionROI 对象）
            face_present: 是否检测到人脸
            face_just_appeared: 人脸是否刚出现
            frame_count: 全局帧计数
            global_frame_interval: 全局帧间隔（秒）
            global_fps: 全局FPS
            person_roi: 预提取的 person ROI（RegionROI 对象，用于姿态检测）
            person_bbox: person 边界框（用于姿态检测）
        """
        if not self.enabled:
            # 同步模式，不做任何操作
            return

        # 递增帧计数器（关键修复：每帧递增，避免死锁）
        self._emotion_frame_count += 1
        self._pose_frame_count += 1

        self._task_counter += 1
        submit_time = time.time()

        # 优化内存复制策略（关键修复：避免全帧复制）
        # 只传递引用，后台线程会在需要时复制 ROI
        # 主循环不会修改这些帧，因此是线程安全的
        task = InferenceTask(
            task_id=self._task_counter,
            rgb_frame=rgb_frame,  # 直接传递引用（零拷贝）
            depth_frame=depth_frame,  # 直接传递引用
            face_bbox=face_bbox.copy() if face_bbox is not None else None,  # bbox 很小，可以复制
            face_roi=face_roi,  # RegionROI 对象（已是不可变的，可以安全共享）
            face_present=face_present,
            face_just_appeared=face_just_appeared,
            frame_count=frame_count,
            global_frame_interval=global_frame_interval,
            global_fps=global_fps,
            submit_time=submit_time,
            person_roi=person_roi,  #  Person ROI（用于姿态检测）
            person_bbox=person_bbox.copy() if person_bbox is not None else None  #  Person bbox
        )

        # 提交到各引擎
        # 注意：should_infer 判断在后台线程中进行，这里直接提交
        # 如果队列满，submit() 会自动丢弃

        # Emotion: 只在有人脸时提交
        if face_present:
            self.runners['emotion'].submit(task)

        # Fatigue: 只在有人脸时提交
        if face_present:
            self.runners['fatigue'].submit(task)

        # Pose: 始终提交（内部会根据face_present判断）
        self.runners['pose'].submit(task)

    def collect_results(self) -> Dict[str, Any]:
        """
        收集各引擎的最新结果

        Returns:
            Dict: {
                'emotion': InferenceResult or None,
                'fatigue': InferenceResult or None,
                'pose': InferenceResult or None
            }
        """
        if not self.enabled:
            # 同步模式，返回空结果（orchestrator 会直接调用引擎）
            return {
                'emotion': None,
                'fatigue': None,
                'pose': None
            }

        results = {}
        for name, runner in self.runners.items():
            results[name] = runner.get_latest()

        return results

    def get_stats(self) -> Dict[str, Dict]:
        """
        获取所有引擎的统计信息

        Returns:
            Dict: {
                'emotion': {...},
                'fatigue': {...},
                'pose': {...}
            }
        """
        if not self.enabled:
            return {}

        stats = {}
        for name, runner in self.runners.items():
            stats[name] = runner.get_stats()

        return stats

    def check_errors(self) -> Optional[str]:
        """
        检查是否有引擎发生错误

        Returns:
            str or None: 错误信息，如果没有错误返回None
        """
        if not self.enabled:
            return None

        errors = []
        for name, runner in self.runners.items():
            if runner.last_exception is not None:
                errors.append(f"[{name}] {runner.last_exception}")

        return "; ".join(errors) if errors else None

    # ========== 引擎特定的 should_infer 和 infer 函数 ==========

    def _should_infer_emotion(self, task: InferenceTask) -> tuple[bool, str]:
        """
        判断是否需要执行情绪推理

        关键修复：使用调度器维护的帧计数器，避免引擎内部计数器死锁
        """
        # 引擎禁用检查
        if self.emotion_engine.classifier is None:
            return (False, "engine_disabled")

        # 无人脸检查
        if not task.face_present:
            return (False, "no_face")

        # 判断是否需要推理（使用调度器的帧计数器）
        cache_expired = (time.time() - self.emotion_engine._last_timestamp) > self.emotion_engine.cache_timeout

        if self.emotion_engine._last_result is None:
            return (True, "first_inference")
        elif cache_expired:
            return (True, "cache_expired")
        elif task.face_just_appeared:
            return (True, "face_appeared")
        elif self._emotion_frame_count % self.emotion_engine.interval_frames == 0:
            return (True, "interval_reached")
        else:
            return (False, "use_cache")

    def _infer_emotion(self, task: InferenceTask):
        """
        执行情绪推理

        关键修复：直接执行推理逻辑，绕过 maybe_infer() 的内部帧计数器
        这样可以避免引擎内部 _frame_count 死锁问题（_frame_count 只在 maybe_infer() 中递增）
        """
        import time
        from modules.engines.emotion_engine import EmotionResult

        # 引擎禁用
        if self.emotion_engine.classifier is None:
            return EmotionResult(data=None, latency_ms=0.0, speed_fps=0.0,
                               source="disabled", timestamp=time.time())

        # 无人脸，返回缓存结果
        if not task.face_present or task.face_roi is None:
            return self.emotion_engine._get_cached_result()

        try:
            # 使用预提取的 ROI（避免重复裁剪）
            # RegionROI.image 已经是副本且只读，可以安全使用
            face_roi_img = task.face_roi.image

            # 直接调用 classifier，跳过 maybe_infer()
            infer_start = time.time()
            batch_results = self.emotion_engine.classifier.predict_batch([face_roi_img])
            infer_end = time.time()

            latency_ms = (infer_end - infer_start) * 1000.0

            # 更新引擎缓存和统计（手动维护引擎状态）
            self.emotion_engine._last_result = batch_results[0]
            self.emotion_engine._last_timestamp = time.time()

            # 更新延迟统计（使用引擎的 EMA 平滑器）
            # 注意：EmotionEngine 使用 _latency_ms 而非 _latency_ema
            alpha = self.emotion_engine.latency_smoothing
            if self.emotion_engine._latency_ms == 0.0:
                self.emotion_engine._latency_ms = latency_ms
            else:
                self.emotion_engine._latency_ms = alpha * latency_ms + (1 - alpha) * self.emotion_engine._latency_ms

            # 构造结果
            # 注意：使用平滑后的 _latency_ms，保持与同步模式一致
            return EmotionResult(
                data=batch_results[0],
                latency_ms=self.emotion_engine._latency_ms,  # 使用平滑值
                speed_fps=1000.0 / self.emotion_engine._latency_ms if self.emotion_engine._latency_ms > 0 else 0.0,
                source="async_scheduler",
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"[InferenceScheduler._infer_emotion] Error: {e}")
            return self.emotion_engine._get_cached_result()

    def _should_infer_fatigue(self, task: InferenceTask) -> tuple[bool, str]:
        """判断是否需要执行疲劳检测"""
        # 调用 FatigueEngine 的 should_infer 方法
        return self.fatigue_engine.should_infer(
            face_present=task.face_present
        )

    def _infer_fatigue(self, task: InferenceTask):
        """
        执行疲劳检测

        关键修复：直接执行推理逻辑，绕过 infer() 的内部逻辑
        Fatigue 每帧执行，所以不涉及帧计数器问题，但保持一致的 ROI 优化
        """
        import time
        from modules.engines.fatigue_engine import FatigueResult

        # 引擎禁用
        if self.fatigue_engine.detector is None:
            return FatigueResult(
                data={},
                latency_ms=0.0,
                speed_fps=0.0,
                source="disabled",
                timestamp=time.time()
            )

        # 无人脸，返回默认结果
        if not task.face_present or task.face_roi is None:
            return FatigueResult(
                data={},
                latency_ms=0.0,
                speed_fps=0.0,
                source="no_face",
                timestamp=time.time()
            )

        try:
            # 使用预提取的 ROI（避免重复裁剪）
            # RegionROI.image 已经是副本且只读，可以安全使用
            face_roi_img = task.face_roi.image

            # 直接调用 detector.detect_fatigue（正确的方法名）
            infer_start = time.time()
            result_data = self.fatigue_engine.detector.detect_fatigue(face_roi_img, is_cropped_face=True)
            infer_end = time.time()

            latency_ms = (infer_end - infer_start) * 1000.0

            # 更新引擎统计（使用引擎的 EMA 平滑器）
            # 注意：FatigueEngine 使用 _latency_ms 而非 _latency_ema
            alpha = self.fatigue_engine.latency_smoothing
            if self.fatigue_engine._latency_ms == 0.0:
                self.fatigue_engine._latency_ms = latency_ms
            else:
                self.fatigue_engine._latency_ms = alpha * latency_ms + (1 - alpha) * self.fatigue_engine._latency_ms

            # 构造结果
            # 注意：使用平滑后的 _latency_ms，保持与同步模式一致
            return FatigueResult(
                data=result_data,
                latency_ms=self.fatigue_engine._latency_ms,  # 使用平滑值
                speed_fps=1000.0 / self.fatigue_engine._latency_ms if self.fatigue_engine._latency_ms > 0 else 0.0,
                source="async_scheduler",
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"[InferenceScheduler._infer_fatigue] Error: {e}")
            return FatigueResult(
                data={},
                latency_ms=0.0,
                speed_fps=0.0,
                source="error",
                timestamp=time.time()
            )

    def _should_infer_pose(self, task: InferenceTask) -> tuple[bool, str]:
        """
        判断是否需要执行姿态检测

        关键修复：使用调度器维护的帧计数器，避免引擎内部计数器死锁
        """
        # 引擎禁用检查
        if not self.pose_engine.pose_detector:
            return (False, "engine_disabled")

        # 计算数据年龄
        current_time = time.monotonic()
        pose_age_ms = (current_time - self.pose_engine._last_timestamp) * 1000.0

        # 判断是否需要推理（使用调度器的帧计数器）
        if self.pose_engine._last_result is None:
            return (True, "first_inference")
        elif pose_age_ms > self.pose_engine.cache_timeout * 1000:
            return (True, "cache_expired")
        elif self._pose_frame_count % self.pose_engine.interval_frames == 0:
            return (True, "interval_reached")
        else:
            return (False, "use_cache")

    def _infer_pose(self, task: InferenceTask):
        """
        执行姿态检测

        关键修复：调用 PoseEngine._run_inference() 复用完整推理流程
        这样可以避免引擎内部帧计数器问题，同时保持与同步模式一致的处理逻辑

        注意：虽然 _run_inference() 是私有方法，但这是最安全的方式，
        因为它包含了完整的深度转换、头部方向过滤、角度计算等逻辑
        """
        import time
        from modules.engines.pose_engine import PoseResult

        # 引擎禁用
        if not self.pose_engine.pose_detector:
            return PoseResult(
                data=None,
                latency_ms=0.0,
                speed_fps=0.0,
                source="disabled",
                timestamp=time.monotonic(),
                fresh=False,
                pose_age_ms=0.0
            )

        # 无 RGB 帧
        if task.rgb_frame is None:
            return self.pose_engine._get_cached_result()

        try:
            # 计算数据年龄
            current_time = time.monotonic()
            pose_age_ms = (current_time - self.pose_engine._last_timestamp) * 1000.0

            # 关键修复：强制复制完整帧，避免后台线程就地修改破坏主线程的 visualization
            # 虽然 image_processor 已经复制了 SDK buffer，但主线程和异步线程仍在共享同一帧
            # 如果 detector/pose 代码中有就地修改（如 cv2.cvtColor/cv2.resize in-place），
            # 会直接破坏主线程中的 visualization 内容
            #
            # 性能影响：Pose 需要复制完整帧（~10MB），耗时 ~0.1ms
            # 但这是保证线程安全的必要代价
            rgb_frame_copy = task.rgb_frame.copy() if task.rgb_frame is not None else None
            depth_frame_copy = task.depth_frame.copy() if task.depth_frame is not None else None

            # 直接调用 PoseEngine._run_inference()，复用完整推理流程
            # 这包括：2D检测、深度转换、头部方向过滤、角度计算、延迟平滑
            #  传递 person_roi 和 person_bbox，确保分析同一主角
            result = self.pose_engine._run_inference(
                rgb_frame=rgb_frame_copy,  # 传递副本（线程安全）
                depth_frame=depth_frame_copy,  # 传递副本（线程安全）
                pose_age_ms=pose_age_ms,
                global_frame_interval=task.global_frame_interval,
                frame_count=task.frame_count,
                global_fps=task.global_fps,
                person_roi=task.person_roi,  #  使用主角的 person ROI
                person_bbox=task.person_bbox  #  使用主角的 person bbox
            )

            return result

        except Exception as e:
            logger.error(f"[InferenceScheduler._infer_pose] Error: {e}")
            return self.pose_engine._get_cached_result()
