"""
异步推理执行器
提供后台线程执行推理任务，避免阻塞主循环
"""

import threading
import queue
import time
import traceback
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from modules.core.logger import logger


@dataclass
class InferenceTask:
    """推理任务封装"""
    task_id: int
    rgb_frame: Any
    depth_frame: Optional[Any]
    face_bbox: Optional[Dict]
    face_roi: Optional[Any]  # RegionROI object (预提取的人脸 ROI)
    face_present: bool
    face_just_appeared: bool
    frame_count: int
    global_frame_interval: float
    global_fps: float
    submit_time: float
    person_roi: Optional[Any] = None  #  Person ROI (用于姿态检测)
    person_bbox: Optional[Dict] = None  #  Person bbox (用于姿态检测)


@dataclass
class InferenceResult:
    """推理结果封装"""
    task_id: int
    result: Any  # EmotionResult/FatigueResult/PoseResult
    inference_time: float
    fresh: bool  # 是否是新推理结果
    source: str  # 'async_inference', 'async_cache', 'error', 'pending'


class AsyncEngineRunner:
    """
    异步引擎执行器

    负责在后台线程运行推理引擎，主线程只需提交任务和获取最新结果

    特性:
    - 单线程执行推理，避免GPU竞争
    - 任务队列最大深度为1，自动丢弃过时任务
    - 线程安全的结果读取
    - 支持优雅关闭
    """

    def __init__(
        self,
        engine: Any,
        engine_name: str,
        should_infer_func: Callable,
        infer_func: Callable,
        max_queue_size: int = 1,
        enable_logging: bool = True
    ):
        """
        初始化异步引擎执行器

        Args:
            engine: 推理引擎实例 (EmotionEngine/FatigueEngine/PoseEngine)
            engine_name: 引擎名称，用于日志
            should_infer_func: 判断是否需要推理的函数，签名 should_infer(task) -> (bool, str)
            infer_func: 实际推理函数，签名 infer(task) -> Result
            max_queue_size: 最大队列深度（建议1，避免积压）
            enable_logging: 是否启用详细日志
        """
        self.engine = engine
        self.engine_name = engine_name
        self.should_infer_func = should_infer_func
        self.infer_func = infer_func
        self.max_queue_size = max_queue_size
        self.enable_logging = enable_logging

        # 任务队列（FIFO）
        self.task_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # 最新结果（线程安全）
        self._result_lock = threading.Lock()
        self._latest_result: Optional[InferenceResult] = None

        # 工作线程
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 统计信息
        self._stats_lock = threading.Lock()
        self._stats = {
            'tasks_submitted': 0,
            'tasks_dropped': 0,
            'tasks_processed': 0,
            'total_inference_time': 0.0,
            'last_inference_time': 0.0,
            'errors': 0,
            'worker_active': False
        }

        # 最后异常
        self.last_exception: Optional[Exception] = None

    def start(self):
        """启动后台工作线程"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning(f"[{self.engine_name}] Worker thread already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.engine_name}_AsyncWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info(f"[{self.engine_name}] Async worker thread started")

    def stop(self, timeout: float = 5.0):
        """
        停止后台工作线程

        Args:
            timeout: 等待线程退出的超时时间（秒）
        """
        if self._worker_thread is None:
            return

        logger.info(f"[{self.engine_name}] Stopping async worker...")
        self._stop_event.set()

        # 清空队列，避免阻塞
        try:
            while not self.task_queue.empty():
                self.task_queue.get_nowait()
        except queue.Empty:
            pass

        # 等待线程结束
        self._worker_thread.join(timeout=timeout)
        if self._worker_thread.is_alive():
            logger.warning(f"[{self.engine_name}] Worker thread did not exit within {timeout}s")
        else:
            logger.info(f"[{self.engine_name}] Async worker stopped")

    def submit(self, task: InferenceTask) -> bool:
        """
        提交推理任务

        Args:
            task: 推理任务

        Returns:
            bool: 是否成功提交（队列满时会丢弃并返回False）
        """
        with self._stats_lock:
            self._stats['tasks_submitted'] += 1

        try:
            # 尝试立即放入队列（非阻塞）
            self.task_queue.put_nowait(task)
            if self.enable_logging:
                logger.debug(f"[{self.engine_name}] Task #{task.task_id} submitted")
            return True
        except queue.Full:
            # 队列满，丢弃任务
            with self._stats_lock:
                self._stats['tasks_dropped'] += 1
            if self.enable_logging:
                logger.debug(
                    f"[{self.engine_name}] Queue full, dropping task #{task.task_id} "
                    f"(frame {task.frame_count})"
                )
            return False

    def get_latest(self) -> Optional[InferenceResult]:
        """
        获取最新的推理结果（线程安全）

        Returns:
            InferenceResult or None: 最新结果，如果还没有任何结果则返回None
        """
        with self._result_lock:
            return self._latest_result

    def get_stats(self) -> Dict:
        """获取统计信息（线程安全）"""
        with self._stats_lock:
            return self._stats.copy()

    def _worker_loop(self):
        """后台工作线程的主循环"""
        logger.info(f"[{self.engine_name}] Worker loop started")

        # 为当前线程创建 CUDA context（PyCUDA 线程安全修复）
        cuda_ctx = None
        try:
            import pycuda.driver as cuda
            cuda.init()
            cuda_ctx = cuda.Device(0).retain_primary_context()
            cuda_ctx.push()
            logger.info(f"[{self.engine_name}] Worker thread CUDA context initialized")
        except Exception as e:
            logger.warning(f"[{self.engine_name}] Failed to initialize CUDA context: {e}")
            # 继续运行，让错误在推理时暴露

        with self._stats_lock:
            self._stats['worker_active'] = True

        try:
            while not self._stop_event.is_set():
                try:
                    # 阻塞获取任务，超时0.1秒以便检查stop_event
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 处理任务
                self._process_task(task)

        except Exception as e:
            logger.error(f"[{self.engine_name}] Worker loop crashed: {e}")
            logger.error(traceback.format_exc())
            self.last_exception = e
        finally:
            # 清理 CUDA context
            if cuda_ctx is not None:
                try:
                    cuda_ctx.pop()
                    logger.debug(f"[{self.engine_name}] Worker thread CUDA context cleaned up")
                except Exception as e:
                    logger.warning(f"[{self.engine_name}] Error cleaning up CUDA context: {e}")

            with self._stats_lock:
                self._stats['worker_active'] = False
            logger.info(f"[{self.engine_name}] Worker loop exited")

    def _process_task(self, task: InferenceTask):
        """
        处理单个推理任务

        Args:
            task: 推理任务
        """
        start_time = time.time()

        try:
            # 1. 判断是否需要推理
            should_run, reason = self.should_infer_func(task)

            if not should_run:
                if self.enable_logging:
                    logger.debug(
                        f"[{self.engine_name}] Skip task #{task.task_id}: {reason}"
                    )
                # 不需要推理，但可能需要返回缓存结果
                # 这里不更新 latest_result，让主线程继续使用之前的结果
                return

            # 2. 执行推理
            result = self.infer_func(task)

            inference_time = time.time() - start_time

            # 3. 包装结果
            wrapped_result = InferenceResult(
                task_id=task.task_id,
                result=result,
                inference_time=inference_time,
                fresh=True,
                source='async_inference'
            )

            # 4. 更新最新结果（线程安全）
            with self._result_lock:
                self._latest_result = wrapped_result

            # 5. 更新统计
            with self._stats_lock:
                self._stats['tasks_processed'] += 1
                self._stats['total_inference_time'] += inference_time
                self._stats['last_inference_time'] = inference_time

            if self.enable_logging:
                logger.debug(
                    f"[{self.engine_name}] Task #{task.task_id} completed "
                    f"in {inference_time*1000:.1f}ms"
                )

        except Exception as e:
            logger.error(f"[{self.engine_name}] Error processing task #{task.task_id}: {e}")
            logger.error(traceback.format_exc())

            # 记录错误
            with self._stats_lock:
                self._stats['errors'] += 1

            self.last_exception = e

            # 创建错误结果（保持接口一致性）
            error_result = InferenceResult(
                task_id=task.task_id,
                result=None,
                inference_time=time.time() - start_time,
                fresh=False,
                source='error'
            )

            with self._result_lock:
                self._latest_result = error_result
