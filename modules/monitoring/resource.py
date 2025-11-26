"""
资源监控模块
===========

后台监控系统资源使用情况（RAM、GPU 显存）。
"""
import os
import time
import threading
from typing import Optional, Tuple
from ..core.logger import logger


class ResourceMonitor:
    """
    资源监控器

    后台线程定期记录 RAM/GPU 使用情况。

    使用示例:
        monitor = ResourceMonitor(interval=5.0)
        monitor.start()
        # ... 运行主程序 ...
        monitor.stop()

        # 或者直接读取内存（无需实例化）
        rss_mb, vms_mb = ResourceMonitor.read_process_memory(os.getpid())
    """

    @staticmethod
    def read_process_memory(pid: int) -> Tuple[Optional[float], Optional[float]]:
        """
        从 /proc 读取指定进程的内存信息（静态方法）

        Args:
            pid: 进程 ID

        Returns:
            (rss_mb, vms_mb): RSS 和 VMS 内存使用量（MB），读取失败返回 (None, None)
        """
        status_path = f"/proc/{pid}/status"
        rss_mb = vms_mb = 0.0
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        rss_mb = float(line.split()[1]) / 1024.0
                    elif line.startswith('VmSize:'):
                        vms_mb = float(line.split()[1]) / 1024.0
            return rss_mb, vms_mb
        except FileNotFoundError:
            return None, None

    def __init__(self, interval: float = 5.0):
        """
        初始化资源监控器

        Args:
            interval: 监控间隔（秒）
        """
        self.interval = max(1.0, float(interval))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cuda_available = False
        self._torch = None

        # 尝试导入 torch 检测 CUDA
        try:
            import torch
            self._torch = torch
            self._cuda_available = torch.cuda.is_available()
        except Exception:
            pass

    def start(self):
        """启动资源监控线程"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ResourceMonitor 已在运行")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._thread.start()
        logger.info(f"ResourceMonitor 已启动（间隔: {self.interval}s）")

    def stop(self):
        """停止资源监控线程"""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        logger.info("ResourceMonitor 已停止")

    def _monitor_loop(self):
        """资源监控主循环（运行在后台线程）"""
        pid = os.getpid()

        while not self._stop_event.is_set():
            # 检查进程是否存在
            if not os.path.exists(f"/proc/{pid}"):
                logger.info("进程已不存在，停止资源监控")
                break

            try:
                # 读取 RAM 使用情况
                rss_mb, vms_mb = self.read_process_memory(pid)
                rss_str = f"{rss_mb:.1f}MB" if rss_mb is not None else "N/A"
                vms_str = f"{vms_mb:.1f}MB" if vms_mb is not None else "N/A"

                # 读取 GPU 使用情况
                if self._cuda_available and self._torch is not None:
                    try:
                        alloc = self._torch.cuda.memory_allocated(0) / 1024**2
                        reserved = self._torch.cuda.memory_reserved(0) / 1024**2
                        total = self._torch.cuda.get_device_properties(0).total_memory / 1024**3
                        gpu_part = f"GPU alloc={alloc:.1f}MB reserved={reserved:.1f}MB (total {total:.1f}GB)"
                    except Exception as e:
                        gpu_part = f"GPU info unavailable ({e})"
                else:
                    gpu_part = "GPU unavailable"

                logger.info(f"[Resource] RAM RSS={rss_str} VMS={vms_str} | {gpu_part}")

            except Exception as e:
                logger.warning(f"[Resource] 监控线程异常: {e}")
                break

            # 等待下一次采样（可被 stop() 中断）
            self._stop_event.wait(self.interval)

    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False
