"""
帧率监控模块
===========

封装 FPS 计算和帧间隔统计逻辑。
"""
import time
from typing import Optional
from ..core.logger import logger


class FrameRateMonitor:
    """
    帧率监控器

    职责:
    - 计算全局 FPS（每帧调用 update）
    - 计算帧间隔（秒）
    - 使用 EMA 平滑帧率

    使用示例:
        fps_monitor = FrameRateMonitor(smoothing=0.9)

        while True:
            fps_monitor.update()
            current_fps = fps_monitor.get_fps()
            frame_interval = fps_monitor.get_interval()
            print(f"FPS: {current_fps:.2f}, Interval: {frame_interval*1000:.2f}ms")
    """

    def __init__(self, smoothing: float = 0.9):
        """
        初始化帧率监控器

        Args:
            smoothing: EMA 平滑系数（0.0-1.0），越大越平滑
        """
        self.smoothing = max(0.0, min(1.0, smoothing))  # 限制在 [0, 1]
        self.last_frame_time: Optional[float] = None
        self.fps: float = 0.0
        self.frame_interval: float = 0.0
        self.frame_count: int = 0

    def update(self, timestamp: Optional[float] = None):
        """
        更新帧率（每帧调用一次）

        Args:
            timestamp: 当前帧时间戳（秒），默认使用 time.monotonic()
        """
        current_time = timestamp if timestamp is not None else time.monotonic()

        if self.last_frame_time is not None:
            elapsed = current_time - self.last_frame_time

            if elapsed > 0:
                # 计算当前帧率
                current_fps = 1.0 / elapsed

                # EMA 平滑
                if self.frame_count == 0:
                    # 第一帧直接使用当前值
                    self.fps = current_fps
                else:
                    # 后续帧使用 EMA
                    self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps

                # 更新帧间隔
                self.frame_interval = elapsed

        self.last_frame_time = current_time
        self.frame_count += 1

    def get_fps(self) -> float:
        """
        获取当前 FPS

        Returns:
            当前平滑后的 FPS
        """
        return self.fps

    def get_interval(self) -> float:
        """
        获取帧间隔（秒）

        Returns:
            最后一帧的时间间隔（秒）
        """
        return self.frame_interval

    def get_interval_ms(self) -> float:
        """
        获取帧间隔（毫秒）

        Returns:
            最后一帧的时间间隔（毫秒）
        """
        return self.frame_interval * 1000.0

    def reset(self):
        """重置所有统计"""
        self.last_frame_time = None
        self.fps = 0.0
        self.frame_interval = 0.0
        self.frame_count = 0
        logger.info("FrameRateMonitor: 已重置")

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            包含统计信息的字典
        """
        return {
            'fps': self.fps,
            'interval_s': self.frame_interval,
            'interval_ms': self.frame_interval * 1000.0,
            'frame_count': self.frame_count
        }

    def __repr__(self) -> str:
        return f"FrameRateMonitor(fps={self.fps:.2f}, interval={self.frame_interval*1000:.2f}ms, frames={self.frame_count})"
