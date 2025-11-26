"""
运行时开关模块
============

提供线程安全的布尔开关,支持订阅者模式,用于热切换系统功能(如UI显示)。
"""
import threading
from typing import Callable, List, Optional

try:
    from .logger import logger
except ImportError:
    # 独立运行或测试环境
    import logging
    logger = logging.getLogger(__name__)


class RuntimeSwitch:
    """
    线程安全的布尔开关,可订阅变化回调

    使用场景:
    - 运行时切换UI显示(无需重启进程)
    - 动态启用/禁用遥测输出
    - 热切换调试模式

    示例:
        >>> ui_switch = RuntimeSwitch(initial=True, name="UI显示")
        >>> ui_switch.subscribe(lambda on: print(f"UI {'开启' if on else '关闭'}"))
        >>> ui_switch.toggle()  # 触发回调
        UI 关闭
    """

    def __init__(self, initial: bool = True, name: str = "Switch"):
        """
        Args:
            initial: 初始状态
            name: 开关名称(用于日志)
        """
        self._flag = initial
        self._lock = threading.Lock()
        self._subs: List[Callable[[bool], None]] = []
        self._name = name
        logger.debug(f"RuntimeSwitch '{name}' 初始化: {initial}")

    def get(self) -> bool:
        """获取当前状态(线程安全)"""
        with self._lock:
            return self._flag

    def set(self, value: bool, source: str = "unknown"):
        """
        设置状态并触发订阅者回调

        Args:
            value: 新状态
            source: 触发来源(用于日志,如 "SIGUSR1", "keyboard", "API")
        """
        with self._lock:
            if value == self._flag:
                return  # 状态未变化,不触发回调
            old_value = self._flag
            self._flag = value
            subs = list(self._subs)  # 复制列表避免回调中修改订阅者

        # 释放锁后再执行回调(避免死锁)
        logger.info(
            f"RuntimeSwitch '{self._name}': {old_value} -> {value} (来源: {source})"
        )

        for fn in subs:
            try:
                fn(value)
            except Exception as e:
                logger.error(
                    f"RuntimeSwitch '{self._name}' 回调执行失败: {e}",
                    exc_info=True
                )

    def toggle(self, source: str = "toggle"):
        """切换状态"""
        self.set(not self.get(), source=source)

    def subscribe(self, fn: Callable[[bool], None]) -> Callable[[], None]:
        """
        订阅状态变化

        Args:
            fn: 回调函数,接收新状态(bool)

        Returns:
            取消订阅函数(调用后移除该回调)

        示例:
            >>> unsubscribe = switch.subscribe(my_callback)
            >>> # ... 稍后 ...
            >>> unsubscribe()  # 移除订阅
        """
        with self._lock:
            self._subs.append(fn)

        # 返回取消订阅函数
        def unsubscribe():
            with self._lock:
                try:
                    self._subs.remove(fn)
                except ValueError:
                    pass

        return unsubscribe

    def __repr__(self) -> str:
        return f"RuntimeSwitch(name='{self._name}', state={self.get()}, subscribers={len(self._subs)})"
