"""
窗口管理模块
===========

封装 OpenCV 窗口创建、显示、键盘处理和健康检查逻辑。
"""
import os
import cv2
import time
from datetime import datetime
from typing import Optional, Tuple
from ..compat import np
from ..core.logger import logger


class WindowManager:
    """
    OpenCV 窗口管理器

    职责:
    - 窗口创建和重建
    - 安全的图像显示（异常处理）
    - 键盘事件处理
    - 窗口健康检查（可见性、去抖）

    使用示例:
        ui = WindowManager(
            title="AI Table Monitor",
            width=960,
            aspect_ratio=1.25,
            enabled=True,
            stay_open=True
        )
        ui.create()

        while True:
            ui.show(frame, overlay_text="FPS: 30")
            status = ui.poll_keys_and_health()
            if status == "quit":
                break

        ui.destroy()
    """

    def __init__(
        self,
        title: str = "AI Table Monitor System - Real-time Measurement",
        width: int = 960,
        aspect_ratio: float = 1.25,
        enabled: bool = True,
        stay_open: bool = True,
        no_autoclose: bool = True,
    ):
        """
        初始化窗口管理器

        Args:
            title: 窗口标题
            width: 窗口初始宽度
            aspect_ratio: 窗口宽高比
            enabled: 是否启用 UI（False 为无头模式）
            stay_open: 窗口关闭时是否自动重建
            no_autoclose: 是否禁用自动关闭（窗口不可见时）
        """
        self.title = title
        self.width = width
        self.aspect_ratio = aspect_ratio
        self.height = int(width / aspect_ratio)
        self.enabled = enabled
        self.stay_open = stay_open
        self.no_autoclose = no_autoclose

        # 内部状态
        self.paused = False
        self._close_state = {"invisible": 0, "imshow_err": 0}

        # 帧状态计数（用于退出逻辑）
        self.empty_frame_streak = 0      # 连续空帧计数
        self.invalid_frame_streak = 0    # 连续无效帧计数
        self.no_result_streak = 0        # 连续无结果计数

        # 去抖阈值
        self.CLOSE_INVISIBLE_THRESH = 45  # 连续 45 帧窗口不可见
        self.IMSHOW_ERR_THRESH = 10       # 连续 10 次 imshow 异常
        self.EMPTY_FRAME_EXIT_THRESH = 120  # 连续 120 帧空帧后退出

        # 可选的 RuntimeSwitch 引用（用于按键 'v' 热切换）
        self._visual_switch = None

        if self.enabled:
            logger.info(f"UI 模式: 已启用（窗口标题: {self.title}）")
        else:
            logger.info("UI 模式: 已禁用（无头运行）")

    def create(self):
        """创建（或重建）窗口"""
        if not self.enabled:
            return

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)
        logger.info(f"[UI] 窗口已创建/重建: {self.width}x{self.height}")

    def show(self, image: Optional[np.ndarray], overlay_text: Optional[str] = None):
        """
        显示图像（带异常处理）

        Args:
            image: 要显示的图像（BGR 格式），None 时显示黑屏
            overlay_text: 可选的覆盖文本（左上角）
        """
        if not self.enabled:
            return

        # 创建画布
        canvas = image if image is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 添加覆盖文本
        if overlay_text:
            cv2.putText(
                canvas,
                overlay_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # 安全显示
        try:
            cv2.imshow(self.title, canvas)
            self._close_state["imshow_err"] = 0
        except cv2.error as e:
            logger.warning(f"imshow 异常（忽略本帧）：{e}")
            self._close_state["imshow_err"] += 1

    def poll_keys_and_health(self, on_reset_callback=None, on_screenshot_callback=None) -> str:
        """
        轮询键盘输入和窗口健康状态

        Args:
            on_reset_callback: 按下 'r' 键时的回调函数（用于重置系统）
            on_screenshot_callback: 按下 's' 键时的回调函数（用于保存截图）

        Returns:
            str: "quit" - 应退出主循环
                 "continue" - 继续运行
                 "paused" - 已暂停（仅状态信息）
        """
        if not self.enabled:
            return "continue"

        # 键盘处理
        key = cv2.waitKey(1 if not self.paused else 30) & 0xFF

        if key in (ord('q'), 27):  # q 或 ESC
            logger.info("退出原因: 用户按键 (q/ESC)")
            return "quit"

        elif key == ord('s'):  # 截图
            if on_screenshot_callback:
                on_screenshot_callback()
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"ai_table_screenshot_{ts}.jpg"
                logger.info(f"截图请求已触发（文件名: {fn}），但未提供截图回调")

        elif key == ord('r'):  # 重置系统
            if on_reset_callback:
                on_reset_callback()
            logger.info("系统已重置")

        elif key == ord(' '):  # 暂停/继续
            self.paused = not self.paused
            logger.info("已暂停" if self.paused else "已继续")

        elif key == ord('f'):  # 全屏
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            logger.info("全屏模式")

        elif key == ord('w'):  # 窗口模式
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)
            logger.info("窗口模式")

        elif key == ord('=') or key == ord('+'):  # 放大
            self.width = min(int(self.width * 1.1), 1920)
            self.height = int(self.width / self.aspect_ratio)
            cv2.resizeWindow(self.title, self.width, self.height)
            logger.info(f"窗口放大: {self.width}x{self.height}")

        elif key == ord('-'):  # 缩小
            self.width = max(int(self.width * 0.9), 320)
            self.height = max(int(self.width / self.aspect_ratio), int(320 / self.aspect_ratio))
            cv2.resizeWindow(self.title, self.width, self.height)
            logger.info(f"窗口缩小: {self.width}x{self.height}")

        elif key == ord('v'):  # 切换可视化（热切换）
            # 如果外部传入了 RuntimeSwitch，触发切换
            if hasattr(self, '_visual_switch') and self._visual_switch:
                self._visual_switch.toggle(source="keyboard_v")
            else:
                logger.info("按键 'v' 触发可视化切换，但未绑定 RuntimeSwitch")

        # 窗口健康检查
        prop_val = self._check_window_health()

        # 窗口被销毁：STAY_OPEN 模式下重建，否则退出
        if prop_val == -1:
            if self.stay_open:
                logger.warning("[UI] 检测到窗口被外部销毁，正在自动重建窗口（STAY_OPEN=1）")
                self.create()
                return "continue"
            else:
                logger.info("退出原因: 窗口被用户关闭")
                return "quit"

        # 窗口不可见去抖
        if not self.no_autoclose:
            if prop_val < 1:
                self._close_state["invisible"] += 1
            else:
                self._close_state["invisible"] = 0

            # 连续多帧不可见
            if self._close_state["invisible"] > self.CLOSE_INVISIBLE_THRESH:
                if self.stay_open:
                    logger.warning("[UI] 连续多帧窗口不可见，但 STAY_OPEN=1，保持运行")
                    self._close_state["invisible"] = 0
                else:
                    logger.info("退出原因: 连续多帧检测到窗口不可见")
                    return "quit"

        # imshow 异常去抖
        if self._close_state["imshow_err"] > self.IMSHOW_ERR_THRESH:
            if self.stay_open:
                logger.warning(f"[UI] 连续 {self.IMSHOW_ERR_THRESH} 次 imshow 异常，尝试重建窗口")
                self.create()
                self._close_state["imshow_err"] = 0
            else:
                logger.info("退出原因: 连续多次 imshow 异常")
                return "quit"

        # 检查其他退出条件（空帧超时等）
        should_exit, exit_reason = self.check_exit_conditions()
        if should_exit:
            logger.info(f"退出原因: {exit_reason}")
            return "quit"

        return "continue"

    def _check_window_health(self) -> float:
        """
        检查窗口健康状态

        Returns:
            float: 窗口可见性属性值
                   1.0: 可见
                   0.0: 不可见
                  -1.0: 窗口被销毁
        """
        try:
            prop_val = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            # Jetson 上偶发异常，按可见处理
            prop_val = 1.0

        # Jetson 平台兼容性：getWindowProperty 在正常运行时也可能返回 -1
        # 这是 Jetson OpenCV 的已知行为，不代表窗口被销毁，应视为窗口可见
        if prop_val < 0:
            prop_val = 1.0

        return prop_val

    def enable(self):
        """热启用窗口（运行时开启可视化）"""
        if not self.enabled:
            self.enabled = True
            self.create()
            logger.info("[UI] 窗口已热启用")

    def disable(self):
        """热禁用窗口（运行时关闭可视化，不影响推理）"""
        if self.enabled:
            self.enabled = False
            self.destroy()
            logger.info("[UI] 窗口已热禁用（推理继续运行）")

    def destroy(self):
        """销毁窗口"""
        if not self.enabled:
            return

        try:
            cv2.destroyAllWindows()
            logger.info("[UI] 窗口已销毁")
        except Exception as e:
            logger.warning(f"[UI] 销毁窗口异常: {e}")

    def save_screenshot(self, image: np.ndarray) -> Optional[str]:
        """
        保存当前帧为截图

        Args:
            image: 要保存的图像

        Returns:
            str: 截图文件路径，失败返回 None
        """
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"ai_table_screenshot_{ts}.jpg"
            cv2.imwrite(fn, image)
            logger.info(f"截图已保存: {fn}")
            return fn
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
            return None

    def is_paused(self) -> bool:
        """返回当前暂停状态"""
        return self.paused

    def on_empty_frame(self):
        """记录空帧（连续空帧可能触发退出）"""
        self.empty_frame_streak += 1

    def on_valid_frame(self):
        """记录有效帧（重置空帧计数）"""
        self.empty_frame_streak = 0

    def on_invalid_frame(self):
        """记录无效帧"""
        self.invalid_frame_streak += 1

    def on_no_result(self):
        """记录无结果帧"""
        self.no_result_streak += 1

    def reset_frame_counters(self):
        """重置所有帧计数器"""
        self.empty_frame_streak = 0
        self.invalid_frame_streak = 0
        self.no_result_streak = 0
        logger.info("[UI] 帧计数器已重置")

    def get_frame_stats(self) -> dict:
        """
        获取帧状态统计

        Returns:
            包含计数器的字典
        """
        return {
            'empty_frame_streak': self.empty_frame_streak,
            'invalid_frame_streak': self.invalid_frame_streak,
            'no_result_streak': self.no_result_streak,
        }

    def check_exit_conditions(self) -> tuple:
        """
        检查退出条件（空帧超时等）

        Returns:
            (should_exit: bool, exit_reason: str or None)
        """
        # 检查空帧超时
        if self.empty_frame_streak >= self.EMPTY_FRAME_EXIT_THRESH:
            return (True, f"empty_frame_timeout ({self.empty_frame_streak} frames)")

        return (False, None)

    def handle_frame_event(
        self,
        event_type: str,
        visualization: Optional[np.ndarray] = None,
        overlay_text: Optional[str] = None,
        on_reset_callback=None,
        on_screenshot_callback=None
    ) -> str:
        """
        处理帧事件（高级接口，封装状态更新+显示+键盘处理）

        Args:
            event_type: 事件类型 ('empty', 'invalid', 'valid', 'no_result')
            visualization: 要显示的图像
            overlay_text: 覆盖文本
            on_reset_callback: 重置回调
            on_screenshot_callback: 截图回调

        Returns:
            str: "quit" - 应退出主循环
                 "continue" - 继续运行
        """
        if not self.enabled:
            return "continue"

        # 更新帧状态
        if event_type == 'empty':
            self.on_empty_frame()
        elif event_type == 'invalid':
            self.on_invalid_frame()
        elif event_type == 'valid':
            self.on_valid_frame()
        elif event_type == 'no_result':
            self.on_no_result()

        # 显示可视化
        if visualization is not None:
            self.show(visualization, overlay_text=overlay_text)

        # 轮询键盘和健康状态
        return self.poll_keys_and_health(
            on_reset_callback=on_reset_callback,
            on_screenshot_callback=on_screenshot_callback
        )

    def get_frame_streak(self, event_type: str) -> int:
        """
        获取指定类型的帧连续计数

        Args:
            event_type: 事件类型 ('empty', 'invalid', 'no_result')

        Returns:
            int: 连续计数
        """
        if event_type == 'empty':
            return self.empty_frame_streak
        elif event_type == 'invalid':
            return self.invalid_frame_streak
        elif event_type == 'no_result':
            return self.no_result_streak
        return 0

    def __enter__(self):
        """支持 with 语句"""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.destroy()
        return False

    def bind_runtime_switch(self, visual_switch):
        """
        绑定 RuntimeSwitch，允许按键 'v' 热切换可视化

        Args:
            visual_switch: RuntimeSwitch 实例
        """
        self._visual_switch = visual_switch
        logger.debug("WindowManager 已绑定 RuntimeSwitch（按 'v' 键可切换可视化）")
