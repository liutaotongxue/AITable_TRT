"""
Camera Warmup Manager
====================

Responsibilities:
- Manage camera warmup process after startup
- Ensure RGB and Depth data streams decode properly
- Display recovery status (if camera is recovering)

Single Responsibility: Warmup process management

Migrated from: main_gui.py:204-225
"""
import time
from typing import Optional, Tuple, TYPE_CHECKING
from ..core.logger import logger

if TYPE_CHECKING:
    from .tof_manager import TOFCameraManager
    from .image_processor import ImageDataProcessor


class CameraWarmupManager:
    """
    Camera Warmup Manager

    Design purpose:
    1. Internalize warmup process (migrated from main_gui)
    2. Auto-detect if RGB/Depth data streams are normal
    3. Wait for camera recovery (if in recovery state)
    """

    def __init__(self,
                 camera_manager,  # DepthCameraInterface instance
                 image_processor: 'ImageDataProcessor',
                 max_attempts: int = 30,  # Reduced to 30 (match old version, avoid long wait)
                 poll_interval: float = 0.05):
        """
        Args:
            camera_manager: DepthCameraInterface instance
            image_processor: Image data processor (for decode testing)
            max_attempts: Maximum attempts (default 30, ~1.5s)
            poll_interval: Poll interval in seconds (default 50ms)
        """
        self.camera = camera_manager  # Use camera instead of camera_manager
        self.image_processor = image_processor
        self.max_attempts = max_attempts
        self.poll_interval = poll_interval

    def warmup(self) -> bool:
        """
        Execute warmup process

        Returns:
            bool: True if warmup succeeds, False on timeout

        Process:
            1. Poll for frame data
            2. Try to decode RGB and Depth
            3. Check camera recovery status
            4. Exit after successful decode
        """
        logger.info("Camera warmup in progress...")

        warm_ok = False

        for attempt in range(self.max_attempts):
            # Get latest frame
            frame_data = self.camera.get_latest_frame()

            if frame_data is None:
                # Display recovery status (every 30 attempts)
                if attempt % 30 == 0 and attempt > 0:
                    camera_status = self.camera.get_camera_status()
                    if camera_status not in ("ok", "initializing"):
                        logger.info(f"  Camera status: {camera_status}")

                time.sleep(self.poll_interval)
                continue

            # Try to decode
            rgb_warm, depth_warm = self._try_decode_frame(frame_data)

            if rgb_warm is not None and depth_warm is not None:
                warm_ok = True
                logger.info("Camera frame ready")
                break

            time.sleep(self.poll_interval)

        if not warm_ok:
            logger.warning(
                "Camera warmup incomplete, continuing (empty frames will be skipped)\n"
                f"   Attempts: {self.max_attempts}, Time: ~{self.max_attempts * self.poll_interval:.1f}s"
            )

        return warm_ok

    def _try_decode_frame(self, frame_data) -> Tuple[Optional[any], Optional[any]]:
        """
        Try to decode frame data

        Args:
            frame_data: MV3D_RGBD_FRAME_DATA

        Returns:
            (rgb_frame, depth_frame): Decoded RGB and depth data

        Notes:
            Returns (None, None) if decode fails
        """
        try:
            rgb_frame, depth_frame = self.image_processor.process_frame_data(frame_data)
            return rgb_frame, depth_frame
        except Exception as e:
            logger.debug(f"Frame decode failed during warmup: {e}")
            return None, None

    def wait_for_recovery(self, timeout: float = 30.0) -> bool:
        """
        Wait for camera recovery (optional interface)

        Args:
            timeout: Timeout in seconds

        Returns:
            bool: True if recovery succeeds, False on timeout

        Notes:
            If camera is in "soft_restarting", "reopening", "recovering" state,
            wait for it to return to "ok" state
        """
        logger.info("Waiting for camera recovery...")

        start_time = time.time()
        poll_interval = 0.1  # 100ms

        while (time.time() - start_time) < timeout:
            status = self.camera.get_camera_status()

            if status == "ok":
                logger.info("Camera recovered")
                return True

            if status == "error":
                logger.error("Camera recovery failed (entered ERROR state)")
                return False

            # Display recovery progress
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0:  # Every 5 seconds
                logger.info(f"  Camera status: {status} (waited {elapsed:.1f}s)")

            time.sleep(poll_interval)

        logger.warning(f"Camera recovery timeout ({timeout}s)")
        return False
