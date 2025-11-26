#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI桌子主程序 - 集成的智能监测系统

集成功能：
- 眼距监测
- 情绪识别
- 疲劳检测
- TOF相机支持
"""

import sys
import os
import cv2
import warnings
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加modules路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, 'modules')
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

# 关闭警告
warnings.filterwarnings('ignore')

# 导入模块
from modules.core import Constants, logger
from modules.core.config_loader import get_config
from modules.camera import TOFCameraManager
from modules.eye_distance import EyeDistanceSystem
from preflight_check import preflight_check


def print_banner():
    """打印系统信息"""
    print("\n" + "="*70)
    print("         AI桌子智能监测系统 v2.0 - TensorRT-Only 架构")
    print("         集成功能：眼距监测 + 情绪识别 + 疲劳检测")
    print("="*70)
    print("\n TOF相机：自动读取SDK参数")
    print(" 检测模块：YOLO TensorRT 加速")
    print(" 情绪识别：EmoNet TensorRT")
    print(" 疲劳检测：TensorRT FaceMesh + EAR + PERCLOS")
    print("\n操作说明：")
    print("  q - 退出系统")
    print("  s - 保存截图")
    print("  r - 重置系统")
    print("  Space - 暂停/继续")
    print("  f - 全屏模式")
    print("  w - 窗口模式")
    print("  +/= - 放大窗口 (110%)")
    print("  - - 缩小窗口 (90%)")
    print("="*70 + "\n")


def run_orchestrator_mode(config):
    """运行 Orchestrator 异步推理模式"""
    from modules.core.orchestrator import DetectionOrchestrator
    from modules.engines import EmotionEngine, FatigueEngine, PoseEngine
    from modules.ui import WindowManager
    from modules.core.telemetry import TelemetryBuilder

    # 从配置获取SDK路径
    sdk_python_path = config.paths.get('sdk_python_path')
    sdk_lib_path = (
        config.paths.get('sdk_lib_path_aarch64') or
        config.paths.get('sdk_lib_path_linux64')
    )

    # 平面模型参数 - 从配置读取相机高度
    camera_height_m = config.desk_plane.get('camera_height_m', 0.5) if hasattr(config, 'desk_plane') else 0.5
    plane_model = (0.0, 1.0, 0.0, camera_height_m)
    logger.info(f"桌面平面配置: camera_height={camera_height_m}m, plane_model={plane_model}")

    # 从配置获取 TensorRT 引擎路径
    yolo_config = config.models.get('yolo_face', {})
    model_path = yolo_config.get('primary', 'models/yolov8n-face.engine')
    logger.info(f"Using TensorRT face detector: {model_path}")

    try:
        with TOFCameraManager(sdk_python_path=sdk_python_path,
                              sdk_library_path=sdk_lib_path) as camera:
            # 创建系统和检测器
            system = EyeDistanceSystem(
                camera=camera,
                plane_model=plane_model,
                model_path=model_path,
                depth_range=(200, 1500)
            )

            # 初始化 TRT 检测器
            from modules.emotion import EmoNetClassifier
            from modules.fatigue import FatigueDetector
            from modules.detection import TRTPoseDetector

            emotion_classifier = EmoNetClassifier()
            fatigue_detector = FatigueDetector(perclos_window=30, fps=30)

            # 姿态检测器
            pose_config = config.models.get('yolo_pose', {})
            pose_model_path = pose_config.get('primary', 'models/yolov8n-pose_fp16.engine')
            pose_detector = TRTPoseDetector(engine_path=pose_model_path)
            logger.info(f" 姿态检测模块初始化成功（TensorRT YOLO Pose）: {pose_model_path}")

            # 创建引擎
            engine_config = config.performance.async_inference.get('engine_config', {})
            emotion_engine = EmotionEngine(
                classifier=emotion_classifier,
                interval_frames=engine_config.get('emotion', {}).get('interval_frames', 10),
                cache_timeout=engine_config.get('emotion', {}).get('cache_timeout', 4.0),
                latency_smoothing=engine_config.get('emotion', {}).get('latency_smoothing', 0.7)
            )

            fatigue_engine = FatigueEngine(
                detector=fatigue_detector,
                latency_smoothing=engine_config.get('fatigue', {}).get('latency_smoothing', 0.7)
            )

            pose_engine = PoseEngine(
                detector=pose_detector,
                interval_frames=engine_config.get('pose', {}).get('interval_frames', 10),
                cache_timeout=engine_config.get('pose', {}).get('cache_timeout', 1.0),
                latency_smoothing=engine_config.get('pose', {}).get('latency_smoothing', 0.7)
            )

            # 创建 Orchestrator
            window_manager = WindowManager(
                window_name='AI Table Monitor - Orchestrator Mode',
                initial_size=(960, 720),
                system=system
            )

            orchestrator = DetectionOrchestrator(
                system=system,
                emotion_engine=emotion_engine,
                fatigue_engine=fatigue_engine,
                pose_engine=pose_engine,
                window_manager=window_manager,
                telemetry_builder=TelemetryBuilder(),
                system_config=config.__dict__ if hasattr(config, '__dict__') else {}
            )

            orchestrator.run()
            orchestrator.cleanup()

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"Orchestrator 模式错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def run_sync_mode(config):
    """
    运行同步推理模式

    Args:
        config: 系统配置对象
    """
    # 从配置获取SDK路径
    sdk_python_path = config.paths.get('sdk_python_path')
    sdk_lib_path = (
        config.paths.get('sdk_lib_path_aarch64') or
        config.paths.get('sdk_lib_path_linux64')
    )

    # 平面模型参数 - 从配置读取相机高度
    camera_height_m = config.desk_plane.get('camera_height_m', 0.5) if hasattr(config, 'desk_plane') else 0.5
    plane_model = (0.0, 1.0, 0.0, camera_height_m)
    logger.info(f"桌面平面配置: camera_height={camera_height_m}m, plane_model={plane_model}")

    # 从配置获取 TensorRT 引擎路径
    yolo_config = config.models.get('yolo_face', {})
    model_path = yolo_config.get('primary', 'models/yolov8n-face.engine')
    logger.info(f"Using TensorRT face detector: {model_path}")

    try:
        with TOFCameraManager(sdk_python_path=sdk_python_path,
                              sdk_library_path=sdk_lib_path) as camera:
            # 创建集成系统
            system = EyeDistanceSystem(
                camera=camera,
                plane_model=plane_model,
                model_path=model_path,
                depth_range=(200, 1500)
            )
            
            # 验证相机确实可用（双重保险）
            if not camera.camera_available:
                error_msg = (
                    "CRITICAL: TOF camera initialization succeeded but camera is not available!\n"
                    "This should not happen. Please check the system logs."
                )
                logger.error(error_msg)
                raise RuntimeError("TOF camera state inconsistency")

            # 初始化检测器
            # [临时禁用] 情绪识别 - 解决 CUDA 上下文问题后恢复
            emotion_classifier = None
            logger.info(" 情绪识别模块已禁用（临时调试）")
            # try:
            #     from modules.emotion import EmoNetClassifier
            #     emotion_classifier = EmoNetClassifier()
            #     logger.info(" 情绪识别模块初始化成功（TensorRT EmoNet）")
            # except Exception as e:
            #     logger.error(f" 情绪识别模块初始化失败: {e}")
            #     raise

            # [临时禁用] 疲劳检测 - 解决 CUDA 上下文问题后恢复
            fatigue_detector = None
            logger.info(" 疲劳检测模块已禁用（临时调试）")
            # try:
            #     from modules.fatigue import FatigueDetector
            #     fatigue_detector = FatigueDetector(perclos_window=30, fps=30)
            #     logger.info(" 疲劳检测模块初始化成功（TensorRT FaceMesh）")
            # except Exception as e:
            #     logger.error(f" 疲劳检测模块初始化失败: {e}")
            #     raise

            pose_detector = None
            try:
                from modules.detection import TRTPoseDetector
                pose_config = config.models.get('yolo_pose', {})
                pose_model_path = pose_config.get('primary', 'models/yolov8n-pose_fp16.engine')
                pose_detector = TRTPoseDetector(engine_path=pose_model_path)
                logger.info(f" 姿态检测模块初始化成功（TensorRT YOLO Pose）: {pose_model_path}")
            except Exception as e:
                logger.error(f" 姿态检测模块初始化失败: {e}")
                raise

            # 预热
            try:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                system.face_detector.detect_face(dummy)
                logger.info(" 检测器预热完成")
            except Exception as e:
                logger.warning(f" 检测器预热失败: {e}")

            # 运行主循环
            logger.info(" 系统启动完成！")
            run_main_loop(
                system,
                emotion_classifier,
                fatigue_detector,
                pose_detector,
                config=config
            )

    except KeyboardInterrupt:
        logger.info("\n 用户中断")
    except Exception as e:
        logger.error(f"\n 系统错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def run_main_loop(system, emotion_classifier=None, fatigue_detector=None, pose_detector=None,
                  config=None):
    """
    主显示循环

    Args:
        system: EyeDistanceSystem 实例
        emotion_classifier: 情绪分类器（可选）
        fatigue_detector: 疲劳检测器（可选）
        pose_detector: 姿态检测器（可选）
        config: 系统配置对象（可选）
    """
    paused = False
    visualization = None
    last_print_time = 0

    window_name = 'AI Table Monitor System - Real-time Measurement'
    cv2.destroyAllWindows()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 获取相机分辨率
    if system.camera.camera_available and system.camera.intrinsics_manager:
        camera_width = system.camera.intrinsics_manager.rgb_intrinsics['width']
        camera_height = system.camera.intrinsics_manager.rgb_intrinsics['height']
    else:
        camera_width = 640
        camera_height = 480
    if camera_width <= 0 or camera_height <= 0:
        logger.warning(
            f"Invalid camera resolution {camera_width}x{camera_height}, "
            "falling back to 640x480 for window sizing"
        )
        camera_width = 640
        camera_height = 480
    camera_aspect_ratio = camera_width / camera_height

    initial_width = 960
    initial_height = int(initial_width / camera_aspect_ratio)
    cv2.resizeWindow(window_name, initial_width, initial_height)

    window_width = initial_width
    window_height = initial_height

    try:
        while True:
            current_time = time.time()

            if not paused:
                if not system.camera.camera_available:
                    placeholder_frame = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
                    cv2.putText(placeholder_frame, "Camera Not Available", (camera_width//4, camera_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(window_name, placeholder_frame)
                else:
                    frame_data = system.camera.fetch_frame()
                    if frame_data is None:
                        continue

                    rgb_frame, depth_frame = system.image_processor.process_frame_data(frame_data)
                    if rgb_frame is None or depth_frame is None:
                        continue

                    # 单帧检测逻辑
                    results, visualization = system.process_frame(rgb_frame, depth_frame)

                    # 情绪识别
                    if emotion_classifier and results and results.get('detection'):
                        try:
                            bbox = results['detection']['bbox']
                            face_x1 = max(0, bbox['x1'])
                            face_y1 = max(0, bbox['y1'])
                            face_x2 = min(rgb_frame.shape[1], bbox['x2'])
                            face_y2 = min(rgb_frame.shape[0], bbox['y2'])

                            if face_x2 > face_x1 and face_y2 > face_y1:
                                face_img = rgb_frame[face_y1:face_y2, face_x1:face_x2]
                                emotion_results = emotion_classifier.predict_batch([face_img])
                                if emotion_results:
                                    results['emotion'] = emotion_results[0]
                        except Exception as e:
                            logger.error(f"情绪识别失败: {e}")

                    # 疲劳检测（使用人脸框裁剪 ROI 提高精度）
                    if fatigue_detector:
                        try:
                            if fatigue_detector.validate_frame(rgb_frame):
                                # 获取人脸框用于 ROI 裁剪（无人脸框时不执行疲劳检测）
                                fatigue_face_bbox = None
                                if results and results.get('detection') and results['detection'].get('bbox'):
                                    fatigue_face_bbox = results['detection']['bbox']

                                fatigue_result = fatigue_detector.detect_fatigue(rgb_frame, face_bbox=fatigue_face_bbox)
                                if fatigue_result is not None:
                                    if results is None:
                                        results = {}
                                    results['fatigue'] = fatigue_result
                        except Exception as e:
                            logger.error(f"疲劳检测失败: {e}")

                    # 姿态检测
                    if pose_detector:
                        try:
                            pose_results = pose_detector.detect_keypoints(rgb_frame)
                            if pose_results:
                                if results is None:
                                    results = {}
                                results['pose'] = pose_results
                        except Exception as e:
                            logger.error(f"姿态检测失败: {e}")

                    # 可视化
                    if results:
                        results['emotion_enabled'] = emotion_classifier is not None
                        results['fatigue_enabled'] = fatigue_detector is not None
                        results['pose_enabled'] = pose_detector is not None

                        if fatigue_detector and results.get('fatigue', {}).get('enabled', False):
                            visualization = system.visualizer.draw_combined_visualization(
                                rgb_frame.copy(), results, fatigue_detector
                            )
                        else:
                            visualization = system.visualizer.draw_visualization(
                                rgb_frame.copy(), results, "YOLO Face Model"
                            )

                    cv2.imshow(window_name, visualization if visualization is not None else rgb_frame)

                    # 控制台输出
                    if results and results.get('stable_distance') and current_time - last_print_time > 0.5:
                        distance_cm = results['stable_distance'] * 100
                        print(f"\r距离: {distance_cm:5.1f}cm | FPS: {results.get('fps', 0):.1f}", end='', flush=True)
                        last_print_time = current_time

            # 按键处理
            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('q'):
                break
            elif key == ord('s') and visualization is not None:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, visualization)
                logger.info(f"截图已保存: {filename}")
            elif key == ord('r'):
                system.reset()
            elif key == ord('c'):
                # 手动触发相机软重启
                logger.info("Manual camera soft restart triggered")
                camera.trigger_manual_restart()
            elif key == ord(' '):
                paused = not paused

    finally:
        cv2.destroyAllWindows()


def run_main_system():
    """系统主入口"""
    print_banner()

    # 运行预检
    logger.info("Running system preflight checks...")
    if not preflight_check():
        logger.error("Preflight check failed.")
        return False

    # 加载配置
    config = get_config(config_path=str(Path.cwd() / 'system_config.json'))

    # 读取异步推理配置开关
    async_enabled = config.performance.get('async_inference', {}).get('enabled', False)

    # 打印运行模式
    logger.info("=" * 60)
    logger.info("运行模式配置:")
    logger.info(f"  - 异步推理 (async_inference.enabled): {'启用' if async_enabled else '禁用'}")
    logger.info("=" * 60)

    # 根据开关选择运行模式
    if async_enabled:
        logger.info("【异步推理模式】Orchestrator + InferenceScheduler")
        return run_orchestrator_mode(config)
    else:
        logger.info("【同步推理模式】直接推理")
        return run_sync_mode(config)


def main():
    """主入口点"""
    try:
        success = run_main_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
