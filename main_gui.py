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
    print("         AI桌子智能监测系统 v3.1 - Pose+Tracker 架构")
    print("         集成功能：眼距监测 + 姿态估计 + 情绪识别 + 疲劳检测")
    print("="*70)
    print("\n TOF相机：自动读取SDK参数")
    print(" 检测模块：YOLO Pose TensorRT 加速")
    print(" 眼距计算：Pose关键点 + TOF深度")
    print(" 情绪识别：EmoNet (使用 tracker face_bbox)")
    print(" 疲劳检测：FaceMesh (使用 tracker face_bbox)")
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
    """运行 Orchestrator 异步推理模式（Pose+Tracker 路径）"""
    from modules.core.orchestrator import DetectionOrchestrator
    from modules.engines import EmotionEngine, FatigueEngine, PoseEngine
    from modules.core.telemetry import TelemetryBuilder
    from modules.core.runtime_switch import RuntimeSwitch

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

    try:
        with TOFCameraManager(sdk_python_path=sdk_python_path,
                              sdk_library_path=sdk_lib_path) as camera:
            # 创建系统（不再需要 YOLO Face 模型路径）
            system = EyeDistanceSystem(
                camera=camera,
                plane_model=plane_model,
                depth_range=(200, 1500)
            )

            # 初始化 TRT 检测器
            from modules.detection import TRTPoseDetector

            # ========== 情绪/疲劳模块初始化（根据 features 配置开关）==========
            features_config = config.__dict__.get('features', {}) if hasattr(config, '__dict__') else {}
            emotion_enabled = features_config.get('emotion', {}).get('enabled', False)
            fatigue_enabled = features_config.get('fatigue', {}).get('enabled', False)

            emotion_classifier = None
            fatigue_detector = None

            # 情绪识别模块
            if emotion_enabled:
                try:
                    from modules.emotion import EmoNetClassifier
                    emonet_config = config.models.get('emonet', {})
                    emonet_path = emonet_config.get('primary', 'models/emonet_fp16.engine')
                    emotion_classifier = EmoNetClassifier(engine_path=emonet_path)
                    logger.info(f"情绪识别模块初始化成功: {emonet_path}")
                except Exception as e:
                    logger.warning(f"情绪识别模块初始化失败: {e}")
                    emotion_classifier = None
            else:
                logger.info("情绪识别模块已禁用（features.emotion.enabled=false）")

            # 疲劳检测模块
            if fatigue_enabled:
                try:
                    from modules.fatigue import FatigueDetector
                    facemesh_config = config.models.get('facemesh', {})
                    facemesh_path = facemesh_config.get('primary', 'models/facemesh_fp16.engine')
                    fatigue_detector = FatigueDetector(model_path=facemesh_path)
                    logger.info(f"疲劳检测模块初始化成功: {facemesh_path}")
                except Exception as e:
                    logger.warning(f"疲劳检测模块初始化失败: {e}")
                    fatigue_detector = None
            else:
                logger.info("疲劳检测模块已禁用（features.fatigue.enabled=false）")

            # 姿态检测器
            pose_config = config.models.get('yolo_pose', {})
            pose_model_path = pose_config.get('primary', 'models/yolov8n-pose_fp16.engine')
            pose_conf = pose_config.get('confidence_threshold', 0.5)
            pose_iou = pose_config.get('iou_threshold', 0.45)
            pose_detector = TRTPoseDetector(
                engine_path=pose_model_path,
                confidence_threshold=pose_conf,
                iou_threshold=pose_iou
            )
            logger.info(f" 姿态检测模块初始化成功（TensorRT YOLO Pose）: {pose_model_path}")

            # Pose 检测器预热
            try:
                pose_detector.warmup(runs=3)
                logger.info(" Pose 检测器预热完成")
            except Exception as e:
                logger.warning(f" Pose 检测器预热失败: {e}")

            # 创建引擎
            engine_config = config.performance.async_inference.get('engine_config', {})

            # 情绪引擎（使用 tracker 的 face_bbox）
            emotion_engine = None
            if emotion_classifier is not None:
                emotion_engine = EmotionEngine(
                    classifier=emotion_classifier,
                    interval_frames=engine_config.get('emotion', {}).get('interval_frames', 10),
                    cache_timeout=engine_config.get('emotion', {}).get('cache_timeout', 4.0),
                    latency_smoothing=engine_config.get('emotion', {}).get('latency_smoothing', 0.7)
                )
                logger.info("EmotionEngine 初始化成功")

            # 疲劳引擎（使用 tracker 的 face_bbox）
            fatigue_engine = None
            if fatigue_detector is not None:
                fatigue_engine = FatigueEngine(
                    detector=fatigue_detector,
                    latency_smoothing=engine_config.get('fatigue', {}).get('latency_smoothing', 0.7)
                )
                logger.info("FatigueEngine 初始化成功")

            # PoseEngine 正确初始化：使用外部检测器避免重复加载模型
            from modules.pose import GetPose3dCoords, HeadOrientationFilter
            pose_3d_detector = GetPose3dCoords(
                external_detector=pose_detector  # 共享 TRTPoseDetector，不重复加载模型
            )
            head_orientation_filter = HeadOrientationFilter(slerp_alpha=0.8)
            pose_engine = PoseEngine(
                pose_detector=pose_3d_detector,
                head_orientation_filter=head_orientation_filter,
                system=system,
                interval_frames=engine_config.get('pose', {}).get('interval_frames', 10),
                cache_timeout=engine_config.get('pose', {}).get('cache_timeout', 1.0),
                latency_smoothing=engine_config.get('pose', {}).get('latency_smoothing', 0.7),
                system_config=config.__dict__ if hasattr(config, '__dict__') else {}
            )
            logger.info("PoseEngine 初始化成功（共享检测器）")

            # Orchestrator
            ui_enabled_env = os.getenv("AITABLE_ENABLE_UI", "0") == "1"
            from modules.ui import WindowManager
            window_manager = WindowManager(
                title='AI Table Monitor - Orchestrator Mode',
                width=960
            )
            if not ui_enabled_env:
                window_manager.disable()
                logger.info("UI 模式: 默认关闭 (AITABLE_ENABLE_UI=0)，可按 'v' 运行时开启。")
            else:
                logger.info("UI 模式: 已启用，窗口标题: AI Table Monitor - Orchestrator Mode。")

            # 可视化开关（允许按 'v' 运行时切换 UI 显示）
            visual_switch = RuntimeSwitch(initial=ui_enabled_env, name="UI显示")
            window_manager.bind_runtime_switch(visual_switch)

            orchestrator = DetectionOrchestrator(
                system=system,
                emotion_engine=emotion_engine,
                fatigue_engine=fatigue_engine,
                pose_engine=pose_engine,
                pose_detector=pose_detector,  # 启用 SimpleTracker 需要的 TRTPoseDetector
                window_manager=window_manager,
                telemetry_builder=TelemetryBuilder(),
                system_config=config.__dict__ if hasattr(config, '__dict__') else {},
                visual_switch=visual_switch
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
    [DEPRECATED] 运行同步推理模式 - 已重定向到 Orchestrator 模式

    Args:
        config: 系统配置对象
    """
    logger.warning("run_sync_mode() 已废弃，自动重定向到 run_orchestrator_mode()")
    return run_orchestrator_mode(config)


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

    # 强制使用 Orchestrator 模式（Pose+Tracker 路径）
    # face-only 同步模式已废弃
    logger.info("【Orchestrator 模式】Pose+Tracker 路径（face-only 已移除）")
    return run_orchestrator_mode(config)


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