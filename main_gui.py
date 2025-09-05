
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
from modules.camera import TOFCameraManager
from modules.eye_distance import EyeDistanceSystem
from modules.emotion import EmoNetClassifier
from modules.fatigue import FatigueDetector


def print_banner():
    """打印系统信息"""
    print("\n" + "="*70)
    print("         AI桌子智能监测系统 v2.0 - 模块化版本")
    print("         集成功能：眼距监测 + 情绪识别 + 疲劳检测")
    print("="*70)
    print("\n TOF相机：自动读取SDK参数")
    print(" 检测模块：YOLO高精度检测")
    print(" 情绪识别：EmoNet深度学习")
    print(" 疲劳检测：MediaPipe + EAR")
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


def run_main_system():
    """运行主系统"""
    print_banner()
    
    # 平面模型参数
    plane_model = (0.0, 1.0, 0.0, 0.3)
    model_path = 'models/yolov8n-face.pt'
    
    # 检查模型文件
    if not os.path.exists(model_path):
        alternative_paths = [
            'yolov8n-face.pt',
            'yolov8n.pt'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            logger.warning("YOLO model not found, will use OpenCV as fallback")
    
    # 使用上下文管理器管理相机资源
    try:
        with TOFCameraManager() as camera_manager:
            # 检查相机是否可用
            if camera_manager.camera_available and camera_manager.intrinsics_manager:
                # 显示从sdK获取的参数信息
                intrinsics_manager = camera_manager.intrinsics_manager
                rgb_params = intrinsics_manager.get_rgb_intrinsics_dict()
                depth_params = intrinsics_manager.get_depth_intrinsics_dict()
                
                logger.info("\n 相机参数从sdK获取：")
                logger.info(f"RGB相机: fx={rgb_params['fx']:.2f}, fy={rgb_params['fy']:.2f}, "
                           f"cx={rgb_params['cx']:.2f}, cy={rgb_params['cy']:.2f}")
                logger.info(f"RGB分辨率: {intrinsics_manager.rgb_intrinsics['width']}x{intrinsics_manager.rgb_intrinsics['height']}")
                logger.info(f"深度相机: fx={depth_params['fx']:.2f}, fy={depth_params['fy']:.2f}, "
                           f"cx={depth_params['cx']:.2f}, cy={depth_params['cy']:.2f}")
                logger.info(f"深度分辨率: {intrinsics_manager.depth_intrinsics['width']}x{intrinsics_manager.depth_intrinsics['height']}")
            else:
                logger.warning("\n 相机硬件不可用，使用默认参数运行")
            
            # 创建集成系统
            logger.info("\n 初始化集成系统...")
            system = EyeDistanceSystem(
                camera_manager=camera_manager,
                plane_model=plane_model,
                model_path=model_path,
                depth_range=(200, 1500)
            )
            
            # 初始化情绪识别器
            emotion_classifier = None
            try:
                emotion_classifier = EmoNetClassifier()
                logger.info(" 情绪识别模块初始化成功")
            except Exception as e:
                logger.error(f" 情绪识别模块初始化失败: {e}")
            
            # 初始化疲劳检测器
            fatigue_detector = None
            try:
                fatigue_detector = FatigueDetector(perclos_window=30, fps=30)
                logger.info(" 疲劳检测模块初始化成功")
            except Exception as e:
                logger.error(f" 疲劳检测模块初始化失败: {e}")
            
            # 预热检测器
            logger.info(" 预热检测器...")
            try:
                import numpy as np
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                system.face_detector.detect_face(dummy)
                logger.info(" 检测器预热完成")
            except Exception as e:
                logger.warning(f" 检测器预热失败: {e}")
            
            # 运行主循环
            logger.info(" 系统启动完成！")
            run_main_loop(system, emotion_classifier, fatigue_detector)
            
    except KeyboardInterrupt:
        logger.info("\n 用户中断")
    except RuntimeError as e:
        logger.error(f"\n TOF相机错误: {e}")
        print(f"\n错误: {e}")
        print("请确保TOF相机正确连接")
        return False
    except Exception as e:
        logger.error(f"\n 系统错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_main_loop(system: EyeDistanceSystem, emotion_classifier=None, fatigue_detector=None):
    """主显示循环"""
    paused = False
    visualization = None
    last_print_time = 0
    
    # 创建可调整大小的窗口（只创建一次）
    # 使用英文窗口名称避免编码问题
    window_name = 'AI Table Monitor System - Real-time Measurement'
    
    # 先销毁所有可能存在的窗口
    cv2.destroyAllWindows()
    
    # 创建新窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取相机的实际分辨率
    if system.camera_manager.camera_available and system.camera_manager.intrinsics_manager:
        camera_width = system.camera_manager.intrinsics_manager.rgb_intrinsics['width']
        camera_height = system.camera_manager.intrinsics_manager.rgb_intrinsics['height']
    else:
        # 使用默认分辨率当相机不可用时
        camera_width = 640
        camera_height = 480
    camera_aspect_ratio = camera_width / camera_height
    
    # 设置初始窗口大小
    initial_width = 960  # 默认宽度
    initial_height = int(initial_width / camera_aspect_ratio)
    cv2.resizeWindow(window_name, initial_width, initial_height)
    
    # 用于跟踪窗口大小
    window_width = initial_width
    window_height = initial_height
    
    logger.info(f"窗口初始化: {window_width}x{window_height} (长宽比: {camera_aspect_ratio:.2f})")
    
    try:
        while True:
            current_time = time.time()
            
            if not paused:
                # 检查相机是否可用
                if not system.camera_manager.camera_available:
                    # 显示无相机提示
                    placeholder_frame = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
                    cv2.putText(placeholder_frame, "Camera Not Available", (camera_width//4, camera_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder_frame, "Running in simulation mode", (camera_width//4, camera_height//2 + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                    
                    # 显示帧
                    cv2.imshow(window_name, placeholder_frame)
                    
                else:
                    # 获取帧数据
                    frame_data = system.camera_manager.fetch_frame()
                    if frame_data is None:
                        continue
                    
                    # 处理图像数据
                    rgb_frame, depth_frame = system.image_processor.process_frame_data(frame_data)
                    
                    if rgb_frame is None or depth_frame is None:
                        logger.warning("图像数据不完整")
                        continue
                    
                    # 处理并显示
                    results, visualization = system.process_frame(rgb_frame, depth_frame)
                    
                    # 情绪识别处理
                    emotion_results = None
                    if emotion_classifier and results and results.get('detection'):
                        try:
                            bbox = results['detection']['bbox']
                            face_x1, face_y1 = max(0, bbox['x1']), max(0, bbox['y1'])
                            face_x2, face_y2 = min(rgb_frame.shape[1], bbox['x2']), min(rgb_frame.shape[0], bbox['y2'])
                            
                            if face_x2 > face_x1 and face_y2 > face_y1:
                                face_img = rgb_frame[face_y1:face_y2, face_x1:face_x2]
                                emotion_batch_results = emotion_classifier.predict_batch([face_img])
                                if emotion_batch_results:
                                    emotion_results = emotion_batch_results[0]
                        except Exception as e:
                            logger.error(f"情绪识别失败: {e}")
                    
                    # 疲劳检测处理
                    fatigue_results = None
                    if fatigue_detector:
                        try:
                            if fatigue_detector.validate_frame(rgb_frame):
                                fatigue_results = fatigue_detector.detect_fatigue(rgb_frame)
                        except Exception as e:
                            logger.error(f"疲劳检测失败: {e}")
                    
                    # 将额外的检测结果添加到results中
                    if results:
                        results['emotion'] = emotion_results
                        results['emotion_enabled'] = emotion_classifier is not None
                        results['fatigue'] = fatigue_results
                        results['fatigue_enabled'] = fatigue_detector is not None
                    
                    # 重新绘制可视化，包含所有检测结果
                    if results:  # 确保results不为None
                        if fatigue_detector and fatigue_results and fatigue_results.get('enabled', False):
                            # 使用增强版可视化
                            visualization = system.visualizer.draw_combined_visualization(
                                rgb_frame.copy(), 
                                results,
                                fatigue_detector
                            )
                        else:
                            # 使用标准可视化
                            visualization = system.visualizer.draw_visualization(
                                rgb_frame.copy(), 
                                results,
                                "YOLO Face Model"
                            )
                    else:
                        # 如果results为None，使用原始帧
                        visualization = rgb_frame.copy()
                    
                    # 显示结果
                    cv2.imshow(window_name, visualization)
                    
                    # 控制台输出（限制频率）
                    if results and results.get('stable_distance') and current_time - last_print_time > 0.5:
                        distance_cm = results['stable_distance'] * 100
                        stability = results['stability_score']
                        fps = results.get('fps', 0)
                        
                        # 构建状态信息
                        status_parts = [
                            f"距离: {distance_cm:5.1f}cm",
                            f"稳定: {stability:3.0%}",
                            f"FPS: {fps:.1f}",
                            "YOLO"
                        ]
                        
                        # 添加情绪信息
                        if results.get('emotion'):
                            emotion = results['emotion']['emotion']
                            status_parts.append(f"{emotion}")
                        
                        # 添加疲劳信息
                        if results.get('fatigue'):
                            fatigue_level = results['fatigue']['fatigue_level']
                            perclos = results['fatigue'].get('perclos', 0)
                            if results['fatigue'].get('perclos_valid', False):
                                status_parts.append(f"{fatigue_level} ({perclos:.1f}%)")
                            else:
                                status_parts.append(f"{fatigue_level}")
                        
                        print(f"\r{' | '.join(status_parts)} ", end='', flush=True)
                        last_print_time = current_time
            
            # 按键处理
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            
            # 检测窗口是否被关闭（点击X按钮）
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n\n窗口已关闭")
                break
                
            if key == ord('q'):
                print("\n\n系统关闭")
                break
            elif key == ord('s'):
                if visualization is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ai_table_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, visualization)
                    logger.info(f"截图已保存: {filename}")
            elif key == ord('r'):
                system.reset()
                logger.info(" 系统已重置")
            elif key == ord(' '):
                paused = not paused
                status = " 已暂停" if paused else " 已继续"
                logger.info(status)
            elif key == ord('f'):
                # 切换全屏模式
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                logger.info(" 全屏模式")
            elif key == ord('w'):
                # 切换为窗口模式
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                cv2.resizeWindow(window_name, window_width, window_height)
                logger.info(" 窗口模式")
            elif key == ord('=') or key == ord('+'):
                # 放大窗口 - 保持长宽比
                window_width = int(window_width * 1.1)
                window_height = int(window_width / camera_aspect_ratio)
                # 限制最大尺寸
                window_width = min(window_width, 1920)
                window_height = min(window_height, int(1920 / camera_aspect_ratio))
                cv2.resizeWindow(window_name, window_width, window_height)
                logger.info(f" 窗口放大: {window_width}x{window_height}")
            elif key == ord('-'):
                # 缩小窗口 - 保持长宽比
                window_width = int(window_width * 0.9)
                window_height = int(window_width / camera_aspect_ratio)
                # 限制最小尺寸
                window_width = max(window_width, 320)
                window_height = max(window_height, int(320 / camera_aspect_ratio))
                cv2.resizeWindow(window_name, window_width, window_height)
                logger.info(f" 窗口缩小: {window_width}x{window_height}")
    
    finally:
        cv2.destroyAllWindows()


def main():
    """主入口点"""
    try:
        success = run_main_system()
        if success:
            print("\n 程序正常结束")
        else:
            print("\n 程序异常结束")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()