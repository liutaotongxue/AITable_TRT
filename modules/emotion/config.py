"""
情绪识别模块配置
"""
import os
import torch

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 情绪识别阈值
EMOTION_CONFIDENCE_THRESHOLD = 0.3  # 情绪识别结果的最低置信度，低于此值则认为是 "neutral"
EMOTION_SMOOTHING_WINDOW = 15  # 用于平滑情绪结果的帧数窗口大小

# 模型路径 - 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
EMOTION_MODEL_PATH = os.path.join(project_root, "models", "emonet_8.pth")

# 如果默认路径不存在，尝试其他路径
if not os.path.exists(EMOTION_MODEL_PATH):
    alternative_paths = [
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models", "emonet_8.pth"),
        "models/emonet_8.pth",
        r"C:\Users\TaoLiu\Desktop\table_final\models\emonet_8.pth"  # 保留原始路径作为备选
    ]
    for path in alternative_paths:
        if os.path.exists(path):
            EMOTION_MODEL_PATH = path
            break

# 情绪类别列表
EMOTION_LIST = [
    'neutral', 'happy', 'sad', 'surprise', 'fear', 
    'disgust', 'anger', 'contempt'
]