"""
检测引擎模块
===========

封装各类检测推理引擎（情绪、疲劳、姿态）。
"""
from .emotion_engine import EmotionEngine, EmotionResult
from .fatigue_engine import FatigueEngine, FatigueResult
from .pose_engine import PoseEngine, PoseResult

__all__ = [
    'EmotionEngine',
    'EmotionResult',
    'FatigueEngine',
    'FatigueResult',
    'PoseEngine',
    'PoseResult'
]
