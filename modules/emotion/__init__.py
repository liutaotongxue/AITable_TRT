"""
情绪识别模块（TensorRT-Only）

仅支持 TensorRT 版本，需要 TensorRT 引擎文件和 PyCUDA
如果 TensorRT 不可用，系统将拒绝启动
"""

from .trt_emonet_classifier import TRTEmoNetClassifier as EmoNetClassifier

__all__ = ['EmoNetClassifier', 'TRTEmoNetClassifier']
_backend = 'tensorrt'

# 向后兼容别名
TRTEmoNetClassifier = EmoNetClassifier
