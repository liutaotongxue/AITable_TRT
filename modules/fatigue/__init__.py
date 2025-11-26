"""
疲劳检测模块（TensorRT-Only）

仅支持 TensorRT 版本，需要 TensorRT 引擎文件和 PyCUDA
如果 TensorRT 不可用，系统将拒绝启动
"""

from .trt_fatigue_detector import TRTFatigueDetector as FatigueDetector
from .tensorrt_facemesh import TensorRTFaceMesh, create_facemesh

__all__ = ['FatigueDetector', 'TensorRTFaceMesh', 'create_facemesh']
_backend = 'tensorrt'
