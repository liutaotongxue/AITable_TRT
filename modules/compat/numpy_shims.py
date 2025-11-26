"""
NumPy 兼容性修补模块
====================

为旧版代码和第三方库（TensorRT等）提供 NumPy 类型别名兼容性。
必须在导入任何使用 NumPy 的模块之前导入此模块。

使用方式：
    from modules.compat.numpy_shims import np

注意：
    此模块必须在 main 程序最开始导入，以确保所有后续导入都能正确使用别名。
"""
import numpy as np
import warnings

# 禁用 NumPy 相关的 Future/Deprecation 警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 检测 NumPy 版本
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
NUMPY_2_0_OR_LATER = NUMPY_VERSION >= (2, 0)

# 为旧代码添加兼容属性（TensorRT/第三方库可能用到）
# NumPy 2.0+ 移除了 np.float_、np.int_ 等别名，这里映射到 Python 内建类型
if NUMPY_2_0_OR_LATER:
    # NumPy 2.0+: 映射到 Python 内建类型
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'str'):
        np.str = str
    if not hasattr(np, 'long'):
        np.long = int  # Python 3 没有 long
else:
    # NumPy 1.x: 映射到 NumPy 的旧别名（如果存在）
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = np.int_
    if not hasattr(np, 'float'):
        np.float = np.float_
    if not hasattr(np, 'complex'):
        np.complex = np.complex_
    if not hasattr(np, 'object'):
        np.object = np.object_
    if not hasattr(np, 'str'):
        np.str = np.str_
    if not hasattr(np, 'long'):
        np.long = np.int_  # Python 3 没有 long，映射到 int


# 导出 numpy 以供其他模块使用
__all__ = ['np']
