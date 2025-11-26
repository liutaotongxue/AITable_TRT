"""
TensorRT 原生 API 封装基类
=============================

提供线程安全的 TensorRT 推理基础设施，替代 PyCUDA

特性:
- 使用 TensorRT 原生 Python API（无 PyCUDA 依赖）
- 线程安全的 execution context（每个线程独立）
- 自动 CUDA 内存管理（使用 cudart）
- 支持多线程并发推理

参考:
- TensorRT Python API: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/
- CUDA Runtime API: https://nvidia.github.io/cuda-python/
"""
import numpy as np
import tensorrt as trt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import ctypes

from .logger import logger


# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class CudaMemory:
    """
    CUDA 内存管理器（使用 CUDA Runtime API）

    替代 PyCUDA 的内存分配，使用 ctypes 调用 CUDA Runtime
    """

    def __init__(self, size: int):
        """
        分配 CUDA 设备内存

        Args:
            size: 字节数
        """
        self.size = size
        self.ptr = None

        # 加载 CUDA Runtime 库
        try:
            import cuda.cudart as cudart
            self.cudart = cudart

            # 分配设备内存
            err, self.ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA memory allocation failed: {cudart.cudaGetErrorString(err)}")

            logger.debug(f"Allocated {size} bytes of CUDA memory at {hex(self.ptr)}")

        except ImportError:
            # 降级到 ctypes 方式（适用于没有 cuda-python 的环境）
            logger.warning("cuda-python not found, falling back to ctypes cudart")
            self._fallback_init(size)

    def _fallback_init(self, size: int):
        """降级初始化：使用 ctypes 直接调用 libcudart"""
        try:
            # 尝试加载 CUDA Runtime 库
            try:
                cudart = ctypes.CDLL('libcudart.so')  # Linux
            except OSError:
                cudart = ctypes.CDLL('cudart64_110.dll')  # Windows (CUDA 11.x)

            # cudaMalloc 函数签名: cudaError_t cudaMalloc(void** devPtr, size_t size)
            cudaMalloc = cudart.cudaMalloc
            cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            cudaMalloc.restype = ctypes.c_int

            # 分配内存
            ptr = ctypes.c_void_p()
            err = cudaMalloc(ctypes.byref(ptr), size)

            if err != 0:  # cudaSuccess = 0
                raise RuntimeError(f"CUDA memory allocation failed with error code {err}")

            self.ptr = ptr.value
            self.cudart = cudart
            logger.debug(f"Allocated {size} bytes of CUDA memory at {hex(self.ptr)} (ctypes fallback)")

        except Exception as e:
            raise RuntimeError(f"Failed to allocate CUDA memory: {e}")

    def copy_from_host(self, host_data: np.ndarray):
        """从 host 拷贝数据到 device"""
        if self.ptr is None:
            raise RuntimeError("CUDA memory not allocated")

        try:
            # 使用 cuda-python
            err = self.cudart.cudaMemcpy(
                self.ptr,
                host_data.ctypes.data,
                self.size,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy H2D failed: {self.cudart.cudaGetErrorString(err)}")
        except AttributeError:
            # 降级到 ctypes
            cudaMemcpy = self.cudart.cudaMemcpy
            cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            cudaMemcpy.restype = ctypes.c_int

            err = cudaMemcpy(self.ptr, host_data.ctypes.data, self.size, 1)  # 1 = cudaMemcpyHostToDevice
            if err != 0:
                raise RuntimeError(f"cudaMemcpy H2D failed with error code {err}")

    def copy_to_host(self, host_data: np.ndarray):
        """从 device 拷贝数据到 host"""
        if self.ptr is None:
            raise RuntimeError("CUDA memory not allocated")

        try:
            # 使用 cuda-python
            err = self.cudart.cudaMemcpy(
                host_data.ctypes.data,
                self.ptr,
                self.size,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy D2H failed: {self.cudart.cudaGetErrorString(err)}")
        except AttributeError:
            # 降级到 ctypes
            cudaMemcpy = self.cudart.cudaMemcpy
            cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            cudaMemcpy.restype = ctypes.c_int

            err = cudaMemcpy(host_data.ctypes.data, self.ptr, self.size, 2)  # 2 = cudaMemcpyDeviceToHost
            if err != 0:
                raise RuntimeError(f"cudaMemcpy D2H failed with error code {err}")

    def free(self):
        """释放 CUDA 内存"""
        if self.ptr is not None:
            try:
                err = self.cudart.cudaFree(self.ptr)
                if hasattr(self.cudart, 'cudaError_t'):
                    # cuda-python
                    if err != self.cudart.cudaError_t.cudaSuccess:
                        logger.warning(f"cudaFree warning: {self.cudart.cudaGetErrorString(err)}")
                else:
                    # ctypes
                    if err != 0:
                        logger.warning(f"cudaFree warning: error code {err}")
            except Exception as e:
                logger.warning(f"Error freeing CUDA memory: {e}")
            finally:
                self.ptr = None

    def __del__(self):
        """析构时自动释放内存"""
        self.free()

    def __int__(self):
        """返回指针整数值（用于 TensorRT bindings）"""
        return self.ptr if self.ptr is not None else 0


class TRTEngineBase:
    """
    TensorRT 引擎基类

    提供线程安全的推理基础设施，适合多线程环境使用

    特性:
    - 自动加载 TensorRT 引擎文件
    - 为每个推理创建独立的 execution context（线程安全）
    - 自动管理 CUDA 内存（输入/输出 buffers）
    - 支持多 batch 推理

    使用示例:
    ```python
    class MyDetector(TRTEngineBase):
        def __init__(self, engine_path):
            super().__init__(engine_path)
            self._setup_bindings()

        def _setup_bindings(self):
            # 设置输入输出绑定
            self.input_shape = (1, 3, 224, 224)
            self.output_shape = (1, 1000)

        def infer(self, input_data):
            return self.execute({"input": input_data})
    ```
    """

    def __init__(self, engine_path: str):
        """
        初始化 TensorRT 引擎

        Args:
            engine_path: TensorRT 引擎文件路径 (.engine)

        Raises:
            FileNotFoundError: 引擎文件不存在
            RuntimeError: 引擎加载失败
        """
        engine_file = Path(engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.engine_path = str(engine_file)
        self.engine = None
        self.context = None

        # 加载引擎
        self._load_engine()

        # 内存缓冲区（每次推理时分配）
        self.buffers: Dict[str, CudaMemory] = {}

        logger.info(f"TensorRT engine loaded: {self.engine_path}")

    def _load_engine(self):
        """加载序列化的 TensorRT 引擎"""
        # 创建 runtime
        runtime = trt.Runtime(TRT_LOGGER)

        # 读取引擎文件
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        # 反序列化引擎
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")

        # 创建 execution context（线程局部，每次推理时创建）
        # 注意：这里只是初始化，实际推理时会在线程内创建独立的 context
        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        logger.debug(f"TensorRT engine loaded successfully: {self.engine.name}")

    def get_binding_shape(self, binding_name: str) -> Tuple[int, ...]:
        """获取绑定的形状"""
        for i in range(self.engine.num_bindings):
            if self.engine.get_binding_name(i) == binding_name:
                return tuple(self.engine.get_binding_shape(i))
        raise ValueError(f"Binding '{binding_name}' not found in engine")

    def get_binding_dtype(self, binding_name: str) -> np.dtype:
        """获取绑定的数据类型"""
        for i in range(self.engine.num_bindings):
            if self.engine.get_binding_name(i) == binding_name:
                trt_dtype = self.engine.get_binding_dtype(i)
                # TensorRT dtype 到 numpy dtype 的映射
                dtype_map = {
                    trt.DataType.FLOAT: np.float32,
                    trt.DataType.HALF: np.float16,
                    trt.DataType.INT8: np.int8,
                    trt.DataType.INT32: np.int32,
                    trt.DataType.BOOL: np.bool_,
                }
                return dtype_map.get(trt_dtype, np.float32)
        raise ValueError(f"Binding '{binding_name}' not found in engine")

    def allocate_buffers(self, bindings: Dict[str, Tuple[int, ...]]):
        """
        分配 CUDA 内存缓冲区

        Args:
            bindings: 绑定名称 -> shape 的字典
        """
        self.buffers = {}

        for name, shape in bindings.items():
            dtype = self.get_binding_dtype(name)
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.buffers[name] = CudaMemory(size)
            logger.debug(f"Allocated buffer for '{name}': shape={shape}, size={size} bytes")

    def execute(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        执行推理

        Args:
            inputs: 输入数据字典 {name: numpy_array}
            output_names: 输出名称列表

        Returns:
            输出数据字典 {name: numpy_array}
        """
        # 创建线程局部的 execution context（线程安全）
        context = self.engine.create_execution_context()

        try:
            # 准备绑定指针数组
            bindings = [0] * self.engine.num_bindings

            # 拷贝输入数据到 device
            for name, host_data in inputs.items():
                # 确保数据是连续的
                host_data = np.ascontiguousarray(host_data)

                # 获取绑定索引
                binding_idx = self.engine.get_binding_index(name)
                if binding_idx == -1:
                    raise ValueError(f"Input binding '{name}' not found in engine")

                # 分配或重用缓冲区
                if name not in self.buffers:
                    size = host_data.nbytes
                    self.buffers[name] = CudaMemory(size)

                # 拷贝数据
                self.buffers[name].copy_from_host(host_data)
                bindings[binding_idx] = int(self.buffers[name])

            # 为输出分配缓冲区
            outputs = {}
            for name in output_names:
                binding_idx = self.engine.get_binding_index(name)
                if binding_idx == -1:
                    raise ValueError(f"Output binding '{name}' not found in engine")

                shape = self.get_binding_shape(name)
                dtype = self.get_binding_dtype(name)

                if name not in self.buffers:
                    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    self.buffers[name] = CudaMemory(size)

                bindings[binding_idx] = int(self.buffers[name])
                outputs[name] = np.zeros(shape, dtype=dtype)

            # 执行推理
            success = context.execute_v2(bindings)

            if not success:
                raise RuntimeError("TensorRT inference execution failed")

            # 拷贝输出数据到 host
            for name, host_data in outputs.items():
                self.buffers[name].copy_to_host(host_data)

            return outputs

        finally:
            # 清理线程局部的 context
            del context

    def close(self):
        """释放所有资源"""
        # 释放 CUDA 内存
        for buffer in self.buffers.values():
            buffer.free()
        self.buffers.clear()

        # 清理 TensorRT 对象
        if self.context:
            del self.context
            self.context = None

        if self.engine:
            del self.engine
            self.engine = None

        logger.debug(f"TensorRT engine closed: {self.engine_path}")

    def __del__(self):
        """析构时自动清理"""
        self.close()
