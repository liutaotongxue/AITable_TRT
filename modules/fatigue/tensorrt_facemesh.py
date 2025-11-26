# -*- coding: utf-8 -*-
"""
TensorRT 加速的 FaceMesh 推理模块（仅 TensorRT）
仅支持 TensorRT 后端，不再降级到 ONNX Runtime

- 不使用 pycuda.autoinit，改为 retain_primary_context 共享主上下文（与 PyTorch 一致）
- 反序列化 + 内存分配在已 push 的主上下文中进行
- infer() 内每次/每线程 push/pop 主上下文，线程局部 stream，避免跨线程崩溃
- 详细日志：模型路径、TRT IO（每个 binding 的 shape/dtype/方向）
- 动态 shape：首次推理按真实输入 shape 分配，后续复用，size 变化则重分配
- 修复：不再访问 DeviceAllocation.size（该类型没有该属性），改为自维护 device_nbytes
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2
import os
import threading

from ..core.logger import logger

# ---------------------------
# 后端可用性
# ---------------------------
TENSORRT_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logger.info(f"[OK] TensorRT {trt.__version__} available in module")
except Exception as e:
    logger.error(f"TensorRT is required but not available: {e}")
    raise RuntimeError("TensorRT is required but not available") from e


# ---------------------------
# MediaPipe 兼容数据结构
# ---------------------------
@dataclass
class Landmark:
    x: float  # [0,1]
    y: float
    z: float = 0.0


class FaceLandmarks:
    def __init__(self, landmarks: List[Landmark]):
        self.landmark = landmarks


class FaceMeshResults:
    def __init__(self, multi_face_landmarks: Optional[List[FaceLandmarks]] = None):
        self.multi_face_landmarks = multi_face_landmarks


# ---------------------------
# TensorRT 原生推理封装（共享主上下文）
# ---------------------------
class TensorRTInferenceEngine:
    """
    原生 TensorRT 引擎封装（共享 CUDA 主上下文；线程安全）：
      - retain_primary_context() 与 PyTorch/其他模块共用同一上下文
      - __init__ 阶段在 push 后做反序列化与（静态）分配
      - infer() 每个调用/每线程 push/pop + 线程局部 cuda.Stream()
      - 动态 shape 首次推理时分配缓冲；shape 变化时重分配
      - 通过 device_nbytes 跟踪每个绑定的已分配显存大小（避免访问 DeviceAllocation.size）
    """

    def __init__(self, engine_path: str, default_nchw: Tuple[int, int] = (256, 256)):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed")

        self.engine_path = engine_path
        self._logger = trt.Logger(trt.Logger.WARNING)

        # 延迟导入 cuda（不使用 autoinit）
        import pycuda.driver as cuda
        self.cuda = cuda

        # 统一主上下文
        self.cuda.init()
        self._device = self.cuda.Device(0)
        self._primary_ctx = self._device.retain_primary_context()

        # TRT 对象
        self.runtime: Optional["trt.Runtime"] = None
        self.engine: Optional["trt.ICudaEngine"] = None
        self.context: Optional["trt.IExecutionContext"] = None

        # 绑定信息
        self.bindings: List[int] = []
        self.binding_is_input: List[bool] = []
        self.binding_dtypes: List[np.dtype] = []
        self.binding_shapes: List[Tuple[int, ...]] = []  # context 当前 shape
        self.input_indices: List[int] = []
        self.output_indices: List[int] = []

        # 缓冲
        self.host_mem: List[Optional[np.ndarray]] = []
        self.device_mem: List[Optional[object]] = []  # DeviceAllocation 指针
        self.device_nbytes: List[int] = []            # 每个 binding 已分配显存字节数

        # 线程局部 stream
        self._tls = threading.local()

        # 默认输入（动态时用于 layout 推断）
        self._default_nchw_wh = default_nchw  # (W,H)

        # 反序列化 & 初步分配
        self._push()
        try:
            self._deserialize()
            self._prepare_bindings()
            logger.info(f"TensorRT engine loaded: {engine_path}")
            self._log_io_summary()
        finally:
            self._pop()

    # ---- 上下文管理 ----
    def _push(self):
        self._primary_ctx.push()

    def _pop(self):
        try:
            self._primary_ctx.pop()
        except Exception:
            pass

    def _get_stream(self):
        if not hasattr(self._tls, "stream") or self._tls.stream is None:
            self._tls.stream = self.cuda.Stream()
        return self._tls.stream

    # ---- 初始化阶段 ----
    def _deserialize(self):
        self.runtime = trt.Runtime(self._logger)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

    def _prepare_bindings(self):
        num = self.engine.num_bindings
        self.bindings = [0] * num
        self.binding_is_input = [self.engine.binding_is_input(i) for i in range(num)]
        self.binding_dtypes = [trt.nptype(self.engine.get_binding_dtype(i)) for i in range(num)]

        # 默认把动态输入设置为 (1,3,H,W)（H,W 来自默认/约定值）
        for i in range(num):
            if self.binding_is_input[i]:
                shp = tuple(self.engine.get_binding_shape(i))
                if -1 in shp:
                    w, h = self._default_nchw_wh
                    self.context.set_binding_shape(i, (1, 3, h, w))

        # 记录实际 shape
        self.binding_shapes = [tuple(self.context.get_binding_shape(i)) for i in range(num)]
        self.host_mem = [None] * num
        self.device_mem = [None] * num
        self.device_nbytes = [0] * num

        # 静态分配（仅对没有 -1 的绑定）
        for i in range(num):
            shp = self.binding_shapes[i]
            if -1 in shp:
                continue
            size = int(np.prod(shp))
            host = self.cuda.pagelocked_empty(size, self.binding_dtypes[i])
            dev = self.cuda.mem_alloc(host.nbytes)
            self.host_mem[i] = host
            self.device_mem[i] = dev
            self.device_nbytes[i] = host.nbytes
            self.bindings[i] = int(dev)

        self.input_indices = [i for i in range(num) if self.binding_is_input[i]]
        self.output_indices = [i for i in range(num) if not self.binding_is_input[i]]

    def _log_io_summary(self):
        parts = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shp_decl = tuple(self.engine.get_binding_shape(i))
            shp_ctx = tuple(self.binding_shapes[i]) if self.binding_shapes else ()
            typ = self.binding_dtypes[i]
            role = "IN" if self.binding_is_input[i] else "OUT"
            parts.append(f"{role}[{i}] {name} shape={shp_decl} ctx_shape={shp_ctx} dtype={typ}")
        logger.info("TensorRT engine IO: " + " | ".join(parts))

    # ---- 内存确保 ----
    def _ensure_alloc(self, idx: int, shape: Tuple[int, ...]):
        size = int(np.prod(shape))
        dtype = self.binding_dtypes[idx]

        # host
        need_host = (
            self.host_mem[idx] is None
            or self.host_mem[idx].dtype != dtype
            or self.host_mem[idx].size < size
        )
        if need_host:
            self.host_mem[idx] = self.cuda.pagelocked_empty(size, dtype)

        required_nbytes = self.host_mem[idx].nbytes

        # device（不访问 DeviceAllocation.size）
        need_dev = (self.device_mem[idx] is None) or (self.device_nbytes[idx] < required_nbytes)
        if need_dev:
            if self.device_mem[idx] is not None:
                try:
                    self.device_mem[idx].free()
                except Exception:
                    pass
            self.device_mem[idx] = self.cuda.mem_alloc(required_nbytes)
            self.device_nbytes[idx] = required_nbytes
            self.bindings[idx] = int(self.device_mem[idx])

    # ---- 推理 ----
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        if not input_data.flags.c_contiguous:
            input_data = np.ascontiguousarray(input_data)

        self._push()
        try:
            stream = self._get_stream()

            if len(self.input_indices) != 1:
                raise RuntimeError(f"Expected 1 input, got {len(self.input_indices)}")
            in_idx = self.input_indices[0]

            if -1 in self.engine.get_binding_shape(in_idx):
                self.context.set_binding_shape(in_idx, tuple(input_data.shape))

            for i in range(self.engine.num_bindings):
                self.binding_shapes[i] = tuple(self.context.get_binding_shape(i))

            in_shape = self.binding_shapes[in_idx]
            if tuple(input_data.shape) != tuple(in_shape):
                raise ValueError(f"Expected input shape {in_shape}, got {input_data.shape}")
            in_dtype = self.binding_dtypes[in_idx]
            if input_data.dtype != in_dtype:
                input_data = input_data.astype(in_dtype, copy=False)

            self._ensure_alloc(in_idx, in_shape)

            # HtoD
            np.copyto(self.host_mem[in_idx], input_data.ravel())
            self.cuda.memcpy_htod_async(self.device_mem[in_idx], self.host_mem[in_idx], stream)

            # 输出缓冲
            for out_idx in self.output_indices:
                out_shape = self.binding_shapes[out_idx]
                self._ensure_alloc(out_idx, out_shape)

            ok = self.context.execute_async_v2(self.bindings, stream_handle=stream.handle)
            if not ok:
                stream.synchronize()
                raise RuntimeError("TensorRT execute_async_v2 failed")

            # DtoH
            for out_idx in self.output_indices:
                self.cuda.memcpy_dtoh_async(self.host_mem[out_idx], self.device_mem[out_idx], stream)
            stream.synchronize()

            # 组装输出（按 binding 顺序）
            outputs: List[np.ndarray] = []
            for out_idx in self.output_indices:
                out = self.host_mem[out_idx].reshape(self.binding_shapes[out_idx]).copy()
                outputs.append(out)

            return outputs
        finally:
            self._pop()

    def close(self):
        try:
            if hasattr(self._tls, "stream") and self._tls.stream is not None:
                self._push()
                try:
                    self._tls.stream.synchronize()
                finally:
                    self._pop()
        except Exception:
            pass

        for i in range(len(self.device_mem)):
            try:
                if self.device_mem[i] is not None:
                    self.device_mem[i].free()
            except Exception:
                pass
            self.device_mem[i] = None
            if i < len(self.device_nbytes):
                self.device_nbytes[i] = 0

        self.context = None
        self.engine = None
        self.runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass




# ---------------------------
# FaceMesh 主类（兼容 MediaPipe 接口）
# ---------------------------
class TensorRTFaceMesh:
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    LANDMARK_OUT_PREF = (1434, 1404)  # 478*3=1434 或 468*3=1404

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = None,  # (W,H)
        input_layout: str = None,            # "NCHW"/"NHWC"
        backend: str = "tensorrt"
    ):
        # 仅支持 TensorRT
        if not model_path.endswith((".engine", ".trt")):
            raise ValueError(
                f"Only TensorRT engine files (.engine/.trt) are supported. Got: {model_path}"
            )

        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is required but not available")

        self.model_path = model_path
        self.backend = "tensorrt"
        self.input_size = input_size
        self.input_layout = input_layout

        # 初始化 TensorRT 引擎
        try:
            self.engine = TensorRTInferenceEngine(model_path)
            logger.info(f"FaceMesh backend=TensorRT, model_path={self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT engine: {e}")
            raise RuntimeError(f"TensorRT initialization failed: {e}") from e

        # 自动探测输入尺寸与布局
        det_size, det_layout = self._detect_input_properties()
        if self.input_size is None:
            self.input_size = det_size
        if self.input_layout is None:
            self.input_layout = det_layout

        logger.info(f"Auto-detected: size={det_size}, layout={det_layout}")
        if (self.input_size != det_size) or (self.input_layout != det_layout):
            logger.info(f"Using custom: size={self.input_size}, layout={self.input_layout}")

        # 合法化
        if self.input_layout not in ("NCHW", "NHWC"):
            logger.warning(f"Invalid input_layout '{self.input_layout}', defaulting to NCHW")
            self.input_layout = "NCHW"

        logger.info(
            f"FaceMesh initialized with {self.backend} backend, "
            f"input size: {self.input_size}, layout: {self.input_layout}"
        )

    def _detect_input_properties(self) -> Tuple[Tuple[int, int], str]:
        """根据模型信息推断 (W,H) 和 layout。"""
        # TensorRT
        if isinstance(self.engine, TensorRTInferenceEngine):
            shp = self.engine.binding_shapes[self.engine.input_indices[0]]
            if len(shp) == 4:
                if shp[1] == 3:  # NCHW
                    return (shp[3], shp[2]), "NCHW"
                if shp[3] == 3:  # NHWC
                    return (shp[2], shp[1]), "NHWC"
            return (256, 256), "NCHW"

        logger.warning("Could not auto-detect input properties, using default (256x256, NCHW)")
        return (256, 256), "NCHW"

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """resize到模型输入、归一化到[0,1]，并按布局打包成 batch"""
        w, h = self.input_size
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0

        if self.input_layout == "NCHW":
            chw = np.transpose(normalized, (2, 0, 1))   # HWC->CHW
            batched = np.expand_dims(chw, axis=0)       # CHW->NCHW
        else:
            batched = np.expand_dims(normalized, axis=0)  # HWC->NHWC

        return np.ascontiguousarray(batched, dtype=np.float32)

    # -------- 新增：从多输出中挑出“关键点张量” --------
    def _pick_landmark_output(self, outs: List[np.ndarray]) -> np.ndarray:
        """
        选择 landmarks 输出：
        - 优先选最后一维为 1434 或 1404 的张量
        - 其次选扁平后 size 为 1434/1404 的
        - 再次选“能被3整除且 >=600”的最大张量
        - 可用环境变量 AITABLE_FACEMESH_OUT_INDEX 强制选择（0 基）
        """
        env_idx = os.getenv("AITABLE_FACEMESH_OUT_INDEX")
        if env_idx is not None:
            try:
                k = int(env_idx)
                if 0 <= k < len(outs):
                    return outs[k]
            except Exception:
                pass

        for pref in self.LANDMARK_OUT_PREF:
            for o in outs:
                if o.ndim >= 1 and o.shape[-1] == pref:
                    return o

        for pref in self.LANDMARK_OUT_PREF:
            for o in outs:
                if o.size == pref:
                    return o

        candidates = [o for o in outs if (o.size % 3 == 0 and o.size >= 600)]
        if candidates:
            candidates.sort(key=lambda x: x.size, reverse=True)
            return candidates[0]

        return sorted(outs, key=lambda x: x.size, reverse=True)[0]

    def _validate_output(self, output: np.ndarray) -> bool:
        if output is None or output.size == 0:
            logger.error("Model output is None or empty")
            return False
        if np.all(output == 0):
            logger.error("Model output is all zeros")
            return False
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            logger.error("Model output contains NaN/Inf")
            return False
        if np.max(np.abs(output)) > 1000:
            logger.error(f"Model output abnormal (max abs={np.max(np.abs(output))})")
            return False
        return True

    def postprocess(self, outputs: List[np.ndarray], original_size: Tuple[int, int]) -> List[Landmark]:
        """
        解析为 (x,y,z)；支持输出形状：
        [1,468,3] / [1,1,1,1434] / [1,1434] / [1,1404] / [1404] 等
        """
        output = self._pick_landmark_output(outputs)

        if not self._validate_output(output):
            raise ValueError("Model output validation failed")

        # 统一成 [N,3]
        if output.ndim == 4:
            landmarks_raw = output.reshape(-1, 3)
        elif output.ndim == 3:
            if output.shape[-1] == 3:
                landmarks_raw = output[0]
            else:
                landmarks_raw = output.reshape(-1, 3)
        elif output.ndim == 2:
            landmarks_raw = output.reshape(-1, 3)
        elif output.ndim == 1:
            landmarks_raw = output.reshape(-1, 3)
        else:
            raise ValueError(f"Unexpected landmark output shape: {output.shape}")

        if landmarks_raw.shape[-1] != 3:
            raise ValueError(f"Landmark last dim must be 3, got {landmarks_raw.shape}")

        landmarks: List[Landmark] = []
        for i in range(landmarks_raw.shape[0]):
            x, y, z = landmarks_raw[i]
            # 把可能的 [-1,1] 映到 [0,1]
            if x < 0 or x > 1 or y < 0 or y > 1:
                x = (float(x) + 1.0) / 2.0
                y = (float(y) + 1.0) / 2.0
            landmarks.append(Landmark(float(x), float(y), float(z)))
        return landmarks

    def process(self, image: np.ndarray) -> FaceMeshResults:
        """兼容 MediaPipe：输入 RGB(H,W,3)，输出 FaceMeshResults"""
        try:
            original_size = (image.shape[1], image.shape[0])
            inp = self.preprocess(image)

            outs = self.engine.infer(inp)  # TRT/ONNX 同名接口

            # # 调试：打印各输出形状，便于核对绑定顺序
            # try:
            #     logger.info(f"FaceMesh raw outputs: {[o.shape for o in outs]}")
            # except Exception:
            #     pass

            outs = [o.astype(np.float32, copy=False) if o.dtype != np.float32 else o for o in outs]
            landmarks = self.postprocess(outs, original_size)

            return FaceMeshResults(multi_face_landmarks=[FaceLandmarks(landmarks)])
        except Exception as e:
            logger.error(f"FaceMesh inference error: {e}")
            return FaceMeshResults(multi_face_landmarks=None)

    def get_eye_landmarks_indices(self) -> Tuple[List[int], List[int]]:
        return self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES


# ---------------------------
# 工厂函数
# ---------------------------
def create_facemesh(
    model_path: Optional[str] = None,
    input_size: Tuple[int, int] = None,
    input_layout: str = None,
    backend: str = "tensorrt"
) -> TensorRTFaceMesh:
    """
    仅使用 TensorRT 引擎。
    支持通过环境变量 AITABLE_FACEMESH_ENGINE 指定模型路径。
    """
    if model_path is None:
        env_path = os.getenv("AITABLE_FACEMESH_ENGINE")
        if env_path and os.path.exists(env_path):
            model_path = env_path
            logger.info(f"Using facemesh engine from env: {model_path}")

    if model_path is None:
        # 仅搜索 TensorRT 引擎
        tensorrt_candidates = [
            os.path.join("models", "facemesh_fp16.engine"),
            os.path.join("models", "facemesh_fp32.engine"),
            os.path.join("models", "face_landmark.engine"),
        ]

        for p in tensorrt_candidates:
            if os.path.exists(p):
                model_path = p
                logger.info(f"Using default TensorRT model: {model_path}")
                break

        if model_path is None:
            raise FileNotFoundError(
                "No TensorRT FaceMesh model found. Please place a .engine file under models/. "
                "You can also set env AITABLE_FACEMESH_ENGINE=/path/to/xxx.engine"
            )

    return TensorRTFaceMesh(
        model_path=model_path,
        input_size=input_size,
        input_layout=input_layout,
        backend=backend
    )
