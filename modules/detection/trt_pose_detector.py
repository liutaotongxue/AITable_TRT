"""
TensorRT YOLO Pose 推理封装
独立于 Ultralytics，直接使用 TensorRT API 进行推理
包含完整的 YOLO Pose 后处理（NMS + 关键点解析）

主要改进（支持多种引擎格式）:
- 自动检测引擎输出 binding shape（通道优先 vs anchor 优先）
- 统一的 reshape/transpose 逻辑，兼容 (1, 56, 8400)、(1, 8400, 56)、动态 batch 等
- 运行时动态查询 binding shape，支持 profile 切换（不同输入尺寸/batch）
- 运行时校验确保关键点坐标正确映射
- 详细的调试日志（通过环境变量 AITABLE_DEBUG_POSE=1 启用）
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # 注意：不使用 pycuda.autoinit，手动管理上下文以避免与 PyTorch 冲突
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ..core.logger import logger


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    转换边界框格式 (center_x, center_y, width, height) -> (x1, y1, x2, y2)

    Args:
        boxes: (N, 4) 数组，格式 [cx, cy, w, h]

    Returns:
        (N, 4) 数组，格式 [x1, y1, x2, y2]
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """
    转换边界框格式 (x1, y1, x2, y2) -> (x, y, width, height)

    Args:
        boxes: (N, 4) 数组，格式 [x1, y1, x2, y2]

    Returns:
        (N, 4) 数组，格式 [x, y, w, h]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1

    return np.stack([x, y, w, h], axis=1)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    非极大值抑制（NMS）

    Args:
        boxes: (N, 4) 边界框，格式 [x1, y1, x2, y2]
        scores: (N,) 置信度分数
        iou_threshold: IoU 阈值

    Returns:
        保留的索引列表
    """
    if len(boxes) == 0:
        return []

    # 重要：cv2.dnn.NMSBoxes 需要 (x, y, w, h) 格式
    boxes_xywh = xyxy2xywh(boxes)

    # 使用 OpenCV 的 NMS 实现
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,  # 已在外部筛选
        nms_threshold=iou_threshold
    )

    if len(indices) > 0:
        return indices.flatten().tolist()
    return []


# ============================================================================
# 以下函数已废弃（Deprecated）
# 原因：已改为直接拉伸模式（与 YOLO Face 一致），不再使用 letterbox + padding
# 保留这些函数仅供参考，实际已不再调用
# ============================================================================

def scale_boxes(boxes: np.ndarray, scale: float, pad_w: int, pad_h: int) -> np.ndarray:
    """
    [DEPRECATED] 将边界框坐标从模型输入空间还原到原图空间（letterbox 模式）

    该函数已废弃，新实现使用直接拉伸模式，不需要处理 padding。

    Args:
        boxes: (N, 4) 边界框，格式 [x1, y1, x2, y2]
        scale: letterbox 缩放比例
        pad_w: letterbox 水平填充
        pad_h: letterbox 垂直填充

    Returns:
        还原后的边界框
    """
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale  # x1, x2
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale  # y1, y2
    return boxes


def scale_coords(coords: np.ndarray, scale: float, pad_w: int, pad_h: int) -> np.ndarray:
    """
    [DEPRECATED] 将关键点坐标从模型输入空间还原到原图空间（letterbox 模式）

    该函数已废弃，新实现使用直接拉伸模式，不需要处理 padding。

    Args:
        coords: (N, 17, 2) 或 (17, 2) 关键点坐标
        scale: letterbox 缩放比例
        pad_w: letterbox 水平填充
        pad_h: letterbox 垂直填充

    Returns:
        还原后的坐标
    """
    coords = coords.copy()
    coords[..., 0] = (coords[..., 0] - pad_w) / scale  # x
    coords[..., 1] = (coords[..., 1] - pad_h) / scale  # y
    return coords


class TRTPoseDetector:
    """
    TensorRT YOLO Pose 推理器

    功能:
    - 加载 .engine 文件并创建执行上下文
    - 预处理输入图像（letterbox + normalization）
    - 执行推理并解析关键点输出
    - 返回与 Ultralytics YOLO 兼容的检测结果格式
    """

    def __init__(
        self,
        engine_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        初始化 TensorRT YOLO Pose 推理器

        Args:
            engine_path: TensorRT 引擎文件路径 (.engine)
            confidence_threshold: 检测置信度阈值（默认 0.25，适用于 sigmoid 后的概率值）
            iou_threshold: NMS IoU 阈值
            input_size: 模型输入尺寸 (height, width)
        """
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT 或 PyCUDA 未安装，无法使用 TensorRT 后端。\n"
                "请安装: pip install tensorrt pycuda"
            )

        self.engine_path = str(Path(engine_path).resolve())
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = input_size  # (H, W)

        # CUDA 上下文管理（手动初始化以避免与 PyTorch 冲突）
        self.cuda_ctx = None
        self._init_cuda_context()

        # TensorRT 组件
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None

        # 输入/输出绑定
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []

        # 输出维度信息（自动检测）
        self.output_shape = None         # 初始引擎 binding shape (用于日志和回退)
        self.output_binding_name = None  # 输出 binding 名称
        self.output_binding_idx = None   # 输出 binding 索引（用于运行时查询）
        self.num_keypoints = 17          # 默认值，将从引擎推断
        self.output_dim = 56             # 默认值，将从引擎推断

        # 统计信息
        self._inference_count = 0
        self._total_time_ms = 0.0
        self._last_latency_ms = None

        # 加载引擎
        self._load_engine()

        logger.info(
            f"TensorRT YOLO Pose 推理器初始化完成: {self.engine_path}, "
            f"输入尺寸: {self.input_size}, 置信度阈值: {self.confidence_threshold}"
        )

    def _init_cuda_context(self):
        """
        初始化 CUDA 上下文（手动管理）

        注意：
        - 避免使用 pycuda.autoinit，以免与 PyTorch 的 CUDA 上下文冲突
        - 如果已有 CUDA 上下文（如 PyTorch 创建），则复用
        """
        try:
            # 尝试初始化 CUDA
            cuda.init()

            # 检查是否已有上下文
            try:
                current_ctx = cuda.Context.get_current()
                if current_ctx:
                    logger.info("检测到已存在的 CUDA 上下文，复用")
                    self.cuda_ctx = current_ctx
                    return
            except cuda.LogicError:
                # 无当前上下文，创建新的
                pass

            # 创建新的 CUDA 上下文
            device = cuda.Device(0)
            self.cuda_ctx = device.make_context()
            logger.info("创建新的 CUDA 上下文（设备 0）")

        except Exception as e:
            logger.warning(f"CUDA 上下文初始化失败: {e}，尝试继续（可能依赖现有上下文）")
            self.cuda_ctx = None

    def _load_engine(self):
        """加载 TensorRT 引擎并创建执行上下文"""
        engine_file = Path(self.engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(f"TensorRT 引擎文件不存在: {self.engine_path}")

        logger.info(f"加载 TensorRT 引擎: {self.engine_path}")

        # 加载序列化引擎
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"无法反序列化 TensorRT 引擎: {self.engine_path}")

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 创建 CUDA stream
        self.stream = cuda.Stream()

        # 分配输入/输出缓冲区
        self._allocate_buffers()

        logger.info(f"TensorRT 引擎加载成功，输入绑定数: {len(self.host_inputs)}, "
                   f"输出绑定数: {len(self.host_outputs)}")

    def _allocate_buffers(self):
        """分配输入/输出 GPU 缓冲区"""
        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            shape = self.engine.get_binding_shape(binding_idx)

            # 计算缓冲区大小
            size = trt.volume(shape)

            # 分配 host 和 device 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            # 保存绑定
            self.bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(binding_idx):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                logger.debug(f"输入绑定 [{binding_idx}] {binding_name}: shape={shape}, dtype={dtype}")
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

                # 保存输出 binding 信息（用于运行时动态查询）
                self.output_shape = tuple(shape)  # 初始 shape（用于日志和回退）
                self.output_binding_name = binding_name
                self.output_binding_idx = binding_idx  # 保存索引，用于动态查询

                # 自动推断关键点数量和输出维度
                # YOLOv8 Pose 输出可能是:
                # - (1, 56, 8400)  -> 通道优先，需要转置为 (1, 8400, 56)
                # - (1, 8400, 56)  -> anchor 优先，无需转置
                # - (batch, 56, N) -> 动态 batch + 通道优先
                # - (batch, N, 56) -> 动态 batch + anchor 优先
                if len(shape) >= 2:
                    # 推断 output_dim: 应该是较小的那个维度（56 vs 8400）
                    dim1, dim2 = shape[1], shape[2] if len(shape) > 2 else shape[1]
                    if dim1 == 56 or dim2 == 56:
                        self.output_dim = 56
                    elif dim1 < dim2:
                        self.output_dim = dim1
                    else:
                        self.output_dim = dim2

                    # 推断关键点数量
                    if self.output_dim > 5:
                        self.num_keypoints = (self.output_dim - 5) // 3
                        logger.info(
                            f"TensorRT 输出 binding 检测: shape={shape}, "
                            f"output_dim={self.output_dim}, "
                            f"num_keypoints={self.num_keypoints}"
                        )

                logger.debug(f"输出绑定 [{binding_idx}] {binding_name}: shape={shape}, dtype={dtype}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理输入图像（letterbox 模式，保持宽高比 + 灰色 padding）

        Args:
            image: 输入图像 (H, W, 3)，BGR 或 RGB

        Returns:
            Tuple[preprocessed, scale, (pad_w, pad_h)]:
            - preprocessed: 预处理后的图像 (1, 3, 640, 640), float32, [0, 1]
            - scale: 缩放比例
            - (pad_w, pad_h): letterbox 填充尺寸
        """
        img_h, img_w = image.shape[:2]
        target_h, target_w = self.input_size  # (640, 640)

        # 计算缩放比例（保持宽高比）
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize（保持宽高比）
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding（灰色填充）
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # 灰色填充
        )

        # BGR -> RGB
        if padded.shape[2] == 3:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, normalize to [0, 1]
        preprocessed = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=0)  # (1, 3, H, W)

        # 1️⃣ 预处理阶段日志
        import os
        if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
            logger.debug(
                f"[Pose] preprocess: orig={img_w}x{img_h} -> resized={new_w}x{new_h} "
                f"(scale={scale:.4f}, pad=({pad_w},{pad_h}))"
            )

        return preprocessed, scale, (pad_w, pad_h)

    def _reshape_predictions(
        self,
        raw_output: np.ndarray,
        binding_shape: Tuple[int, ...],
        output_dim: int
    ) -> np.ndarray:
        """
        统一的 reshape 工具函数，处理各种 TensorRT 输出格式

        Args:
            raw_output: TensorRT 原始输出，1D 展平数组
            binding_shape: 引擎 binding 的原始 shape（从 get_binding_shape 获取）
            output_dim: 期望的输出维度（例如 56 表示 4+1+17*3）

        Returns:
            重塑后的 predictions，格式固定为 (batch, num_boxes, output_dim)

        说明:
            支持的 binding_shape 格式:
            - (1, 56, 8400)   -> 通道优先 (N, C, K)，需转置为 (N, K, C)
            - (1, 8400, 56)   -> anchor 优先 (N, K, C)，直接 reshape
            - (batch, 56, N)  -> 动态 batch + 通道优先
            - (batch, N, 56)  -> 动态 batch + anchor 优先
            - (-1, 56, 8400)  -> 动态维度（TensorRT 返回 -1）
        """
        import os

        # 1. 处理动态形状：从实际数据推断真实 shape
        total_elements = raw_output.size
        if -1 in binding_shape:
            # 动态 batch 或 dynamic axes，需要推断
            if len(binding_shape) == 3:
                # 假设格式为 (-1, dim1, dim2) 或 (batch, -1, dim2)
                batch_idx = binding_shape.index(-1)
                other_dims = [d for d in binding_shape if d != -1]

                if len(other_dims) == 2:
                    dim_product = other_dims[0] * other_dims[1]
                    inferred_batch = total_elements // dim_product

                    if batch_idx == 0:
                        actual_shape = (inferred_batch, other_dims[0], other_dims[1])
                    else:
                        # 不常见情况，保守处理
                        actual_shape = binding_shape
                        logger.warning(
                            f"动态形状推断: binding_shape={binding_shape}, "
                            f"total_elements={total_elements}, 使用原始 shape"
                        )
                else:
                    actual_shape = binding_shape
            else:
                actual_shape = binding_shape
        else:
            actual_shape = binding_shape

        # 2. 验证总元素数一致性
        expected_elements = np.prod([d for d in actual_shape if d > 0])
        if total_elements != expected_elements:
            logger.warning(
                f"元素数不匹配: raw_output.size={total_elements}, "
                f"binding_shape={binding_shape} 期望 {expected_elements}。"
                f"尝试基于 output_dim={output_dim} 进行 fallback reshape。"
            )
            # Fallback: 基于 output_dim 强制 reshape
            if total_elements % output_dim == 0:
                num_boxes = total_elements // output_dim
                predictions = raw_output.reshape(1, num_boxes, output_dim)
                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug(
                        f"[Reshape] Fallback: {total_elements} elements -> "
                        f"(1, {num_boxes}, {output_dim})"
                    )
                return predictions
            else:
                raise ValueError(
                    f"无法 reshape: total_elements={total_elements} 不能被 "
                    f"output_dim={output_dim} 整除"
                )

        # 3. 根据 actual_shape 判断格式并 reshape
        if len(actual_shape) == 3:
            n, dim1, dim2 = actual_shape

            # 判断是否为通道优先格式 (N, C, K)
            # 规则: 如果 dim1 == output_dim 且 dim2 != output_dim，则为通道优先
            if dim1 == output_dim and dim2 != output_dim:
                # 通道优先 (N, C, K) -> reshape 后转置为 (N, K, C)
                reshaped = raw_output.reshape(n, dim1, dim2)
                predictions = reshaped.transpose(0, 2, 1)  # (N, K, C)

                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug(
                        f"[Reshape] 通道优先: binding_shape={actual_shape} -> "
                        f"reshape({n}, {dim1}, {dim2}) -> "
                        f"transpose -> ({n}, {dim2}, {dim1})"
                    )

            # 判断是否为 anchor 优先格式 (N, K, C)
            elif dim2 == output_dim:
                # anchor 优先，直接 reshape
                predictions = raw_output.reshape(n, dim1, dim2)

                if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                    logger.debug(
                        f"[Reshape] Anchor 优先: binding_shape={actual_shape} -> "
                        f"reshape({n}, {dim1}, {dim2})"
                    )

            else:
                # 未知格式，使用启发式规则
                # 假设较小的维度是 output_dim
                if dim1 < dim2:
                    # 可能是 (N, C, K)
                    reshaped = raw_output.reshape(n, dim1, dim2)
                    predictions = reshaped.transpose(0, 2, 1)
                    logger.warning(
                        f"启发式推断（通道优先）: binding_shape={actual_shape} -> "
                        f"reshape + transpose"
                    )
                else:
                    # 可能是 (N, K, C)
                    predictions = raw_output.reshape(n, dim1, dim2)
                    logger.warning(
                        f"启发式推断（anchor 优先）: binding_shape={actual_shape} -> "
                        f"reshape only"
                    )

        elif len(actual_shape) == 2:
            # 2D 输出，直接 reshape 并添加 batch 维度
            predictions = raw_output.reshape(1, actual_shape[0], actual_shape[1])

            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                logger.debug(f"[Reshape] 2D: binding_shape={actual_shape} -> add batch dim")

        else:
            # 其他情况，fallback 到基于 output_dim 的 reshape
            if total_elements % output_dim == 0:
                num_boxes = total_elements // output_dim
                predictions = raw_output.reshape(1, num_boxes, output_dim)
                logger.warning(
                    f"未知 binding_shape={actual_shape}，使用 fallback reshape: "
                    f"(1, {num_boxes}, {output_dim})"
                )
            else:
                raise ValueError(
                    f"无法处理的 binding_shape={actual_shape}, "
                    f"total_elements={total_elements}, output_dim={output_dim}"
                )

        # 4. 最终验证：确保最后一维等于 output_dim
        if predictions.shape[-1] != output_dim:
            raise ValueError(
                f"Reshape 后验证失败: predictions.shape={predictions.shape}, "
                f"期望最后一维为 {output_dim}"
            )

        return predictions

    def postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        后处理推理输出（letterbox 模式）

        Args:
            outputs: TensorRT 输出 tensor，1D 展开数组
            scale: 预处理时的缩放比例
            padding: letterbox 填充 (pad_w, pad_h)
            original_shape: 原始图像尺寸 (h, w)

        Returns:
            检测结果列表，每个元素格式:
            {
                'bbox': {'x1': float, 'y1': float, 'x2': float, 'y2': float},
                'confidence': float,
                'class_id': int,
                'keypoints': [
                    {'index': int, 'x': float, 'y': float, 'confidence': float},
                    ...
                ]
            }
        """
        pad_w, pad_h = padding

        # ========== 使用统一的 reshape 函数处理 TensorRT 输出 ==========
        # 替代原有的简单 reshape + 转置逻辑
        # 现在支持:
        # - (1, 56, 8400)   -> 通道优先，自动转置
        # - (1, 8400, 56)   -> anchor 优先，直接 reshape
        # - 动态 batch 格式
        # - 动态维度（-1）
        # - 动态 profile 切换（运行时查询真实 binding shape）
        try:
            # 运行时动态获取当前帧的真实 binding shape
            # 这样即使通过 context.set_binding_shape() 切换了 profile，也能正确处理
            if self.output_binding_idx is not None:
                try:
                    runtime_shape = self.context.get_binding_shape(self.output_binding_idx)
                    actual_binding_shape = tuple(runtime_shape)
                except Exception as e:
                    # 如果运行时查询失败（例如旧版 TensorRT），回退到初始 shape
                    logger.warning(
                        f"运行时查询 binding shape 失败: {e}，使用初始 shape {self.output_shape}"
                    )
                    actual_binding_shape = self.output_shape
            else:
                # 没有保存 binding index（不应该发生），使用初始 shape
                actual_binding_shape = self.output_shape

            # 调用统一 reshape 函数
            predictions = self._reshape_predictions(
                raw_output=outputs,
                binding_shape=actual_binding_shape,
                output_dim=self.output_dim
            )

            # 使用自动检测的关键点数量
            num_kps = self.num_keypoints

            # 调试日志：输出 reshape 后的统计信息
            import os
            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                # 显示运行时 shape（如果与初始 shape 不同，说明发生了 profile 切换）
                shape_info = f"runtime_shape={actual_binding_shape}"
                if actual_binding_shape != self.output_shape:
                    shape_info += f" (初始: {self.output_shape}, 已切换 profile)"

                logger.debug(
                    f"[Reshape] {shape_info}, "
                    f"最终 predictions.shape={predictions.shape}, "
                    f"output_dim={self.output_dim}, num_keypoints={num_kps}"
                )
                # 打印前几个 box 的 objectness 置信度（原始 logit 值）
                if predictions.shape[1] > 0:
                    sample_obj_logits = predictions[0, :min(5, predictions.shape[1]), 4]
                    logger.debug(
                        f"[Reshape] 前 {len(sample_obj_logits)} 个 box 的 obj_logit: "
                        f"{sample_obj_logits}"
                    )

        except Exception as e:
            # 尝试获取运行时 shape 用于错误诊断
            try:
                runtime_shape = self.context.get_binding_shape(self.output_binding_idx)
                shape_info = f"runtime_shape={tuple(runtime_shape)}, initial_shape={self.output_shape}"
            except:
                shape_info = f"binding_shape={self.output_shape}"

            logger.error(
                f"Reshape TensorRT 输出失败: {e}, "
                f"raw_output.size={outputs.size}, "
                f"{shape_info}, "
                f"output_dim={self.output_dim}"
            )
            return []

        # 提取第一个 batch（单图推理）
        preds = predictions[0]  # (num_boxes, output_dim)

        # 解析输出（动态适配关键点数量）
        # [0:4]   -> bbox (cx, cy, w, h)
        # [4]     -> objectness confidence (logit, 需要 sigmoid)
        # [5:]    -> num_keypoints * 3 (x, y, conf)
        boxes_xywh = preds[:, :4]  # (num_boxes, 4)
        obj_conf = preds[:, 4]     # (num_boxes,) - 原始 logit 值

        # 重要：TensorRT 输出的是 logit 值，需要 sigmoid 转换为概率
        # confidences = sigmoid(obj_conf) = 1 / (1 + exp(-obj_conf))
        confidences = 1.0 / (1.0 + np.exp(-obj_conf))

        kps_raw = preds[:, 5:]     # (num_boxes, num_kps*3)

        # 调试：打印 boxes_xywh 范围，判断 engine 输出类型
        # 范围 0~640 = 已解码像素坐标，范围 0~2 = raw 预测值
        import os
        if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
            logger.info(
                f"[EngineOutput] boxes_xywh range: "
                f"cx=[{boxes_xywh[:, 0].min():.2f}, {boxes_xywh[:, 0].max():.2f}], "
                f"cy=[{boxes_xywh[:, 1].min():.2f}, {boxes_xywh[:, 1].max():.2f}], "
                f"w=[{boxes_xywh[:, 2].min():.2f}, {boxes_xywh[:, 2].max():.2f}], "
                f"h=[{boxes_xywh[:, 3].min():.2f}, {boxes_xywh[:, 3].max():.2f}]"
            )

        # 1. 过滤低置信度检测
        mask = confidences >= self.confidence_threshold
        num_before_filter = len(confidences)
        num_after_filter = mask.sum()
        max_conf = confidences.max() if confidences.size else 0.0

        # 2️⃣ 后处理阶段日志 - 过滤前统计
        import os
        if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
            logger.debug(
                f"[Pose] boxes before filter: total={num_before_filter}, "
                f"max_conf={max_conf:.3f}, threshold={self.confidence_threshold:.2f}"
            )

        if num_after_filter == 0:
            logger.debug(
                f"所有候选框被过滤: {num_before_filter} 个候选框，"
                f"最高置信度 {max_conf:.4f}，"
                f"阈值 {self.confidence_threshold:.2f}"
            )
            return []

        logger.debug(
            f"置信度过滤: {num_before_filter} -> {num_after_filter} 个候选框，"
            f"最高置信度 {confidences[mask].max():.4f}"
        )

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        kps_raw = kps_raw[mask]

        # 2. 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        # 3. 应用 NMS
        keep_indices = nms_boxes(boxes_xyxy, confidences, self.iou_threshold)
        num_after_nms = len(keep_indices)

        # 2️⃣ 后处理阶段日志 - NMS 后统计
        if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
            logger.debug(
                f"[Pose] boxes after NMS: after_conf={num_after_filter}, "
                f"after_nms={num_after_nms}, iou_threshold={self.iou_threshold:.2f}"
            )

        if not keep_indices:
            logger.debug(f"NMS 后无保留框")
            return []

        logger.debug(f"NMS 保留: {num_after_nms} / {len(boxes_xyxy)} 个框")

        # 4. 构造检测结果
        detections = []
        for idx in keep_indices:
            box = boxes_xyxy[idx]
            conf = float(confidences[idx])
            kps = kps_raw[idx]  # (51,)

            # 解析关键点: [x1, y1, c1, x2, y2, c2, ...]
            kps_reshaped = kps.reshape(num_kps, 3)  # (num_kps, 3)
            kps_xy = kps_reshaped[:, :2]  # (num_kps, 2) - 归一化值 [0, 1]
            kps_conf_logit = kps_reshaped[:, 2]  # (num_kps,) - 原始 logit 值

            # 重要：关键点置信度也是 logit 值，需要 sigmoid 转换
            kps_conf = 1.0 / (1.0 + np.exp(-kps_conf_logit))
            kps_conf = np.clip(kps_conf, 0.0, 1.0)

            # ========== letterbox 模式的坐标还原（正确处理）==========
            # 判断 YOLO Pose 输出格式：归一化值 [0, 1] 还是像素值 [0, 640]
            # 根据实际观察，某些 TensorRT 引擎输出已经是像素坐标

            target_h, target_w = self.input_size  # (640, 640)

            # 步骤 1: 检测坐标格式并转换为 letterbox 像素坐标
            if kps_xy.max() <= 1.5:
                # 真正的归一化坐标 [0, 1]，需要乘以目标尺寸
                kps_xy_letterbox = kps_xy.copy()
                kps_xy_letterbox[:, 0] = kps_xy[:, 0] * target_w
                kps_xy_letterbox[:, 1] = kps_xy[:, 1] * target_h
            else:
                # 输出已经是 letterbox 像素坐标 [0, 640]，直接使用
                kps_xy_letterbox = kps_xy.copy()

            # 步骤 2: letterbox 像素坐标 → 原图坐标（减 padding，除 scale）
            kps_xy_scaled = kps_xy_letterbox.copy()
            kps_xy_scaled[:, 0] = (kps_xy_letterbox[:, 0] - pad_w) / scale
            kps_xy_scaled[:, 1] = (kps_xy_letterbox[:, 1] - pad_h) / scale

            # 边界框同样处理
            box_scaled = box.copy()
            box_scaled[[0, 2]] = (box_scaled[[0, 2]] - pad_w) / scale
            box_scaled[[1, 3]] = (box_scaled[[1, 3]] - pad_h) / scale

            # 调试日志：验证关键点坐标转换（可通过环境变量 AITABLE_DEBUG_POSE=1 启用）
            import os
            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                logger.debug(
                    f"[PoseCoords] 坐标转换链（letterbox 模式）: "
                    f"原始值范围=[{kps_reshaped[:, :2].min():.3f}, {kps_reshaped[:, :2].max():.3f}] -> "
                    f"letterbox像素范围=[{kps_xy_letterbox.min():.1f}, {kps_xy_letterbox.max():.1f}] -> "
                    f"原图像素范围=[{kps_xy_scaled.min():.1f}, {kps_xy_scaled.max():.1f}] "
                    f"(scale={scale:.3f}, pad=({pad_w},{pad_h}))"
                )
                # 打印关键关键点（眼睛、肩膀、鼻子）的具体坐标
                key_indices = [0, 1, 2, 5, 6]  # 鼻子、左眼、右眼、左肩、右肩
                kp_names = ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']
                for i, kp_idx in enumerate(key_indices):
                    if kp_idx < num_kps and kps_conf[kp_idx] > 0.3:
                        logger.debug(
                            f"  KP[{kp_idx}] {kp_names[i]}: "
                            f"raw=({kps_reshaped[kp_idx, 0]:.3f}, {kps_reshaped[kp_idx, 1]:.3f}) -> "
                            f"letterbox=({kps_xy_letterbox[kp_idx, 0]:.1f}, {kps_xy_letterbox[kp_idx, 1]:.1f}) -> "
                            f"orig=({kps_xy_scaled[kp_idx, 0]:.1f}, {kps_xy_scaled[kp_idx, 1]:.1f}), "
                            f"conf={kps_conf[kp_idx]:.3f}"
                        )

            # 限制坐标在原图范围内
            orig_h, orig_w = original_shape
            box_scaled = np.clip(box_scaled, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])
            kps_xy_scaled[:, 0] = np.clip(kps_xy_scaled[:, 0], 0, orig_w)
            kps_xy_scaled[:, 1] = np.clip(kps_xy_scaled[:, 1], 0, orig_h)

            # 构造关键点列表（使用动态关键点数量）
            keypoints_list = []
            for kp_idx in range(num_kps):
                keypoints_list.append({
                    'index': kp_idx,
                    'x': float(kps_xy_scaled[kp_idx, 0]),
                    'y': float(kps_xy_scaled[kp_idx, 1]),
                    'confidence': float(kps_conf[kp_idx]),  # 已裁剪到 [0, 1]
                })

            # 添加检测结果
            detection_dict = {
                'bbox': {
                    'x1': float(box_scaled[0]),
                    'y1': float(box_scaled[1]),
                    'x2': float(box_scaled[2]),
                    'y2': float(box_scaled[3]),
                },
                'confidence': conf,
                'class_id': 0,  # person 类
                'keypoints': keypoints_list,
            }
            detections.append(detection_dict)

            # 2️⃣ 后处理阶段日志 - 检测输出摘要
            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                # 打印关键关键点的坐标和置信度（鼻子、左肩、右肩）
                nose_x = kps_xy_scaled[0, 0] if num_kps > 0 else 0
                nose_y = kps_xy_scaled[0, 1] if num_kps > 0 else 0
                nose_conf = kps_conf[0] if num_kps > 0 else 0

                left_shoulder_x = kps_xy_scaled[5, 0] if num_kps > 5 else 0
                left_shoulder_y = kps_xy_scaled[5, 1] if num_kps > 5 else 0
                left_shoulder_conf = kps_conf[5] if num_kps > 5 else 0

                logger.debug(
                    f"[Pose] detection #{len(detections)}: conf={conf:.3f}, "
                    f"nose=({nose_x:.1f}, {nose_y:.1f}, {nose_conf:.2f}), "
                    f"left_shoulder=({left_shoulder_x:.1f}, {left_shoulder_y:.1f}, {left_shoulder_conf:.2f})"
                )

        return detections

    def detect_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测单帧图像中的人体关键点

        Args:
            frame: 输入图像 (H, W, 3)，支持 BGR/RGB

        Returns:
            包含检测结果的字典，若未检测到人体则返回 None
            格式与 BodyKeyPointDetector.detect_keypoints() 保持一致
        """
        if frame is None or frame.size == 0:
            logger.warning("TRT Pose: 检测输入为空，跳过本帧")
            return None

        start = time.perf_counter()

        # 预处理（letterbox 模式）
        preprocessed, scale, padding = self.preprocess(frame)

        # 拷贝输入数据到 GPU
        np.copyto(self.host_inputs[0], preprocessed.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 拷贝输出数据到 CPU
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        # 后处理
        output_data = self.host_outputs[0]  # 展开的 1D 数组

        # 注意：output_data 已经是 1D 数组，postprocess() 会自动 reshape
        detections = self.postprocess(
            output_data,
            scale,
            padding,
            (frame.shape[0], frame.shape[1])
        )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        self._update_stats(latency_ms)

        # 3️⃣ 推理阶段日志
        import os
        if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
            logger.debug(f"[Pose] inference latency: {latency_ms:.2f} ms")

        if not detections:
            if os.getenv('AITABLE_DEBUG_POSE', '0') == '1':
                logger.debug("[Pose] no detections in this frame")
            return None

        return {
            "detections": detections,
            "inference_time_ms": latency_ms,
            "device": "TensorRT",
        }

    def _update_stats(self, latency_ms: float):
        """更新推理统计信息"""
        self._last_latency_ms = latency_ms
        self._inference_count += 1
        self._total_time_ms += latency_ms

        if self._inference_count % 50 == 0:
            avg = self._total_time_ms / max(1, self._inference_count)
            fps = 1000.0 / avg if avg > 0 else 0.0
            logger.info(
                f"TRTPoseDetector 平均延迟: {avg:.2f} ms, 约 {fps:.1f} FPS "
                f"(前 {self._inference_count} 帧)"
            )

    def get_model_info(self) -> Dict:
        """返回模型状态信息"""
        avg_latency = (
            self._total_time_ms / self._inference_count if self._inference_count else None
        )
        info = {
            "engine_path": self.engine_path,
            "backend": "TensorRT",
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
        }
        if avg_latency is not None:
            info["avg_inference_latency_ms"] = round(avg_latency, 2)
            info["fps"] = round(1000.0 / avg_latency, 2) if avg_latency > 0 else 0.0
        if self._last_latency_ms is not None:
            info["last_inference_latency_ms"] = round(self._last_latency_ms, 2)
        return info

    def warmup(self, runs: int = 3):
        """预热模型，减少首帧延迟"""
        logger.info("开始进行 TensorRT YOLO Pose 推理器预热")
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect_keypoints(dummy)
        logger.info("TensorRT 预热完成")

    def __del__(self):
        """清理资源"""
        try:
            # 确保在正确的上下文中释放资源
            if self.cuda_ctx:
                self.cuda_ctx.push()

            # 释放 CUDA 内存
            for cuda_mem in self.cuda_inputs + self.cuda_outputs:
                try:
                    cuda_mem.free()
                except Exception:
                    pass

            # 销毁上下文和引擎
            if self.context:
                del self.context
            if self.engine:
                del self.engine

            # 如果是我们创建的上下文，销毁它
            if self.cuda_ctx:
                try:
                    self.cuda_ctx.pop()
                    # 注意：不调用 detach()，让 Python 垃圾回收处理
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"TRTPoseDetector 资源清理时出错: {e}")
