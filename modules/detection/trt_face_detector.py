"""
TensorRT YOLO Face 推理封装
独立于 Ultralytics，直接使用 TensorRT API 进行推理
兼容由 trtexec 或板级工具生成的 TensorRT engine 文件

主要特性:
- 直接使用 TensorRT Python API 加载 .engine 文件
- 手动管理 CUDA 上下文（避免与 PyTorch 冲突）
- 自动检测引擎输出格式（通道优先 vs anchor 优先）
- 完整的 YOLO 后处理（NMS + 坐标转换）
- 与 SimpleFaceDetector 兼容的输出格式
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


class TRTFaceDetector:
    """
    TensorRT YOLO Face 推理器

    功能:
    - 加载 .engine 文件并创建执行上下文
    - 预处理输入图像（直接拉伸到 640x640）
    - 执行推理并解析人脸框输出
    - 返回与 SimpleFaceDetector 兼容的检测结果格式
    """

    def __init__(
        self,
        engine_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        初始化 TensorRT YOLO Face 推理器

        Args:
            engine_path: TensorRT 引擎文件路径 (.engine)
            confidence_threshold: 检测置信度阈值（默认 0.25）
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
        self.output_shape = None  # 初始引擎 binding shape
        self.output_binding_name = None  # 输出 binding 名称
        self.output_binding_idx = None  # 输出 binding 索引（用于运行时查询）
        self.output_dim = 5  # YOLO Face: [x, y, w, h, conf]

        # 统计信息
        self._inference_count = 0
        self._total_time_ms = 0.0
        self._last_latency_ms = None

        # 加载引擎
        self._load_engine()

        logger.info(
            f"TensorRT YOLO Face 推理器初始化完成: {self.engine_path}, "
            f"输入尺寸: {self.input_size}, 置信度阈值: {self.confidence_threshold}"
        )

    def _init_cuda_context(self):
        """
        初始化 CUDA 上下文（共享主上下文模式）

        注意：
        - 使用 retain_primary_context() 与其他模块共享主上下文
        - 避免 make_context() 创建独立上下文（导致异步推理冲突）
        - 线程安全：多个线程可以同时使用同一个主上下文
        """
        try:
            # 初始化 CUDA 驱动
            cuda.init()

            # 获取设备并保留主上下文（与异步推理引擎兼容）
            self._device = cuda.Device(0)
            self.cuda_ctx = self._device.retain_primary_context()
            logger.info("使用共享 CUDA 主上下文（设备 0）- 线程安全模式")

        except Exception as e:
            logger.warning(f"CUDA 上下文初始化失败: {e}，尝试继续（可能依赖现有上下文）")
            self.cuda_ctx = None
            self._device = None

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

        # 确保上下文已推入，避免 PyCUDA 资源分配错误
        if self.cuda_ctx:
            self.cuda_ctx.push()
        
        try:
            # 创建 CUDA stream
            self.stream = cuda.Stream()

            # 分配输入/输出缓冲区
            self._allocate_buffers()
        finally:
            # 恢复上下文栈
            if self.cuda_ctx:
                self.cuda_ctx.pop()

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
                self.output_shape = tuple(shape)  # 初始 shape
                self.output_binding_name = binding_name
                self.output_binding_idx = binding_idx

                # 自动推断输出维度
                # YOLO Face 输出可能是:
                # - (1, 5, N)  -> 通道优先，需要转置为 (1, N, 5)
                # - (1, N, 5)  -> anchor 优先，无需转置
                if len(shape) >= 2:
                    dim1, dim2 = shape[1], shape[2] if len(shape) > 2 else shape[1]
                    if dim1 == 5 or dim2 == 5:
                        self.output_dim = 5
                    elif dim1 < dim2:
                        self.output_dim = dim1
                    else:
                        self.output_dim = dim2

                    logger.info(
                        f"TensorRT 输出 binding 检测: shape={shape}, "
                        f"output_dim={self.output_dim}"
                    )

                logger.debug(f"输出绑定 [{binding_idx}] {binding_name}: shape={shape}, dtype={dtype}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        预处理输入图像（直接拉伸模式，与 SimpleFaceDetector 一致）

        Args:
            image: 输入图像 (H, W, 3)，BGR 或 RGB

        Returns:
            Tuple[preprocessed, scale_x, scale_y]:
            - preprocessed: 预处理后的图像 (1, 3, 640, 640), float32, [0, 1]
            - scale_x: 宽度缩放比例
            - scale_y: 高度缩放比例
        """
        img_h, img_w = image.shape[:2]
        target_h, target_w = self.input_size  # (640, 640)

        # 直接拉伸到目标尺寸（不保持宽高比）
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        if resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, normalize to [0, 1]
        preprocessed = resized.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=0)  # (1, 3, H, W)

        # 计算缩放比例（用于坐标还原）
        scale_x = img_w / target_w
        scale_y = img_h / target_h

        return preprocessed, scale_x, scale_y

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
            output_dim: 期望的输出维度（例如 5 表示 x, y, w, h, conf）

        Returns:
            重塑后的 predictions，格式固定为 (batch, num_boxes, output_dim)

        说明:
            支持的 binding_shape 格式:
            - (1, 5, 8400)   -> 通道优先 (N, C, K)，需转置为 (N, K, C)
            - (1, 8400, 5)   -> anchor 优先 (N, K, C)，直接 reshape
            - (batch, 5, N)  -> 动态 batch + 通道优先
            - (batch, N, 5)  -> 动态 batch + anchor 优先
            - (-1, 5, 8400)  -> 动态维度（TensorRT 返回 -1）
        """
        # 1. 处理动态形状：从实际数据推断真实 shape
        total_elements = raw_output.size
        if -1 in binding_shape:
            # 动态 batch 或 dynamic axes，需要推断
            if len(binding_shape) == 3:
                batch_idx = binding_shape.index(-1)
                other_dims = [d for d in binding_shape if d != -1]

                if len(other_dims) == 2:
                    dim_product = other_dims[0] * other_dims[1]
                    inferred_batch = total_elements // dim_product

                    if batch_idx == 0:
                        actual_shape = (inferred_batch, other_dims[0], other_dims[1])
                    else:
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
            if dim1 == output_dim and dim2 != output_dim:
                # 通道优先 (N, C, K) -> reshape 后转置为 (N, K, C)
                reshaped = raw_output.reshape(n, dim1, dim2)
                predictions = reshaped.transpose(0, 2, 1)  # (N, K, C)
                logger.debug(
                    f"[Reshape] 通道优先: binding_shape={actual_shape} -> "
                    f"transpose -> ({n}, {dim2}, {dim1})"
                )

            # 判断是否为 anchor 优先格式 (N, K, C)
            elif dim2 == output_dim:
                # anchor 优先，直接 reshape
                predictions = raw_output.reshape(n, dim1, dim2)
                logger.debug(
                    f"[Reshape] Anchor 优先: binding_shape={actual_shape}"
                )

            else:
                # 未知格式，使用启发式规则
                if dim1 < dim2:
                    # 可能是 (N, C, K)
                    reshaped = raw_output.reshape(n, dim1, dim2)
                    predictions = reshaped.transpose(0, 2, 1)
                    logger.warning(
                        f"启发式推断（通道优先）: binding_shape={actual_shape}"
                    )
                else:
                    # 可能是 (N, K, C)
                    predictions = raw_output.reshape(n, dim1, dim2)
                    logger.warning(
                        f"启发式推断（anchor 优先）: binding_shape={actual_shape}"
                    )

        elif len(actual_shape) == 2:
            # 2D 输出，直接 reshape 并添加 batch 维度
            predictions = raw_output.reshape(1, actual_shape[0], actual_shape[1])
            logger.debug(f"[Reshape] 2D: binding_shape={actual_shape}")

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
        scale_x: float,
        scale_y: float,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        后处理推理输出（直接拉伸模式）

        Args:
            outputs: TensorRT 输出 tensor，1D 展开数组
            scale_x: 宽度缩放比例
            scale_y: 高度缩放比例
            original_shape: 原始图像尺寸 (h, w)

        Returns:
            检测结果列表，每个元素格式:
            {
                'face_bbox': {'x1': float, 'y1': float, 'x2': float, 'y2': float},
                'confidence': float,
                'class_id': int,
            }
        """
        # ========== 使用统一的 reshape 函数处理 TensorRT 输出 ==========
        try:
            # 运行时动态获取当前帧的真实 binding shape
            if self.output_binding_idx is not None:
                try:
                    runtime_shape = self.context.get_binding_shape(self.output_binding_idx)
                    actual_binding_shape = tuple(runtime_shape)
                except Exception as e:
                    logger.warning(
                        f"运行时查询 binding shape 失败: {e}，使用初始 shape {self.output_shape}"
                    )
                    actual_binding_shape = self.output_shape
            else:
                actual_binding_shape = self.output_shape

            # 调用统一 reshape 函数
            predictions = self._reshape_predictions(
                raw_output=outputs,
                binding_shape=actual_binding_shape,
                output_dim=self.output_dim
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

        # 解析输出
        # [0:4]   -> bbox (cx, cy, w, h)
        # [4]     -> objectness confidence (logit, 需要 sigmoid)
        boxes_xywh = preds[:, :4]  # (num_boxes, 4)
        obj_conf = preds[:, 4]     # (num_boxes,) - 原始 logit 值

        # 重要：TensorRT 输出的是 logit 值，需要 sigmoid 转换为概率
        confidences = 1.0 / (1.0 + np.exp(-obj_conf))

        # 1. 过滤低置信度检测
        mask = confidences >= self.confidence_threshold
        num_before_filter = len(confidences)
        num_after_filter = mask.sum()

        if num_after_filter == 0:
            max_conf = confidences.max() if confidences.size else 0.0
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

        # 2. 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        # 3. 应用 NMS
        keep_indices = nms_boxes(boxes_xyxy, confidences, self.iou_threshold)
        num_after_nms = len(keep_indices)

        if not keep_indices:
            logger.debug(f"NMS 后无保留框")
            return []

        logger.debug(f"NMS 保留: {num_after_nms} / {len(boxes_xyxy)} 个框")

        # 4. 构造检测结果
        detections = []
        for idx in keep_indices:
            box = boxes_xyxy[idx]
            conf = float(confidences[idx])

            # 坐标还原（直接拉伸模式）
            # 模型输出的坐标相对于 640x640 输入，需要乘以缩放比例还原到原图
            x1 = float(box[0] * scale_x)
            y1 = float(box[1] * scale_y)
            x2 = float(box[2] * scale_x)
            y2 = float(box[3] * scale_y)

            # 限制坐标在原图范围内
            orig_h, orig_w = original_shape
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # 添加检测结果（新格式：face_bbox）
            detection_dict = {
                'face_bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                },
                'confidence': conf,
                'class_id': 0,  # face 类
            }
            detections.append(detection_dict)

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测单帧图像中的人脸

        Args:
            frame: 输入图像 (H, W, 3)，支持 BGR/RGB

        Returns:
            检测结果列表，格式与 SimpleFaceDetector.detect() 保持一致
        """
        if frame is None or frame.size == 0:
            logger.warning("TRT Face: 检测输入为空，跳过本帧")
            return []

        start = time.perf_counter()

        # 预处理（直接拉伸模式）
        preprocessed, scale_x, scale_y = self.preprocess(frame)

        # 确保上下文已推入，支持异步推理调用
        if self.cuda_ctx:
            self.cuda_ctx.push()
        
        try:
            # 拷贝输入数据到 GPU
            np.copyto(self.host_inputs[0], preprocessed.ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

            # 执行推理
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # 拷贝输出数据到 CPU
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()
        finally:
            # 恢复上下文栈
            if self.cuda_ctx:
                self.cuda_ctx.pop()

        # 后处理
        output_data = self.host_outputs[0]  # 展开的 1D 数组

        detections = self.postprocess(
            output_data,
            scale_x,
            scale_y,
            (frame.shape[0], frame.shape[1])
        )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        self._update_stats(latency_ms)

        return detections

    def detect_face(self, rgb_frame: np.ndarray) -> Optional[Dict]:
        """
        检测单张人脸（与 SimpleFaceDetector 接口兼容）

        Args:
            rgb_frame: 输入图像 (H, W, 3)，RGB 格式

        Returns:
            单个人脸检测结果，格式:
            {
                'face_bbox': {'x1': int, 'y1': int, 'x2': int, 'y2': int},
                'left_eye': tuple,
                'right_eye': tuple,
                'confidence': float,
                'method': str
            }
            如果未检测到人脸，返回 None
        """
        detections = self.detect(rgb_frame)

        if not detections:
            return None

        # 选择最大面积且置信度最高的人脸
        if len(detections) > 1:
            best_detection = max(detections, key=lambda d: (
                # 先按面积排序
                (d['face_bbox']['x2'] - d['face_bbox']['x1']) * (d['face_bbox']['y2'] - d['face_bbox']['y1']),
                # 再按置信度排序
                d['confidence']
            ))
        else:
            best_detection = detections[0]

        bbox = best_detection['face_bbox']
        conf = best_detection['confidence']

        # 估算眼睛位置（基于人脸框）
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        face_width = x2 - x1
        face_height = y2 - y1

        # 根据人脸比例估算眼睛位置
        # 左眼约在人脸宽度的33%位置，高度的38%位置
        left_eye = (int(x1 + face_width * 0.33), int(y1 + face_height * 0.38))
        # 右眼约在人脸宽度的67%位置，高度的38%位置
        right_eye = (int(x1 + face_width * 0.67), int(y1 + face_height * 0.38))

        return {
            'face_bbox': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            },
            'left_eye': left_eye,
            'right_eye': right_eye,
            'confidence': conf,
            'method': 'TensorRT'
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
                f"TRTFaceDetector 平均延迟: {avg:.2f} ms, 约 {fps:.1f} FPS "
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
        logger.info("开始进行 TensorRT YOLO Face 推理器预热")
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(dummy)
        logger.info("TensorRT 预热完成")

    def close(self):
        """
        显式释放 CUDA 资源（推荐）

        重要：
        - 应在系统停机、热切换或长时间空闲时显式调用
        - 避免依赖 __del__，它在异常退出或循环引用时可能不执行
        - 调用后检测器不可再使用，需要重新实例化
        """
        if not hasattr(self, '_closed'):
            self._closed = False

        if self._closed:
            return  # 避免重复清理

        try:
            # 确保在正确的上下文中释放资源
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                self.cuda_ctx.push()

            # 释放 CUDA 内存
            if hasattr(self, 'cuda_inputs') and hasattr(self, 'cuda_outputs'):
                for cuda_mem in self.cuda_inputs + self.cuda_outputs:
                    try:
                        cuda_mem.free()
                    except Exception:
                        pass

            # 销毁上下文和引擎
            if hasattr(self, 'context') and self.context:
                del self.context
                self.context = None
            if hasattr(self, 'engine') and self.engine:
                del self.engine
                self.engine = None

            # 如果是我们创建的上下文，销毁它
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                try:
                    self.cuda_ctx.pop()
                    # 注意：不调用 detach()，让 Python 垃圾回收处理
                except Exception:
                    pass

            self._closed = True
            logger.info("TRTFaceDetector CUDA resources released")

        except Exception as e:
            logger.error(f"Error during TRTFaceDetector cleanup: {e}")
            raise

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口 - 确保资源释放"""
        self.close()
        return False  # 不抑制异常

    def __del__(self):
        """
        析构函数 - 最后的资源清理手段

        注意：
        - 仅作为兜底措施，不应依赖此方法
        - 优先使用显式 close() 或 with 语句
        - 在异常退出或循环引用时可能不执行
        """
        try:
            self.close()
        except Exception as e:
            # 已在 close() 中记录日志，这里只静默处理
            pass
