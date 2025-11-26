"""
TensorRT 情绪识别分类器（原生 API 版本）
使用 TensorRT 原生 Python API，支持多线程

替代 PyCUDA，提供线程安全的推理
"""
import numpy as np
import cv2
from typing import List, Dict, Optional
from pathlib import Path

from ..core.logger import logger
from ..core.trt_engine import TRTEngineBase


class TRTEmoNetClassifier(TRTEngineBase):
    """
    EmoNet TensorRT 情绪识别分类器（原生 API 版本）

    输入: 人脸图像 (256x256 RGB)
    输出:
        - expression: 情绪类别（8类）
        - valence: 效价（-1 到 1）
        - arousal: 唤醒度（-1 到 1）

    特性:
    - 线程安全（使用 TensorRT 原生 API）
    - 无 PyCUDA 依赖
    - 支持多线程并发推理
    """

    # 情绪标签（EmoNet 8类情绪）
    EMOTION_LABELS = [
        'neutral',    # 中性
        'happy',      # 快乐
        'sad',        # 悲伤
        'surprise',   # 惊讶
        'fear',       # 恐惧
        'disgust',    # 厌恶
        'anger',      # 愤怒
        'contempt'    # 轻蔑
    ]

    def __init__(self, engine_path: Optional[str] = None, input_size: int = 256):
        """
        初始化 TensorRT EmoNet 分类器

        Args:
            engine_path: TensorRT 引擎文件路径 (.engine)
                        如果未提供，从 system_config.json 读取
            input_size: 输入图像尺寸（默认 256x256）

        Raises:
            FileNotFoundError: 引擎文件不存在
            RuntimeError: TensorRT 初始化失败
        """
        # 从配置文件获取模型路径（如果未提供）
        if engine_path is None:
            from ..core.config_loader import get_config
            config = get_config()

            # 使用 resolve_model_path 解析模型路径（自动处理 primary/fallback）
            resolved_path = config.resolve_model_path("emonet")

            if resolved_path is None:
                # 获取配置信息用于错误提示
                emonet_config = config.models.get("emonet")
                expected = emonet_config.get("primary") if emonet_config else "models/emonet_fp16.engine"
                raise FileNotFoundError(
                    f"EmoNet TensorRT engine not found.\n"
                    f"Expected: {expected}\n"
                    f"Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)"
                )

            engine_path = str(resolved_path)

        # 初始化基类
        super().__init__(engine_path)

        self.input_size = input_size

        # 获取输入输出名称（EmoNet 引擎的绑定名称）
        self.input_name = self._find_binding_name("input")
        self.output_names = self._find_output_names()

        # 分配 CUDA 缓冲区
        self._allocate_buffers()

        logger.info("TRTEmoNetClassifier (native API) initialized successfully")

    def _find_binding_name(self, prefix: str) -> str:
        """查找包含指定前缀的绑定名称"""
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if prefix.lower() in name.lower():
                return name
        # 如果找不到，返回第一个输入绑定
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                continue
            return self.engine.get_binding_name(i)
        raise ValueError(f"No input binding found with prefix '{prefix}'")

    def _find_output_names(self) -> List[str]:
        """查找所有输出绑定名称"""
        outputs = []
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                outputs.append(self.engine.get_binding_name(i))
        return outputs

    def _allocate_buffers(self):
        """分配 CUDA 缓冲区"""
        bindings = {}

        # 输入绑定
        input_shape = self.get_binding_shape(self.input_name)
        bindings[self.input_name] = input_shape

        # 输出绑定
        for name in self.output_names:
            output_shape = self.get_binding_shape(name)
            bindings[name] = output_shape

        self.allocate_buffers(bindings)

        logger.debug(f"Allocated buffers for input '{self.input_name}' and {len(self.output_names)} outputs")

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        预处理人脸图像

        Args:
            face_img: BGR 格式的人脸图像

        Returns:
            预处理后的图像 (1, 3, 256, 256)
        """
        # BGR -> RGB
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize to input size
        img = cv2.resize(img, (self.input_size, self.input_size))

        # 归一化 [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # 添加 batch 维度: (3, 256, 256) -> (1, 3, 256, 256)
        img = np.expand_dims(img, axis=0)

        # 确保连续内存布局
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img

    def _postprocess(self, outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        后处理推理结果

        Args:
            outputs: TensorRT 输出字典

        Returns:
            情绪识别结果 {emotion, valence, arousal, confidence, ...}
        """
        # EmoNet 输出通常包含:
        # - expression: (1, 8) 情绪类别概率
        # - valence: (1, 1) 效价值
        # - arousal: (1, 1) 唤醒度

        result = {}

        # 查找 expression 输出
        expr_output = None
        for name, data in outputs.items():
            if 'expr' in name.lower() or 'output' in name.lower():
                if data.shape[-1] == 8:  # 8 类情绪
                    expr_output = data
                    break

        if expr_output is not None:
            # 获取情绪类别
            expr_probs = expr_output.flatten()
            emotion_idx = np.argmax(expr_probs)
            confidence = float(expr_probs[emotion_idx])

            result['emotion'] = self.EMOTION_LABELS[emotion_idx]
            result['confidence'] = confidence
            result['probabilities'] = {
                label: float(prob)
                for label, prob in zip(self.EMOTION_LABELS, expr_probs)
            }

        # 查找 valence/arousal 输出
        for name, data in outputs.items():
            name_lower = name.lower()
            if 'valence' in name_lower:
                result['valence'] = float(data.flatten()[0])
            elif 'arousal' in name_lower:
                result['arousal'] = float(data.flatten()[0])

        # 如果未找到情绪输出，使用第一个输出
        if 'emotion' not in result and len(outputs) > 0:
            first_output = list(outputs.values())[0]
            if first_output.shape[-1] == 8:
                expr_probs = first_output.flatten()
                emotion_idx = np.argmax(expr_probs)
                result['emotion'] = self.EMOTION_LABELS[emotion_idx]
                result['confidence'] = float(expr_probs[emotion_idx])

        return result

    def predict_single(self, face_img: np.ndarray) -> Optional[Dict[str, float]]:
        """
        对单张人脸图像进行情绪识别

        Args:
            face_img: 输入图像 (BGR 格式)

        Returns:
            情绪识别结果字典，包含:
                - emotion: 情绪类别字符串
                - confidence: 置信度 [0, 1]
                - valence: 效价 [-1, 1]（可选）
                - arousal: 唤醒度 [-1, 1]（可选）
                - probabilities: 各情绪类别概率
        """
        if face_img is None or face_img.size == 0:
            logger.warning("Empty face image provided")
            return None

        try:
            # 预处理
            input_data = self._preprocess(face_img)

            # 推理
            outputs = self.execute(
                inputs={self.input_name: input_data},
                output_names=self.output_names
            )

            # 后处理
            result = self._postprocess(outputs)

            return result

        except Exception as e:
            logger.error(f"Emotion recognition failed: {e}")
            return None

    def predict_batch(self, face_batch: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        对一批人脸图像进行情绪识别

        Args:
            face_batch: 人脸图像列表

        Returns:
            情绪识别结果列表
        """
        if not face_batch:
            return []

        results = []
        for face_img in face_batch:
            result = self.predict_single(face_img)
            if result:
                results.append(result)

        return results


# 向后兼容别名
__all__ = ['TRTEmoNetClassifier']
