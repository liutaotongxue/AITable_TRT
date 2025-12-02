"""
TensorRT-based EmoNet classifier (native API, thread-safe).

- Uses TensorRT native Python API (no PyCUDA)
- Supports multithreaded inference
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Dict, Optional
from pathlib import Path

from ..core.logger import logger
from ..core.trt_engine import TRTEngineBase


class TRTEmoNetClassifier(TRTEngineBase):
    """
    TensorRT EmoNet classifier (native API).

    Input: face image (256x256 RGB)
    Output:
        - expression (8 classes)
        - valence (float)
        - arousal (float)
    """

    # Emotion labels for 8 classes
    EMOTION_LABELS = [
        "neutral",
        "happy",
        "sad",
        "surprise",
        "fear",
        "disgust",
        "anger",
        "contempt",
    ]

    def __init__(self, engine_path: Optional[str] = None, input_size: int = 256):
        """
        Initialize TensorRT EmoNet classifier.

        Args:
            engine_path: Path to TensorRT engine (.engine). If None, resolve via system_config.json.
            input_size: Face crop size (default 256x256).
        """
        # Resolve engine path from config if not provided
        if engine_path is None:
            from ..core.config_loader import get_config

            config = get_config()
            resolved_path = config.resolve_model_path("emonet")

            if resolved_path is None:
                emonet_config = config.models.get("emonet")
                expected = (
                    emonet_config.get("primary")
                    if emonet_config
                    else "models/emonet_fp16.engine"
                )
                raise FileNotFoundError(
                    "EmoNet TensorRT engine not found.\n"
                    f"Expected: {expected}\n"
                    "Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)."
                )

            engine_path = str(resolved_path)

        # Initialize base
        super().__init__(engine_path)

        self.input_size = input_size

        # Resolve bindings
        self.input_name = self._find_binding_name("input")
        self.output_names = self._find_output_names()
        self.output_shapes = {
            name: self.get_binding_shape(name) for name in self.output_names
        }
        logger.info(
            f"EmoNet bindings -> input '{self.input_name}' shape="
            f"{self.get_binding_shape(self.input_name)}, outputs={self.output_shapes}"
        )

        # Allocate CUDA buffers
        self._allocate_buffers()

        logger.info("TRTEmoNetClassifier (native API) initialized successfully")

        # Diagnostics / warnings
        self._debug_logged = False
        self._valence_warned = False
        self._arousal_warned = False

    def _find_binding_name(self, prefix: str) -> str:
        """Find first binding name containing the prefix (case-insensitive)."""
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if prefix.lower() in name.lower():
                return name
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                continue
            return self.engine.get_binding_name(i)
        raise ValueError(f"No input binding found with prefix '{prefix}'")

    def _find_output_names(self) -> List[str]:
        """Return all output binding names."""
        outputs: List[str] = []
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                outputs.append(self.engine.get_binding_name(i))
        return outputs

    def _allocate_buffers(self) -> None:
        """Allocate CUDA buffers for all bindings."""
        bindings: Dict[str, tuple] = {}

        input_shape = self.get_binding_shape(self.input_name)
        bindings[self.input_name] = input_shape

        for name in self.output_names:
            output_shape = self.get_binding_shape(name)
            bindings[name] = output_shape

        self.allocate_buffers(bindings)
        logger.debug(
            f"Allocated buffers for input '{self.input_name}' and {len(self.output_names)} outputs"
        )

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR face image to NCHW float32 tensor (1, 3, input_size, input_size).
        """
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # add batch dim
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

    def _postprocess(self, outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Postprocess TensorRT outputs:
        - expression logits -> softmax -> emotion/confidence/probabilities
        - valence/arousal from named outputs or size==1 outputs (order)
        """

        result: Dict[str, float] = {}

        # ---------------- Expression logits ----------------
        expression_logits = None
        expr_source = None

        # Prefer outputs that look like expression logits (last dim 8)
        for name, data in outputs.items():
            if data.shape[-1] == 8:
                expression_logits = data.flatten()
                expr_source = name
                break

        if expression_logits is not None:
            if expression_logits.size > 0:
                # Stable softmax
                logits_shift = expression_logits - np.max(expression_logits)
                exp_logits = np.exp(logits_shift)
                probs = exp_logits / np.sum(exp_logits)
            else:
                probs = expression_logits

            emotion_idx = int(np.argmax(probs)) if probs.size > 0 else 0
            confidence = float(probs[emotion_idx]) if probs.size > 0 else 0.0

            result["emotion"] = self.EMOTION_LABELS[emotion_idx]
            result["confidence"] = confidence
            result["probabilities"] = {
                label: float(prob) for label, prob in zip(self.EMOTION_LABELS, probs)
            }

            if not self._debug_logged:
                logger.info(
                    f"[EmoNet Debug] Using output '{expr_source}' as emotion logits, "
                    f"argmax={emotion_idx}, max_val={confidence:.4f}"
                )
                self._debug_logged = True

        # ---------------- Valence / Arousal ----------------
        for name, data in outputs.items():
            name_lower = name.lower()
            if "valence" in name_lower:
                result["valence"] = float(data.flatten()[0])
            elif "arousal" in name_lower:
                result["arousal"] = float(data.flatten()[0])

        # If not found by name, use size==1 outputs in order
        if "valence" not in result or "arousal" not in result:
            single_values = []
            for _, data in outputs.items():
                arr = np.asarray(data).flatten()
                if arr.size == 1:
                    single_values.append(float(arr[0]))

            if "valence" not in result and single_values:
                result["valence"] = single_values[0]
            if "arousal" not in result and len(single_values) > 1:
                result["arousal"] = single_values[1]

            if "valence" not in result and not self._valence_warned:
                logger.warning("TensorRT outputs missing valence; defaulting to 0.0")
                self._valence_warned = True
                result["valence"] = 0.0
            if "arousal" not in result and not self._arousal_warned:
                logger.warning("TensorRT outputs missing arousal; defaulting to 0.0")
                self._arousal_warned = True
                result["arousal"] = 0.0

        return result

    def predict_single(self, face_img: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Run emotion recognition on a single face image (BGR).
        """
        if face_img is None or face_img.size == 0:
            logger.warning("Empty face image provided")
            return None

        try:
            input_data = self._preprocess(face_img)
            outputs = self.execute(
                inputs={self.input_name: input_data},
                output_names=self.output_names,
            )
            return self._postprocess(outputs)
        except Exception as e:
            logger.error(f"Emotion recognition failed: {e}")
            return None

    def predict_batch(self, face_batch: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Run emotion recognition on a list of face images.
        """
        if not face_batch:
            return []

        results: List[Dict[str, float]] = []
        for face_img in face_batch:
            result = self.predict_single(face_img)
            if result:
                results.append(result)

        return results


__all__ = ["TRTEmoNetClassifier"]
