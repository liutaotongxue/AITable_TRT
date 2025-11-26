# 模型转换指南 - TensorRT 引擎生成

本文档说明如何将 PyTorch 模型转换为 TensorRT 引擎，用于 Jetson Orin Nano 部署。

## 概述

**转换流程**: PyTorch (`.pt/.pth`) → ONNX (`.onnx`) → TensorRT (`.engine`)

**工具要求**:
- PyTorch (用于导出 ONNX)
- trtexec (JetPack 自带)
- onnx (验证 ONNX 模型)

---

## 1. EmoNet 情绪识别模型

### 步骤 1: PyTorch → ONNX

```python
# export_emonet_to_onnx.py
import torch
import torch.onnx
from modules.emotion.emonet_classifier import EmoNet

# 加载 PyTorch 模型
model = EmoNet(n_expression=8)
checkpoint = torch.load('models/emonet_8.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 创建 dummy 输入
dummy_input = torch.randn(1, 3, 256, 256)

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    'models/emonet.onnx',
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['expression', 'valence', 'arousal'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'expression': {0: 'batch_size'},
        'valence': {0: 'batch_size'},
        'arousal': {0: 'batch_size'}
    }
)

print("EmoNet exported to ONNX successfully")
```

### 步骤 2: ONNX → TensorRT (FP16)

```bash
# 在 Jetson 上执行
trtexec --onnx=models/emonet.onnx \
        --saveEngine=models/emonet_fp16.engine \
        --fp16 \
        --workspace=2048 \
        --minShapes=input:1x3x256x256 \
        --optShapes=input:1x3x256x256 \
        --maxShapes=input:1x3x256x256 \
        --verbose

# 可选：FP32 版本（更高精度，更慢）
trtexec --onnx=models/emonet.onnx \
        --saveEngine=models/emonet_fp32.engine \
        --workspace=2048 \
        --minShapes=input:1x3x256x256 \
        --optShapes=input:1x3x256x256 \
        --maxShapes=input:1x3x256x256
```

### 步骤 3: 验证引擎

```bash
# 测试加载引擎
trtexec --loadEngine=models/emonet_fp16.engine --dumpProfile

# 预期输出
# Input: input, shape=(1, 3, 256, 256)
# Output: expression, shape=(1, 8)
# Output: valence, shape=(1, 1)
# Output: arousal, shape=(1, 1)
```

---

## 2. FaceMesh 疲劳检测模型

### 步骤 1: 获取 MediaPipe FaceMesh ONNX

```python
# export_facemesh_to_onnx.py
# MediaPipe 提供了预训练的 FaceMesh 模型
# 从 MediaPipe GitHub 下载或从 .tflite 转换

# 如果有 .tflite 模型
import tensorflow as tf

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='facemesh.tflite')
interpreter.allocate_tensors()

# 转换为 TensorFlow SavedModel
# 然后转换为 ONNX (使用 tf2onnx)
!python -m tf2onnx.convert \
    --tflite facemesh.tflite \
    --output facemesh.onnx \
    --opset 12
```

### 步骤 2: ONNX → TensorRT (FP16)

```bash
# 在 Jetson 上执行
trtexec --onnx=models/facemesh.onnx \
        --saveEngine=models/facemesh_fp16.engine \
        --fp16 \
        --workspace=2048 \
        --minShapes=input:1x3x192x192 \
        --optShapes=input:1x3x192x192 \
        --maxShapes=input:1x3x192x192 \
        --verbose

# 可选：FP32 版本
trtexec --onnx=models/facemesh.onnx \
        --saveEngine=models/facemesh_fp32.engine \
        --workspace=2048
```

### 步骤 3: 验证引擎

```bash
# 测试加载引擎
trtexec --loadEngine=models/facemesh_fp16.engine --dumpProfile

# 预期输出
# Input: input, shape=(1, 3, 192, 192)
# Output: landmarks, shape=(1, 468, 3)  # 468 个关键点，每个 (x, y, z)
```

---

## 3. YOLO 人脸检测模型（参考）

### 步骤 1: PyTorch → ONNX

```python
# export_yolo_to_onnx.py
from ultralytics import YOLO

model = YOLO('models/yolov8n-face.pt')
model.export(format='onnx', simplify=True)
```

### 步骤 2: ONNX → TensorRT

```bash
trtexec --onnx=yolov8n-face.onnx \
        --saveEngine=yolov8n-face.engine \
        --fp16 \
        --workspace=4096 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640
```

---

## 常见问题

### Q1: trtexec 找不到？

**A**: trtexec 由 JetPack 提供，通常在 `/usr/src/tensorrt/bin/trtexec`

```bash
# 添加到 PATH
export PATH=$PATH:/usr/src/tensorrt/bin

# 或使用完整路径
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine
```

### Q2: ONNX 导出失败？

**A**: 检查 PyTorch 和 ONNX 版本兼容性

```bash
pip3 install onnx onnxruntime

# 验证 ONNX 模型
python3 -c "import onnx; onnx.checker.check_model('model.onnx')"
```

### Q3: TensorRT 引擎在其他设备无法使用？

**A**: TensorRT 引擎是**平台特定**的，必须在目标设备上生成。

- Jetson Orin Nano 生成的引擎 ≠ x86 GPU 生成的引擎
- 不同 CUDA/TensorRT 版本生成的引擎不兼容

### Q4: FP16 vs FP32 如何选择？

**A**:
- **FP16**: 推荐（更快，内存更小，Jetson 优化）
- **FP32**: 更高精度，但速度慢 2-3 倍

```bash
# 性能对比（Jetson Orin Nano）
# FP16: ~5ms/frame
# FP32: ~15ms/frame
```

### Q5: 如何验证引擎输入输出？

**A**: 使用 `trtexec --dumpProfile`

```bash
trtexec --loadEngine=model.engine --dumpProfile

# 输出示例
# [I] Input: input, shape=(1, 3, 256, 256), dtype=Float
# [I] Output: output, shape=(1, 8), dtype=Float
```

---

## 性能优化建议

### 1. 动态 Shape 优化

```bash
# 固定 shape（最快）
--minShapes=input:1x3x256x256 \
--optShapes=input:1x3x256x256 \
--maxShapes=input:1x3x256x256

# 动态 batch（灵活但稍慢）
--minShapes=input:1x3x256x256 \
--optShapes=input:4x3x256x256 \
--maxShapes=input:8x3x256x256
```

### 2. INT8 量化（实验性）

```bash
# 需要校准数据集
trtexec --onnx=model.onnx \
        --saveEngine=model_int8.engine \
        --int8 \
        --calib=calibration_data.txt
```

### 3. 工作空间大小

```bash
# 默认 1024MB，可根据 GPU 内存调整
--workspace=2048  # 2GB（推荐 Jetson Orin）
--workspace=4096  # 4GB（如果内存充足）
```

---

## 自动化转换脚本

```bash
# tools/convert_all_models.sh
#!/bin/bash
set -e

echo "Converting EmoNet..."
python3 tools/export_emonet_to_onnx.py
trtexec --onnx=models/emonet.onnx --saveEngine=models/emonet_fp16.engine --fp16 --workspace=2048

echo "Converting FaceMesh..."
python3 tools/export_facemesh_to_onnx.py
trtexec --onnx=models/facemesh.onnx --saveEngine=models/facemesh_fp16.engine --fp16 --workspace=2048

echo "All models converted successfully!"
```

---

## 相关文档

- [TensorRT-Only 架构说明](./TENSORRT_ONLY_ARCHITECTURE.md)
- [Jetson 依赖清单](../requirements-jetson.txt)
- [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/)

---

**最后更新**: 2025-01-19
