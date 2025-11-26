# TensorRT-Only 架构说明

## 架构决策

**决策日期**: 2025-01-19
**架构类型**: TensorRT-only（单一推理后端）
**部署平台**: NVIDIA Jetson Orin Nano Super

## 概述

本项目已从**双后端架构**（TensorRT + Ultralytics）简化为 **TensorRT-only 架构**，移除所有 Ultralytics/PyTorch 推理路径。

### 架构变更对比

| 方面 | 旧架构（双后端） | 新架构（TensorRT-only） |
|------|-----------------|----------------------|
| **人脸检测器** | SimpleFaceDetector + TRTFaceDetector | 仅 TRTFaceDetector |
| **支持格式** | `.pt` (PyTorch) + `.engine` (TensorRT) | 仅 `.engine` (TensorRT) |
| **依赖** | ultralytics, torch, tensorrt, pycuda | 仅 tensorrt, pycuda |
| **后端选择** | 配置参数 `backend: auto/tensorrt/ultralytics` | 无需配置，强制 TensorRT |
| **代码复杂度** | 高（条件分支、回退逻辑） | 低（单一路径） |
| **维护负担** | 高（需同时维护两套检测器） | 低（只维护一套） |

---

## 变更详情

### 1. 已删除文件

```bash
modules/detection/face_detector.py  # SimpleFaceDetector（Ultralytics 后端）
```

### 2. 已修改文件

#### **modules/detection/__init__.py**
```python
# 旧版本（双后端）
from .face_detector import SimpleFaceDetector
try:
    from .trt_face_detector import TRTFaceDetector
    __all__ = ['SimpleFaceDetector', 'DistanceProcessor', 'TRTFaceDetector']
except ImportError:
    __all__ = ['SimpleFaceDetector', 'DistanceProcessor']

# 新版本（TensorRT-only）
from .trt_face_detector import TRTFaceDetector
from .distance_processor import DistanceProcessor
__all__ = ['TRTFaceDetector', 'DistanceProcessor']
```

#### **modules/eye_distance/eye_distance_system.py**
- **移除**: `backend` 参数（构造函数）
- **移除**: `_create_face_detector()` 中的后端选择逻辑
- **简化**: 直接使用 `TRTFaceDetector`，验证 `.engine` 文件格式
- **移除**: `SimpleFaceDetector` 导入

#### **main_gui.py**
- **移除**: `backend` 配置读取
- **移除**: `backend` 参数传递给 `EyeDistanceSystem`
- **简化**: 日志输出（不再显示 backend 信息）

#### **system_config.json**
```json
// 旧版本
{
  "yolo_face": {
    "backend": "auto",
    "backend_options": { "auto": "...", "tensorrt": "...", "ultralytics": "..." }
  }
}

// 新版本
{
  "yolo_face": {
    "primary": "models/yolov8n-face.engine",
    "description": "人脸检测模型（仅支持 TensorRT .engine 文件）"
  }
}
```

#### **preflight_check.py**
- **增强**: 强制检查 `yolo_face` 模型必须是 `.engine` 格式
- **错误提示**: 如果模型不是 `.engine`，抛出清晰错误信息

### 3. 新增文件

```
requirements-jetson.txt  # Jetson TensorRT 专用依赖清单
docs/TENSORRT_ONLY_ARCHITECTURE.md  # 本文档
```

---

## 模型要求

### 支持的模型格式

- **TensorRT 引擎文件** (`.engine`)
  - 必须是板级工具生成（trtexec、onnx2trt 等）
  - 平台特定（Jetson Orin Nano 生成的引擎无法在其他平台使用）
  - 示例：`yolov8n-face.engine`

### 不支持的模型格式

- **PyTorch 权重** (`.pt`, `.pth`) - 推理不使用 PyTorch
- **ONNX 模型** (`.onnx`) - 需先转换为 `.engine`
- **Ultralytics 导出的 .engine** - SimpleFaceDetector 已删除

### 模型转换示例

```bash
# ONNX → TensorRT (FP16)
trtexec --onnx=yolov8n-face.onnx \
        --saveEngine=yolov8n-face.engine \
        --fp16 \
        --workspace=4096

# 验证引擎
trtexec --loadEngine=yolov8n-face.engine --dumpProfile
```

---

## 依赖管理

### 核心依赖（必需）

| 包 | 用途 | 来源 |
|----|------|------|
| `tensorrt` | TensorRT 推理引擎 | JetPack SDK（预装） |
| `pycuda` | CUDA Python 绑定 | `pip install pycuda` |
| `numpy` | 数值计算 | `pip install numpy` |
| `opencv-python` | 图像处理 | JetPack（预装）或 pip |

### 已移除依赖

| 包 | 旧用途 | 移除原因 |
|----|--------|---------|
| `ultralytics` | SimpleFaceDetector | SimpleFaceDetector 已删除 |
| `torch` | PyTorch 推理 | 推理不使用 PyTorch（preflight_check 仍需要） |
| `torchvision` | PyTorch 视觉工具 | 不再需要 |

### 安装指南

```bash
# 1. 确认 JetPack 已安装
sudo apt-cache show nvidia-jetpack

# 2. 安装 Python 依赖
pip3 install -r requirements-jetson.txt

# 3. 验证 TensorRT
python3 -c "import tensorrt; import pycuda.driver; print('TensorRT ready')"
```

---

## 错误处理

### 运行时错误场景

#### 场景 1: 模型格式错误
```python
# 错误：使用 .pt 文件
model_path = "models/yolov8n-face.pt"

# 结果：
ValueError: TensorRT-only 架构要求: model_path 必须是 .engine 文件
当前路径: models/yolov8n-face.pt
不支持 .pt (PyTorch) 或其他格式
```

#### 场景 2: TensorRT 未安装
```bash
# preflight_check 输出：
[ERROR] 未安装 TensorRT（tensorrt）
        商业版本要求 TensorRT 必须安装。请安装 tensorrt 包。
```

#### 场景 3: 引擎平台不匹配
```python
# 错误：在 Jetson 上加载 x86 引擎
RuntimeError: Failed to initialize TRTFaceDetector:
  [TensorRT] ERROR: engine built for incompatible architecture
```

### 预检验证

所有错误在 `preflight_check()` 中提前捕获：

```bash
$ python3 preflight_check.py

[PASS] 系统: Linux 5.10.104-tegra (aarch64)
[PASS] Python 3.8
[PASS] PyTorch 2.0.0
[PASS] CUDA 可用: NVIDIA Orin Nano (8.0 GB, CC 8.7, CUDA 11.4)
[PASS] TensorRT 8.5.2
[PASS] PyCUDA driver 初始化成功
[PASS] 人脸检测模型（仅支持 TensorRT .engine 文件） [TensorRT Engine] => models/yolov8n-face.engine
[PASS] TOFCameraManager 正确实现 DepthCameraInterface 接口

All checks passed
```

---

## 代码示例

### 初始化人脸检测器

```python
from modules.detection import TRTFaceDetector

# 直接加载 TensorRT 引擎（无需指定 backend）
detector = TRTFaceDetector(
    engine_path="models/yolov8n-face.engine",
    confidence_threshold=0.5
)

# 推理
detections = detector.detect(rgb_frame)
```

### 集成到 EyeDistanceSystem

```python
from modules.eye_distance import EyeDistanceSystem

# 简化的构造函数（无 backend 参数）
system = EyeDistanceSystem(
    camera=camera,
    model_path="models/yolov8n-face.engine",  # 必须是 .engine
    depth_range=(200, 1500)
)
```

---

## 迁移指南

### 从旧架构迁移

如果你有旧版本代码使用 `backend` 参数：

```python
# 旧代码（双后端）
system = EyeDistanceSystem(
    camera=camera,
    model_path="models/yolov8n-face.engine",
    backend="tensorrt"  # 不再支持
)

# 新代码（TensorRT-only）
system = EyeDistanceSystem(
    camera=camera,
    model_path="models/yolov8n-face.engine"  # 自动使用 TensorRT
)
```

### 配置文件迁移

```json
// 旧配置
{
  "models": {
    "yolo_face": {
      "primary": "models/yolov8n-face.engine",
      "backend": "auto"  // 删除此字段
    }
  }
}

// 新配置
{
  "models": {
    "yolo_face": {
      "primary": "models/yolov8n-face.engine"  // 仅需路径
    }
  }
}
```

---

## 性能优势

### 代码简化

| 指标 | 旧架构 | 新架构 | 改进 |
|------|--------|--------|------|
| **LOC（face_detector.py）** | 228 | 0（已删除） | -228 行 |
| **LOC（EyeDistanceSystem）** | 186 | 140 | -46 行 |
| **条件分支** | 7 个（backend 选择） | 0 | 消除分支 |
| **导入依赖** | 5 个模块 | 2 个模块 | -60% |

### 维护优势

- **单一责任**: 只维护 TRTFaceDetector
- **无回退逻辑**: 消除"尝试 TRT 失败回退 Ultralytics"的复杂性
- **清晰错误**: 预检阶段就明确报错，不在运行时切换后端
- **依赖精简**: 减少 Ultralytics/PyTorch 安装和版本冲突

---

## 常见问题 (FAQ)

### Q1: 为什么移除 Ultralytics 支持？

**A**: Ultralytics 的 `.engine` 支持仅限于其自己导出的引擎，无法加载板级工具（trtexec）生成的引擎。在 Jetson 部署中，板级引擎是标准做法，保留 Ultralytics 分支只会增加复杂度而无实际价值。

### Q2: PyTorch 是否完全移除？

**A**: 推理时不使用 PyTorch（已移除 torch 导入），但 `preflight_check.py` 仍需要 torch 来验证 CUDA 环境。如果完全不需要 preflight_check，可以移除 torch 依赖。

### Q3: 如何添加新的检测器（如姿态检测）？

**A**: 遵循相同模式：
1. 创建 `TRTPoseDetector`（仅支持 `.engine`）
2. 在 `preflight_check.py` 中验证模型格式
3. 不引入 PyTorch/Ultralytics 后端

### Q4: 旧的 .pt 模型怎么办？

**A**: 需要转换：
```bash
# 1. PyTorch → ONNX
python export_to_onnx.py --weights yolov8n-face.pt --output yolov8n-face.onnx

# 2. ONNX → TensorRT
trtexec --onnx=yolov8n-face.onnx --saveEngine=yolov8n-face.engine --fp16
```

### Q5: 如何验证架构是否正确？

**A**: 运行预检：
```bash
python3 preflight_check.py

# 应看到：
# [PASS] 人脸检测模型（仅支持 TensorRT .engine 文件） [TensorRT Engine] => ...
```

---

## 相关文档

- [ARCHITECTURE_DECISION_ORCHESTRATOR.md](./ARCHITECTURE_DECISION_ORCHESTRATOR.md) - 同步/异步架构决策
- [requirements-jetson.txt](../requirements-jetson.txt) - Jetson 专用依赖
- [TENSORRT_ONLY_EMOTION.md](../TENSORRT_ONLY_EMOTION.md) - 情绪识别 TensorRT 集成（如有）

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2025-01-19 | 初始版本：TensorRT-only 架构重构 |

---

**维护者**: AI Table Team
**最后更新**: 2025-01-19
