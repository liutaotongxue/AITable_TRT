# 疲劳检测模块 TensorRT-Only 迁移完成报告

**日期**: 2025-01-19
**模块**: Fatigue Detection (疲劳检测)
**架构变更**: MediaPipe 可选回退 → TensorRT-Only 必需

---

## 概述

疲劳检测模块已完成从"可选模块（MediaPipe 回退）"到"TensorRT-Only 必需模块"的彻底迁移。

---

## 变更总结

### ✅ 删除的文件
- `modules/fatigue/fatigue_detector.py` - MediaPipe 版本已删除

### ✅ 修改的文件

| 文件 | 变更内容 |
|------|---------|
| `modules/fatigue/__init__.py` | 移除 fallback 机制，直接导入 TensorRT 版本 |
| `modules/fatigue/trt_fatigue_detector.py` | 从 system_config.json 读取模型路径 |
| `system_config.json` | facemesh 模块设为 `required: true` |
| `main_gui.py` | 疲劳检测失败时阻止启动（raise） |
| `preflight_check.py` | 添加 facemesh .engine 文件强制检查 |
| `README.md` | 更新说明，移除 MediaPipe 引用 |

---

## 详细变更

### 1. `modules/fatigue/__init__.py`

**之前**:
```python
# 优先尝试 TensorRT 版本
try:
    from .trt_fatigue_detector import TRTFatigueDetector as FatigueDetector
except ImportError:
    # 回退到 MediaPipe 版本
    from .fatigue_detector import FatigueDetector
```

**现在**:
```python
"""
疲劳检测模块（TensorRT-Only）
仅支持 TensorRT 版本，需要 TensorRT 引擎文件和 PyCUDA
如果 TensorRT 不可用，系统将拒绝启动
"""
from .trt_fatigue_detector import TRTFatigueDetector as FatigueDetector
from .tensorrt_facemesh import TensorRTFaceMesh, create_facemesh

__all__ = ['FatigueDetector', 'TensorRTFaceMesh', 'create_facemesh']
_backend = 'tensorrt'
```

---

### 2. `modules/fatigue/trt_fatigue_detector.py`

**新增功能**:
- 从 `ConfigLoader` 读取 `system_config.json` 中的 facemesh 模型路径
- 自动尝试 primary → fallback 路径
- 文件不存在时抛出 `FileNotFoundError`，给出明确错误信息

**关键代码**:
```python
def __init__(self, model_path: Optional[str] = None, perclos_window=30, fps=30):
    # 从配置文件获取模型路径（如果未提供）
    if model_path is None:
        config = ConfigLoader.get_instance()
        facemesh_config = config.get("models", {}).get("facemesh", {})

        # 尝试 primary 路径
        model_path = facemesh_config.get("primary")
        if model_path and not Path(model_path).exists():
            # 尝试 fallback 路径
            for fallback in facemesh_config.get("fallback", []):
                if Path(fallback).exists():
                    model_path = fallback
                    break

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"FaceMesh TensorRT engine not found.\n"
                f"Expected: {facemesh_config.get('primary')}\n"
                f"Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)"
            )
```

---

### 3. `system_config.json`

**变更**:
```json
"facemesh": {
  "primary": "models/facemesh_fp16.engine",
  "fallback": ["models/facemesh_fp32.engine"],
  "required": true,  // ← 从 false 改为 true
  "description": "FaceMesh 模型（TensorRT-Only，用于疲劳检测）",
  "note": "必需模块，系统启动前必须提供 TensorRT 引擎文件。详见 docs/MODEL_CONVERSION_GUIDE.md"
}
```

---

### 4. `main_gui.py`

**之前**:
```python
# 初始化疲劳检测器（可选模块，需要 MediaPipe）
fatigue_detector = None
try:
    from modules.fatigue import FatigueDetector
    fatigue_detector = FatigueDetector(perclos_window=30, fps=30)
    logger.info("✓ 疲劳检测模块初始化成功")
except ImportError as e:
    logger.warning(f"⚠ 疲劳检测模块不可用: {e}")
    logger.warning("  提示：疲劳检测需要 MediaPipe，TensorRT-only 模式下可跳过")
except Exception as e:
    logger.error(f"✗ 疲劳检测模块初始化失败: {e}")
```

**现在**:
```python
# 初始化疲劳检测器（TensorRT-Only，必需模块）
fatigue_detector = None
try:
    from modules.fatigue import FatigueDetector
    fatigue_detector = FatigueDetector(perclos_window=30, fps=30)
    logger.info("✓ 疲劳检测模块初始化成功（TensorRT FaceMesh）")
except FileNotFoundError as e:
    logger.error(f"✗ 疲劳检测模块初始化失败：缺少 FaceMesh 引擎文件")
    logger.error(f"  详细信息: {e}")
    logger.error("  请运行模型转换：参见 docs/MODEL_CONVERSION_GUIDE.md")
    raise  # 阻止启动
except ImportError as e:
    logger.error(f"✗ 疲劳检测模块不可用（TensorRT 依赖缺失）: {e}")
    logger.error("  TensorRT-only 模式下必需 TensorRT 和 PyCUDA")
    raise  # 阻止启动
except Exception as e:
    logger.error(f"✗ 疲劳检测模块初始化失败: {e}")
    raise  # 阻止启动
```

---

### 5. `preflight_check.py`

**新增检查**:
```python
# 疲劳检测 FaceMesh：TensorRT-Only，必需 .engine 文件
elif name == "facemesh":
    if not str(resolved).endswith('.engine'):
        self.report.add_error(
            f"{description} 格式错误: {resolved}\n"
            f"      TensorRT-only 架构要求：FaceMesh 必须使用 .engine 文件\n"
            f"      疲劳检测模块为必需模块\n"
            f"      请参考 docs/MODEL_CONVERSION_GUIDE.md 转换模型"
        )
    else:
        self.report.add_pass(f"{description} [TensorRT Engine] => {resolved}")
```

---

### 6. `README.md`

**功能特性更新**:
```markdown
- **疲劳检测**: 基于 TensorRT FaceMesh 的眼部姿态分析（✅ **TensorRT-Only**，必需模块）

> **TensorRT-Only 架构**: 疲劳检测模块现已完全迁移到 TensorRT-only 模式。
> - ✅ **疲劳检测**: TensorRT-Only（必需 .engine 文件，无 MediaPipe 回退）
> - ✅ **情绪识别**: 优先 TensorRT，可回退 PyTorch（可选）
```

**依赖安装更新**:
```bash
# 移除了
# pip3 install mediapipe
```

---

## 启动流程变更

### 之前（可选模块）
```
1. preflight_check.py
   └─ facemesh: 可选，不检查

2. main_gui.py
   ├─ 尝试加载 FatigueDetector
   ├─ 失败 → 警告，继续运行
   └─ 系统启动成功（无疲劳检测）
```

### 现在（必需模块）
```
1. preflight_check.py
   ├─ facemesh: required=true
   ├─ 检查 .engine 文件存在
   └─ 不存在 → 错误，阻止启动

2. main_gui.py
   ├─ 加载 FatigueDetector
   ├─ FileNotFoundError → 错误，退出
   ├─ ImportError → 错误，退出
   └─ 成功 → 系统启动
```

---

## 错误处理

### 缺少引擎文件
```
错误: FileNotFoundError
消息: FaceMesh TensorRT engine not found.
      Expected: models/facemesh_fp16.engine
      Please run model conversion (see docs/MODEL_CONVERSION_GUIDE.md)
结果: 程序退出，阻止启动
```

### TensorRT 未安装
```
错误: ImportError
消息: TensorRT FaceMesh not available - required for TensorRT-Only architecture
      Ensure tensorrt and pycuda are installed.
结果: 程序退出，阻止启动
```

---

## 部署要求

### 必需文件
```
models/
├── facemesh_fp16.engine  ← 必需（primary）
└── facemesh_fp32.engine  ← 可选（fallback）
```

### 必需依赖
```bash
tensorrt >= 8.5.0
pycuda >= 2021.1
scipy  # EAR 计算
```

### 不再需要
```bash
mediapipe  # 已移除
```

---

## 模型转换

### 从 TFLite → ONNX
```bash
python -m tf2onnx.convert \
    --tflite facemesh.tflite \
    --output facemesh.onnx \
    --opset 12
```

### 从 ONNX → TensorRT（在 Jetson 上执行）
```bash
trtexec --onnx=models/facemesh.onnx \
        --saveEngine=models/facemesh_fp16.engine \
        --fp16 \
        --workspace=2048 \
        --minShapes=input:1x3x192x192 \
        --optShapes=input:1x3x192x192 \
        --maxShapes=input:1x3x192x192
```

---

## 验证清单

### ✅ 代码变更
- [x] 删除 fatigue_detector.py
- [x] 简化 __init__.py（移除 fallback）
- [x] trt_fatigue_detector.py 读取配置
- [x] main_gui.py 失败时阻止启动
- [x] preflight_check.py 检查 .engine
- [x] system_config.json 设为 required
- [x] README.md 移除 MediaPipe 说明

### 🔄 待测试（需 Jetson 硬件）
- [ ] 转换 facemesh 模型到 TensorRT
- [ ] 验证 .engine 加载成功
- [ ] 验证 EAR/PERCLOS 计算准确性
- [ ] 性能基准测试（vs MediaPipe）
- [ ] 长时间运行稳定性

---

## 性能预期

| 指标 | MediaPipe | TensorRT FP16 | 加速比 |
|------|-----------|---------------|--------|
| FaceMesh 推理 | ~15ms | ~3ms | 5x |
| EAR 计算 | ~0.5ms | ~0.5ms | 1x |
| PERCLOS 计算 | ~0.2ms | ~0.2ms | 1x |
| **总延迟** | ~16ms | ~4ms | **4x** |

---

## 架构优势

### ✅ 优点
1. **性能提升**: 5x 推理加速
2. **依赖精简**: 移除 MediaPipe（~200MB）
3. **启动保障**: 必需模块缺失时立即报错
4. **维护简化**: 单一后端，无 fallback 逻辑
5. **错误明确**: 清晰的错误消息和修复指引

### ⚠️ 注意事项
1. **必需模型**: 启动前必须转换 facemesh 模型
2. **平台特定**: .engine 文件必须在目标 Jetson 上生成
3. **版本锁定**: TensorRT 版本必须一致

---

## 相关文档

- [模型转换指南](./MODEL_CONVERSION_GUIDE.md)
- [TensorRT 迁移完成报告](./TENSORRT_MIGRATION_COMPLETE.md)
- [系统配置](../system_config.json)

---

**最后更新**: 2025-01-19
**架构版本**: TensorRT-only v2.0
**状态**: 代码完成 ✅ | 硬件测试待进行 🔄
