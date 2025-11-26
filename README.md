# AITable - 智能桌面监测系统

> **TensorRT-only 架构** | Jetson Orin Nano Super 专用

基于 NVIDIA Jetson 和 TensorRT 的实时视觉分析系统，提供高性能人脸检测、情绪识别、疲劳检测和眼距测量等功能。

## 功能特性

- **人脸检测**: TensorRT 加速的实时人脸检测（**TensorRT-Only**，必需模块）
- **情绪识别**: EmoNet 深度学习模型（**TensorRT-Only**，必需模块）
- **疲劳检测**: 基于 TensorRT FaceMesh 的眼部姿态分析（**TensorRT-Only**，必需模块）
- **姿态检测**: YOLO Pose 人体关键点检测（**TensorRT-Only**，必需模块）
- **眼距测量**: 基于 TOF 深度相机的精确测距
- **实时可视化**: OpenCV 交互式 GUI 界面
- **相机软重启**: 自动检测异常并恢复相机功能，支持指数退避策略

> **完全 TensorRT-Only 架构**: 所有推理模块现已完全迁移到 TensorRT-only 模式。
> - **人脸检测**: TensorRT-Only（必需 .engine 文件）
> - **情绪识别**: TensorRT-Only（必需 .engine 文件）
> - **疲劳检测**: TensorRT-Only（必需 .engine 文件）
> - **姿态检测**: TensorRT-Only（必需 .engine 文件）
> - **无任何 PyTorch/MediaPipe/Ultralytics 依赖**（推理时）
> - 详见 [模型转换指南](docs/MODEL_CONVERSION_GUIDE.md) 和 [TensorRT-Only 架构说明](docs/TENSORRT_ONLY_ARCHITECTURE.md)

## 架构特性

### TensorRT-Only 架构
- **单一推理后端**: 只使用 TensorRT（移除 Ultralytics/PyTorch 推理）
- **板级引擎**: 支持任意 TensorRT 引擎（trtexec、onnx2trt 等）
- **依赖精简**: 无需安装 ultralytics 或 torch（推理时）
- **预检验证**: 启动前强制检查 TensorRT/PyCUDA 和模型格式

### 接口驱动设计
- **DepthCameraInterface**: 抽象相机接口（支持多种 TOF 相机）
- **TOFSDKLoader**: 动态加载 SDK（支持不同路径配置）
- **上下文管理**: 自动资源清理（CUDA、相机、文件句柄）

### 双模式推理架构

系统支持两种推理模式，通过 `system_config.json` 配置：

#### **同步推理模式**（默认，推荐）
```json
"async_inference": { "enabled": false }
```
**特点**：
- 主循环直接执行推理（串行）
- 稳定可靠，无线程安全问题
- 适合当前 PyCUDA 架构
- 资源管理简单，易于调试

**架构**：
```
main_gui.py → EyeDistanceSystem → TRT 检测器
            ↓
         OpenCV 主循环（同步推理）
```

#### **异步推理模式**（实验性）
```json
"async_inference": { "enabled": true }
```
**特点**：
- 后台线程推理，主循环不阻塞
- 提升帧率和响应速度
- 需要修复 PyCUDA 线程安全问题
- 当前不可用（CUDA context 限制）

**架构**：
```
main_gui.py → DetectionOrchestrator → Engine 层 → InferenceScheduler
                                       ↓
                                  AsyncEngineRunner (后台线程)
                                       ↓
                                  TRT 检测器（需要独立 CUDA context）
```

**已知限制**：
- 当前 TRT 检测器使用 `pycuda.autoinit`，仅创建主线程 CUDA context
- Worker 线程无法访问主线程的 CUDA context，导致推理失败
- 需要实现以下解决方案之一：
  1. 使用 `pycuda.driver.Context.attach()` 在每个线程创建 context
  2. 替换为 TensorRT 原生 Python API（推荐）
  3. 使用 execution context 的 `enqueue_v2`（线程安全）

**参见**：
- [system_config.json](system_config.json) - 异步推理配置
- [modules/core/orchestrator.py](modules/core/orchestrator.py) - 协调器实现
- [modules/engines/](modules/engines/) - Engine 适配层

## 项目结构

```
AITable_jerorin_TRT/
├── modules/                   # 主功能模块
│   ├── __init__.py
│   │
│   ├── core/                  # 核心功能
│   │   ├── constants.py       # 系统常量
│   │   ├── logger.py          # 日志系统（读取 system_config.json）
│   │   ├── config_loader.py   # 配置加载器（单例）
│   │   └── hardware_context.py # 硬件上下文
│   │
│   ├── camera/                # 相机模块
│   │   ├── depth_camera_interface.py  # 抽象接口
│   │   ├── tof_manager.py     # TOF 相机管理器
│   │   └── ...
│   │
│   ├── detection/             # 检测模块（TensorRT-only）
│   │   ├── trt_face_detector.py    # TensorRT 人脸检测器
│   │   └── ...
│   │
│   ├── emotion/               # 情绪识别（TensorRT）
│   │   └── trt_emonet_classifier.py    # EmoNet TensorRT 推理
│   │
│   ├── fatigue/               # 疲劳检测（TensorRT-Only）
│   │   ├── trt_fatigue_detector.py # TensorRT 疲劳检测器
│   │   └── tensorrt_facemesh.py    # TensorRT FaceMesh 引擎
│   │
│   └── visualization/         # 可视化
│       └── visualizer.py      # OpenCV 可视化
│
├── models/                    # TensorRT 引擎目录
│   ├── yolov8n-face.engine    # 人脸检测引擎（板级生成）
│   └── emonet_fp16.engine     # 情绪识别引擎
│
├── docs/                      # 文档
│   ├── TENSORRT_ONLY_ARCHITECTURE.md   # 架构说明
│   └── ARCHITECTURE_DECISION_ORCHESTRATOR.md  # 架构决策
│
├── system_config.json         # 系统配置文件
├── preflight_check.py         # 启动前预检
├── main_gui.py                # 主程序入口
├── requirements-jetson.txt    # Jetson 专用依赖
└── README.md                  # 本文档
```

### 架构说明

**入口点调用链**（脚本模式）：
```
用户运行: python3 main_gui.py
    ↓
  main_gui.py: sys.path 添加 modules/ 目录
    ↓
  导入 modules.core、modules.camera 等
    ↓
  运行主程序逻辑（GUI、检测循环）
```

## 快速开始

### 系统要求

| 组件 | 要求 |
|------|------|
| **平台** | NVIDIA Jetson Orin Nano Super |
| **操作系统** | Ubuntu 20.04 (JetPack 5.x) |
| **Python** | 3.8+ |
| **JetPack** | 5.0+ (含 CUDA 11.4, cuDNN 8.6, TensorRT 8.5) |
| **TOF 相机** | Vzense DCAM710 或兼容设备 |

### 1. 安装 JetPack SDK

```bash
# 验证 JetPack 安装
sudo apt-cache show nvidia-jetpack

# 验证 TensorRT
python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
```

### 2. 克隆项目

```bash
git clone https://github.com/yourusername/AITable_jerorin_TRT.git
cd AITable_jerorin_TRT
```

### 3. 安装依赖

```bash
# 安装 Jetson 专用依赖
pip3 install -r requirements-jetson.txt

# 验证依赖
python3 -c "import pycuda.driver; import tensorrt; print('Dependencies OK')"
```

### 4. 配置 TOF 相机 SDK

编辑 `system_config.json`，配置 SDK 路径：

```json
{
  "paths": {
    "sdk_python_path": "/home/jetorin/Downloads/Res/demo/python",
    "sdk_lib_path_aarch64": "/opt/Mv3dRgbdSDK/lib/aarch64"
  }
}
```

### 5. 准备 TensorRT 模型

```bash
# 方法 1: 使用预生成的引擎（推荐）
cp /path/to/yolov8n-face.engine models/
cp /path/to/emonet_fp16.engine models/

# 方法 2: 从 ONNX 转换（需要 trtexec）
trtexec --onnx=yolov8n-face.onnx \
        --saveEngine=models/yolov8n-face.engine \
        --fp16 \
        --workspace=4096
```

### 6. 运行预检

```bash
python3 preflight_check.py
```

**预期输出**:
```
[PASS] 系统: Linux 5.10.104-tegra (aarch64)
[PASS] Python 3.8
[PASS] CUDA 可用: NVIDIA Orin Nano (8.0 GB)
[PASS] TensorRT 8.5.2
[PASS] PyCUDA driver 初始化成功
[PASS] 人脸检测模型 [TensorRT Engine] => models/yolov8n-face.engine
[PASS] TOFCameraManager 正确实现 DepthCameraInterface 接口

All checks passed
```

### 7. 启动主程序

```bash
python3 main_gui.py
```

## 使用说明

### 主程序操作

| 按键 | 功能 |
|------|------|
| `q` | 退出系统 |
| `s` | 保存当前帧截图 |
| `r` | 重置系统状态 |
| `c` | 相机软重启（自动恢复） |
| `Space` | 暂停/继续 |
| `f` | 全屏模式 |
| `w` | 窗口模式 |
| `+` / `=` | 放大窗口 (110%) |
| `-` | 缩小窗口 (90%) |

### systemd 服务部署

在 Jetson 上部署为 systemd 服务（推荐生产环境）：

### 1. 部署项目文件

```bash
# 将项目部署到 /opt/AITable
sudo cp -r AITable_jerorin_TRT /opt/AITable
cd /opt/AITable

# 确保依赖已安装
pip3 install -r requirements-jetson.txt
```

### 2. 创建 systemd 服务文件

创建 `/etc/systemd/system/aitable.service`:

```ini
[Unit]
Description=AITable Intelligent Monitoring System
After=network.target

[Service]
Type=simple
User=jetorin
Group=jetorin
WorkingDirectory=/opt/AITable
ExecStart=/usr/bin/python3 /opt/AITable/main_gui.py
Restart=on-failure
RestartSec=10

# 环境变量配置
Environment="AITABLE_LOG_LEVEL=INFO"
Environment="PYTHONUNBUFFERED=1"

# 资源限制
MemoryLimit=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

### 3. 启用并启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable aitable
sudo systemctl start aitable

# 查看状态
sudo systemctl status aitable

# 查看日志
sudo journalctl -u aitable -f
```

### 优势

- **自动启动**：系统启动时自动运行
- **崩溃恢复**：进程异常退出自动重启
- **资源管理**：限制内存和 CPU 使用
- **日志集成**：systemd 统一日志管理
- **简单部署**：无需包安装，直接运行脚本

## 配置文件说明

**system_config.json** - 主配置文件

```json
{
  "system": {
    "platform": "jetson_orin_nano_super"
  },
  "models": {
    "yolo_face": {
      "primary": "models/yolov8n-face.engine",
      "required": true,
      "description": "人脸检测模型（仅支持 TensorRT .engine 文件）"
    }
  },
  "hardware": {
    "tof_camera": {
      "enabled": true,
      "required": true
    }
  },
  "logging": {
    "level": "INFO",
    "enable_console": true,
    "enable_file": true,
    "file_rotation": "daily",
    "max_size_mb": 100
  },
  "paths": {
    "logs_dir": "logs",
    "models_dir": "models",
    "screenshots_dir": "screenshots"
  }
}
```

## 模型管理

### 支持的模型格式

| 格式 | 支持 | 说明 |
|------|------|------|
| `.engine` (TensorRT) | 是 | 板级生成引擎（trtexec、onnx2trt） |
| `.pt` (PyTorch) | 否 | TensorRT-only 架构不支持 |
| `.onnx` | 否 | 需先转换为 `.engine` |

### 模型转换示例

#### YOLO 人脸检测模型

```bash
# 从 PyTorch 导出 ONNX
python export_yolo_to_onnx.py --weights yolov8n-face.pt --output yolov8n-face.onnx

# ONNX → TensorRT (FP16)
trtexec --onnx=yolov8n-face.onnx \
        --saveEngine=yolov8n-face.engine \
        --fp16 \
        --workspace=4096 \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x640x640

# 验证引擎
trtexec --loadEngine=yolov8n-face.engine --dumpProfile
```

#### EmoNet 情绪识别模型

```bash
# PyTorch → ONNX → TensorRT
trtexec --onnx=emonet.onnx \
        --saveEngine=emonet_fp16.engine \
        --fp16 \
        --workspace=2048
```

## 故障排除

### 常见错误

#### 1. TensorRT 未安装
```
[ERROR] 未安装 TensorRT（tensorrt）
```
**解决**: 安装 JetPack SDK
```bash
sudo apt install nvidia-jetpack
```

#### 2. 模型格式错误
```
ValueError: TensorRT-only 架构要求: model_path 必须是 .engine 文件
```
**解决**: 将 `.pt` 或 `.onnx` 模型转换为 `.engine`

#### 3. TOF 相机连接失败
```
RuntimeError: TOF camera SDK not available
```
**解决**: 检查 SDK 路径配置
```bash
# 验证 SDK 路径
ls /opt/Mv3dRgbdSDK/lib/aarch64/libMv3dRgbd.so
```

#### 4. CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 减小模型输入尺寸或使用 INT8 量化

### 日志查看

```bash
# 查看实时日志
tail -f logs/aitable_YYYYMMDD.log

# 查看预检日志
python3 preflight_check.py 2>&1 | tee preflight.log
```

## 相关文档

- [TensorRT-Only 架构说明](docs/TENSORRT_ONLY_ARCHITECTURE.md) - 架构决策和迁移指南
- [架构决策记录](docs/ARCHITECTURE_DECISION_ORCHESTRATOR.md) - 同步 vs 异步架构
- [相机软重启功能](docs/CAMERA_SOFT_RESTART.md) - TOF相机自动恢复机制
- [Jetson 依赖清单](requirements-jetson.txt) - 完整依赖列表

## 贡献指南

### 代码规范

- 使用 TensorRT 推理（不使用 PyTorch/Ultralytics）
- 所有检测器必须实现 `.detect_face()` 接口
- 使用上下文管理器管理资源（`with` 语句）
- 模型文件必须是 `.engine` 格式

### 提交前检查

```bash
# 1. 运行预检
python3 preflight_check.py

# 2. 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 3. 验证代码风格
# (可选) black modules/ tests/
```

## 许可证

[MIT License](LICENSE)

## 致谢

- **NVIDIA TensorRT** - 高性能推理引擎
- **Ultralytics YOLO** - 人脸检测模型训练
- **EmoNet** - 情绪识别模型
- **Vzense** - TOF 深度相机 SDK

## 联系方式

- **Issues**: [GitHub Issues](https://github.com/yourusername/AITable_jerorin_TRT/issues)
- **Email**: your.email@example.com

---

**最后更新**: 2025-01-19 | **架构版本**: TensorRT-only v1.0
