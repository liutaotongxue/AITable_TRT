# AITable - 智能视觉分析系统

基于海康TOF相机的实时视觉分析系统，提供人脸检测、情绪识别、疲劳检测和距离测量等功能。

## 功能特性

- 🎯 **人脸检测**: 实时人脸检测与跟踪
- 😊 **情绪识别**: 基于深度学习的表情分析
- 😴 **疲劳检测**: 眼部和头部姿态分析
- 📏 **距离测量**: 基于TOF深度相机的精确测距
- 🖼️ **实时可视化**: 交互式GUI界面

## 项目结构

```
AITable_final/
├── camera/                     # 相机SDK相关文件
│   ├── Mv3dRgbd.dll           # TOF相机主DLL
│   ├── Mv3dRgbdSDK.dll        # SDK支持库
│   └── Mv3dRgbdImport/        # Python接口封装
│       ├── __init__.py
│       ├── Mv3dRgbdApi.py     # API封装
│       └── Mv3dRgbdDefine.py  # 常量定义
│
├── modules/                    # 功能模块
│   ├── core/                  # 核心功能
│   │   ├── __init__.py
│   │   ├── constants.py      # 系统常量
│   │   └── logger.py         # 日志系统
│   │
│   ├── camera/                # 相机模块
│   │   ├── __init__.py
│   │   ├── tof_manager.py    # TOF相机管理
│   │   ├── intrinsics.py     # 内参管理
│   │   └── image_processor.py # 图像处理
│   │
│   ├── detection/             # 检测模块
│   │   ├── __init__.py
│   │   ├── face_detector.py  # 人脸检测
│   │   └── distance_processor.py # 距离处理
│   │
│   ├── emotion/               # 情绪识别
│   │   ├── __init__.py
│   │   ├── config.py         # 配置参数
│   │   └── emonet_classifier.py # 分类器
│   │
│   ├── fatigue/               # 疲劳检测
│   │   ├── __init__.py
│   │   └── fatigue_detector.py
│   │
│   ├── eye_distance/          # 眼距测量
│   │   ├── __init__.py
│   │   └── eye_distance_system.py
│   │
│   └── visualization/         # 可视化
│       ├── __init__.py
│       └── visualizer.py
│
├── models/                    # 模型文件目录
│   └── (模型文件放置于此)
│
├── logs/                      # 日志目录
│
├── config.py                  # 全局配置文件
├── utils.py                   # 工具函数
├── main_gui.py               # 主程序GUI
├── CAMERA_SDK_GUIDE.md       # 相机SDK使用指南
├── README.md                 # 项目说明文档
└── .gitignore                # Git忽略配置
```

## 环境要求

### 系统要求
- Windows 10/11 (64位)
- Python 3.7-3.10
- Visual C++ Redistributable 2015-2022

### Python依赖
```bash
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0
scipy>=1.5.0
torch>=1.9.0 (可选，用于深度学习模型)
```

## 安装指南

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/AITable_final.git
cd AITable_final
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装TOF相机SDK
- 从[海康官网](https://www.hikrobotics.com/)下载TOF SDK
- 安装到默认路径: `C:\Program Files (x86)\Common Files\Mv3dRgbdSDK\`
- 复制DLL文件到项目camera目录

### 4. 下载模型文件
将所需的模型文件放置到`models/`目录

## 使用方法

### 启动主程序
```bash
python main_gui.py
```

### 基本操作
1. 连接相机：程序启动时自动连接TOF相机
2. 选择功能：在GUI界面选择需要的分析功能
3. 实时分析：系统将实时显示分析结果
4. 数据记录：分析结果自动保存到日志

## API使用示例

```python
from utils import get_calibration_data
from modules.camera.tof_manager import TOFCameraManager

# 初始化相机
with TOFCameraManager() as camera:
    # 获取标定参数
    params = get_calibration_data(camera.camera)
    
    # 获取帧数据
    frame = camera.fetch_frame()
    
    # 处理数据...
```

## 配置说明

主要配置文件：`config.py`

```python
class CameraSettings:
    RGB_RESOLUTION = (1280, 1024)
    DEPTH_RESOLUTION = (1280, 1024)
    MIN_VALID_DEPTH = 200  # mm
    MAX_VALID_DEPTH = 1500  # mm

class FilterSettings:
    TEMPORAL_HISTORY_SIZE = 10
    SPATIAL_KERNEL_SIZE = 5
```

## 故障排除

### DLL加载失败
- 确保已安装Visual C++ Runtime
- 检查DLL文件是否在camera目录
- 验证Python版本和系统架构匹配

### 相机连接失败
- 检查USB连接
- 安装相机驱动
- 使用设备管理器确认设备识别

详细故障排除请参考[CAMERA_SDK_GUIDE.md](CAMERA_SDK_GUIDE.md)

## 开发指南

### 添加新功能模块
1. 在`modules/`下创建新目录
2. 实现模块接口
3. 在主程序中集成


