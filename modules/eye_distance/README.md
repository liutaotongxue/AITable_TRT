# Eye Distance 模块

## 概述
Eye Distance模块是整个AITable系统的核心集成模块，负责协调和管理所有子系统的运行。该模块整合了相机管理、人脸检测、距离测量、情绪识别和疲劳检测等功能，提供了一个统一的眼距监测系统接口。

## 核心组件

### EyeDistanceSystem (`eye_distance_system.py`)
**功能**：眼距监测系统的主控制器，集成所有功能模块并协调其运行。

**主要职责**：
- 系统初始化和资源管理
- 模块间的数据流协调
- 实时处理pipeline管理
- 性能监控和优化
- 异常处理和容错

## 系统架构

### 模块集成关系
```
EyeDistanceSystem
    ├── TOFCameraManager      # 相机管理
    │   ├── 图像采集
    │   └── 深度数据
    ├── SimpleFaceDetector    # 人脸检测
    │   └── YOLO模型
    ├── DistanceProcessor     # 距离处理
    │   └── 3D计算
    ├── EmoNetClassifier      # 情绪识别
    │   └── 深度学习
    ├── FatigueDetector       # 疲劳检测
    │   └── PERCLOS算法
    └── EnhancedVisualizer    # 可视化
        └── UI渲染
```

### 数据流程
```
相机采集 → RGB/深度数据 → 人脸检测 → 特征提取
                                   ↓
可视化输出 ← 结果整合 ← 并行处理 → 距离计算
                                   ↓
                              情绪识别/疲劳检测
```

## 功能特性

### 1. 智能初始化
**自适应配置**：
- 自动检测硬件能力
- 动态加载可用模块
- 优雅的降级策略
- 资源优化分配

**初始化参数**：
```python
system = EyeDistanceSystem(
    camera_manager=camera,      # TOF相机管理器
    plane_model=(0,1,0,0.25),  # 平面方程参数
    model_path='yolov8n.pt',   # 检测模型路径
    depth_range=(200, 1500),   # 深度范围(mm)
    enable_emotion=True,        # 启用情绪识别
    enable_fatigue=True         # 启用疲劳检测
)
```

### 2. 相机参数管理
**SDK内参获取**：
- 自动从相机SDK读取内参
- RGB和深度相机分离管理
- 实时校准支持

**默认参数fallback**：
```python
# 相机不可用时的默认参数
camera_params_rgb = {
    'fx': 500, 'fy': 500,
    'cx': 320, 'cy': 240
}
```

### 3. 平面模型
**平面方程**：`ax + by + cz + d = 0`
- 用于计算点到平面距离
- 支持自定义工作平面
- 提高测量稳定性

### 4. 多功能处理
**并行处理能力**：
- 人脸检测
- 距离测量
- 情绪识别
- 疲劳检测
- 数据记录

**模块化设计**：
- 各功能独立开关
- 按需加载资源
- 灵活的配置选项

## 核心方法

### 系统初始化
```python
def __init__(self, camera_manager, plane_model, 
             model_path, depth_range, 
             enable_emotion, enable_fatigue)
```
初始化系统，配置所有子模块。

### 帧处理
```python
def process_frame(self, rgb_frame, depth_frame=None)
```
处理单帧数据，返回综合检测结果。

**返回数据结构**：
```python
{
    'detection': {          # 人脸检测结果
        'bbox': dict,
        'left_eye': tuple,
        'right_eye': tuple,
        'confidence': float
    },
    'distance': {           # 距离测量结果
        'raw_distance': float,
        'stable_distance': float,
        'depth_available': bool
    },
    'emotion': {            # 情绪识别结果
        'emotion': str,
        'valence': float,
        'arousal': float
    },
    'fatigue': {            # 疲劳检测结果
        'ear_avg': float,
        'perclos': float,
        'fatigue_level': str,
        'alarm': bool
    },
    'performance': {        # 性能统计
        'fps': float,
        'processing_time': float
    }
}
```

### 相机参数初始化
```python
def init_camera_params_from_sdk(self)
```
从SDK获取相机内参并初始化。

```python
def init_default_camera_params(self)
```
使用默认参数初始化（相机不可用时）。

### 3D坐标计算
```python
def compute_3d_coordinates(self, x, y, depth_value, camera_params)
```
将2D像素坐标转换为3D世界坐标。

**计算公式**：
```python
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
Z = depth_value
```

### 距离计算
```python
def calculate_distance_to_plane(self, point_3d)
```
计算3D点到平面的距离。

## 性能优化

### 1. 缓存机制
- 模型预加载
- 结果缓存
- 帧缓冲管理

### 2. 并行处理
- 多线程图像处理
- 异步I/O操作
- GPU加速推理

### 3. 资源管理
- 动态内存分配
- 自动垃圾回收
- 智能资源调度

### 4. 性能监控
```python
processing_times = deque(maxlen=30)  # 最近30帧的处理时间
frame_count = 0                      # 总帧数统计
```

## 使用示例

### 基础使用
```python
from modules.eye_distance import EyeDistanceSystem
from modules.camera import TOFCameraManager

# 初始化相机
with TOFCameraManager() as camera:
    # 创建系统
    system = EyeDistanceSystem(
        camera_manager=camera,
        enable_emotion=True,
        enable_fatigue=True
    )
    
    # 处理视频流
    while True:
        frame_data = camera.fetch_frame()
        if frame_data:
            rgb, depth = extract_images(frame_data)
            results = system.process_frame(rgb, depth)
            
            # 使用结果
            if results['distance']['stable_distance']:
                print(f"距离: {results['distance']['stable_distance']:.2f}m")
```

### 高级配置
```python
# 自定义平面模型
plane_model = (0, 0, 1, -0.5)  # z = 0.5m平面

# 自定义深度范围
depth_range = (300, 2000)  # 30cm - 2m

# 创建定制系统
system = EyeDistanceSystem(
    camera_manager=camera,
    plane_model=plane_model,
    depth_range=depth_range,
    model_path='custom_model.pt',
    enable_emotion=False,  # 禁用情绪识别
    enable_fatigue=True
)
```

### 错误处理
```python
try:
    results = system.process_frame(rgb, depth)
except CameraException as e:
    logger.error(f"相机错误: {e}")
    # 降级到纯RGB模式
    results = system.process_frame(rgb, None)
except Exception as e:
    logger.error(f"处理错误: {e}")
    # 返回默认结果
    results = system.get_default_results()
```

## 配置选项

### 系统配置
```python
# 性能模式
PERFORMANCE_MODE = 'balanced'  # 'low', 'balanced', 'high'

# 处理优先级
PRIORITY_ORDER = ['detection', 'distance', 'fatigue', 'emotion']

# 超时设置
PROCESSING_TIMEOUT = 100  # ms
```

### 功能开关
```python
# 功能启用控制
ENABLE_FACE_DETECTION = True
ENABLE_DISTANCE_MEASUREMENT = True
ENABLE_EMOTION_RECOGNITION = True
ENABLE_FATIGUE_DETECTION = True
ENABLE_VISUALIZATION = True
```

## 扩展接口

### 添加新模块
```python
class CustomModule:
    def process(self, data):
        # 自定义处理
        return results

# 注册到系统
system.register_module('custom', CustomModule())
```

### 自定义处理pipeline
```python
def custom_pipeline(self, rgb, depth):
    # 自定义处理流程
    detection = self.face_detector.detect(rgb)
    if detection:
        # 自定义逻辑
        pass
    return results

# 替换默认pipeline
system.process_frame = custom_pipeline
```

## 故障诊断

### 常见问题

1. **系统初始化失败**
   - 检查相机连接
   - 验证模型文件
   - 查看日志错误

2. **处理速度慢**
   - 检查GPU可用性
   - 减少启用的功能
   - 优化图像分辨率

3. **内存泄漏**
   - 检查资源释放
   - 监控内存使用
   - 重启系统

### 性能调优
1. 降低图像分辨率
2. 减少处理频率
3. 使用更轻量的模型
4. 优化算法参数

## 最佳实践

### 1. 资源管理
- 使用上下文管理器
- 及时释放资源
- 避免内存泄漏

### 2. 错误处理
- 完善的异常捕获
- 优雅的降级策略
- 详细的日志记录

### 3. 性能优化
- 按需加载模块
- 合理的缓存策略
- 避免重复计算

## 依赖关系
- **内部模块**：
  - camera: 相机管理
  - detection: 人脸检测
  - emotion: 情绪识别
  - fatigue: 疲劳检测
  - visualization: 可视化
  - core: 基础设施
  
- **外部依赖**：
  - numpy: 数值计算
  - opencv-python: 图像处理
  - torch: 深度学习