# Visualization 模块

## 概述
Visualization模块负责系统的所有可视化输出，包括实时检测结果显示、状态面板绘制、历史数据图表和用户界面渲染。该模块提供了丰富的视觉反馈，让用户能够直观地了解系统的运行状态和检测结果。

## 核心组件

### EnhancedVisualizer (`visualizer.py`)
**功能**：增强可视化器，提供全面的可视化功能和优雅的界面设计。

**主要特性**：
- 实时检测结果标注
- 动态状态面板
- 历史趋势图表
- 多语言支持（中英文）
- 自适应布局
- 主题定制

## 可视化功能

### 1. 人脸检测可视化
**边框绘制**：
- 检测成功：绿色边框
- 仅RGB模式：橙色边框
- 置信度显示

**关键点标注**：
- 左眼标记（L）：蓝色圆点
- 右眼标记（R）：绿色圆点
- 眼睛轮廓连线

**效果示例**：
```
┌─────────────────┐
│     [Face]      │  <- 绿色边框
│   L●     ●R     │  <- 眼睛标记
│                 │
│                 │  
└─────────────────┘
```

### 2. 距离显示
**主距离显示**：
- 大字体醒目显示
- 单位自动转换（m/cm）
- 颜色编码状态

**距离状态指示**：
```python
# 颜色方案
optimal_range = (0.4, 0.6)  # 最佳距离：绿色
warning_range = (0.3, 0.7)  # 警告距离：黄色
danger_range = other         # 危险距离：红色
```

**显示格式**：
```
Distance: 0.52m ✓ (Optimal)
Too Close: < 0.3m ⚠
Too Far: > 0.7m ⚠
```

### 3. 状态面板
**信息面板布局**：
```
╔══════════════════════════════════════╗
║ Model: YOLOv8n | FPS: 30.2          ║
║──────────────────────────────────────║
║ Distance: 0.52m (Stable)             ║
║ Emotion: Happy (V:0.8, A:0.6)        ║
║ Fatigue: Normal (PERCLOS: 15%)       ║
║──────────────────────────────────────║


### 5. 情绪可视化
**情绪显示**：
```python
emotion_colors = {
    'happy': (0, 255, 0),      # 绿色
    'sad': (255, 0, 0),        # 蓝色
    'angry': (0, 0, 255),      # 红色
    'neutral': (128, 128, 128) # 灰色
}
```

**情绪仪表盘**：
```
Emotion: Happy 
Valence:  +0.8
Arousal:  +0.6
```

### 6. 疲劳状态显示
**PERCLOS进度条**：
```
PERCLOS: 15.3% 
         Normal  Mild    Severe
          <20%   20-40%   >40%
```

**EAR实时显示**：
```
EAR: 0.285  OPEN
EAR: 0.125  CLOSED
```

**疲劳警告**：
- 轻度疲劳：橙色提示
- 重度疲劳：红色闪烁边框


## 视觉设计

### 配色方案
```python
# 主题颜色
THEME_COLORS = {
    'primary': (66, 165, 245),    # 主色调
    'success': (76, 175, 80),     # 成功
    'warning': (255, 152, 0),     # 警告
    'danger': (244, 67, 54),      # 危险
    'info': (0, 188, 212),        # 信息
    'dark': (33, 33, 33),         # 深色背景
    'light': (250, 250, 250)      # 浅色背景
}
```

### 字体设置
```python
FONTS = {
    'title': cv2.FONT_HERSHEY_DUPLEX,
    'normal': cv2.FONT_HERSHEY_SIMPLEX,
    'small': cv2.FONT_HERSHEY_PLAIN
}

FONT_SIZES = {
    'large': 1.2,
    'medium': 0.8,
    'small': 0.6
}
```

### 布局规范
- 边距：10px
- 行间距：1.5倍
- 面板圆角：5px
- 阴影效果：可选

## 核心方法

### 主绘制方法
```python
def draw_visualization(self, image, results, model_info)
```
绘制完整的可视化界面。

**参数**：
- `image`: 输入图像
- `results`: 检测结果字典
- `model_info`: 模型信息字符串

### 专项绘制方法

#### 人脸标注
```python
def draw_face_detection(self, image, detection)
```
绘制人脸检测框和关键点。

#### 距离显示
```python
def draw_main_distance(self, image, distance)
```
在图像顶部显示主要距离信息。

#### 状态面板
```python
def draw_status_panel(self, image, results)
```
绘制包含所有状态信息的面板。

#### 历史图表
```python
def draw_history_graph(self, width, height)
```
绘制距离历史趋势图。

#### 警告提示
```python
def draw_warning(self, image, message, severity)
```
显示警告或错误信息。

### 辅助方法

#### 颜色管理
```python
def get_distance_color(self, distance)
```
根据距离值返回对应的颜色。

#### 文本渲染
```python
def draw_text_with_background(self, image, text, position, color)
```
绘制带背景的文本，提高可读性。

#### 进度条
```python
def draw_progress_bar(self, image, value, max_value, position, size)
```
绘制进度条组件。

## 数据管理

### 历史数据缓存
```python
class VisualizationData:
    def __init__(self):
        self.distance_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        self.emotion_history = deque(maxlen=50)
        self.fatigue_history = deque(maxlen=50)
```

### 更新机制
- 实时数据推送
- 异步更新渲染
- 帧率自适应

## 性能优化

### 渲染优化
1. **脏区域更新**：只更新变化的区域
2. **缓存机制**：缓存静态元素
3. **批量绘制**：减少绘制调用
4. **硬件加速**：使用GPU渲染

### 内存管理
- 限制历史数据长度
- 及时释放临时图像
- 使用对象池技术

### 响应性能
- 目标帧率：30 FPS
- 渲染延迟：<16ms
- 更新频率：自适应

## 使用示例

### 基础使用
```python
from modules.visualization import EnhancedVisualizer

# 创建可视化器
visualizer = EnhancedVisualizer()

# 绘制检测结果
results = {
    'detection': face_detection,
    'distance': distance_data,
    'emotion': emotion_result,
    'fatigue': fatigue_status
}

# 生成可视化图像
visualized_image = visualizer.draw_visualization(
    image, results, "YOLOv8n-Face"
)

# 显示结果
cv2.imshow("AITable Monitor", visualized_image)
```

### 自定义主题
```python
# 设置自定义颜色
visualizer.set_theme({
    'primary': (100, 200, 100),
    'warning': (200, 200, 0)
})

# 设置语言
visualizer.set_language('zh_CN')  # 中文
visualizer.set_language('en_US')  # 英文
```

### 导出功能
```python
# 保存截图
visualizer.save_screenshot("detection_result.png")

# 导出历史数据
data = visualizer.export_history_data()
save_to_csv(data, "history.csv")
```

## 界面定制

### 布局配置
```python
LAYOUT_CONFIG = {
    'show_distance': True,
    'show_emotion': True,
    'show_fatigue': True,
    'show_history': True,
    'panel_position': 'bottom',  # 'top', 'bottom', 'left', 'right'
    'panel_opacity': 0.8
}
```

### 显示选项
```python
DISPLAY_OPTIONS = {
    'show_fps': True,
    'show_confidence': True,
    'show_coordinates': False,
    'show_grid': False,
    'antialiasing': True
}
```

## 多语言支持

### 语言包
```python
LANGUAGES = {
    'en_US': {
        'distance': 'Distance',
        'emotion': 'Emotion',
        'fatigue': 'Fatigue',
        'warning': 'Warning'
    },
    'zh_CN': {
        'distance': '距离',
        'emotion': '情绪',
        'fatigue': '疲劳度',
        'warning': '警告'
    }
}
```

## 扩展功能

### 插件系统
```python
class CustomVisualizationPlugin:
    def draw(self, image, data):
        # 自定义绘制逻辑
        return modified_image

# 注册插件
visualizer.register_plugin('custom', CustomVisualizationPlugin())
```

### 动画效果
- 淡入淡出过渡
- 平滑曲线动画
- 闪烁警告效果
- 数值变化动画

## 最佳实践

### 1. 视觉层次
- 重要信息突出显示
- 次要信息淡化处理
- 合理的信息密度

### 2. 颜色使用
- 遵循色彩心理学
- 考虑色盲用户
- 保持一致性

### 3. 响应式设计
- 自适应不同分辨率
- 动态布局调整
- 可缩放界面元素

## 故障排除

### 常见问题

1. **显示异常**
   - 检查OpenCV版本
   - 验证图像格式
   - 确认显示设备

2. **性能问题**
   - 降低渲染质量
   - 减少视觉效果
   - 优化更新频率

3. **文字乱码**
   - 安装中文字体
   - 设置正确编码
   - 使用PIL渲染中文

## 依赖关系
- **图形库**：
  - opencv-python: 基础绘图
  - numpy: 数组操作
  - matplotlib: 图表生成（可选）
  
- **其他模块**：
  - core: 常量和配置
  - 所有检测模块: 数据源