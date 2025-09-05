# Core 模块

## 概述
Core模块提供系统的核心功能和基础设施，包括常量定义、日志系统和公共工具。这是整个AITable系统的基础层，为其他模块提供统一的配置和工具支持。

## 核心组件

### 1. Constants (`constants.py`)
**功能**：系统全局常量和配置参数的集中管理。

**主要常量类别**：

#### 检测参数
- `FACE_CONFIDENCE_THRESHOLD`: 人脸检测置信度阈值（默认0.5）
- `MIN_DETECTION_SIZE`: 最小检测尺寸
- `MAX_DETECTION_DISTANCE`: 最大检测距离

#### 距离测量参数
- `OPTIMAL_DISTANCE_RANGE`: 最佳观看距离范围 (0.4m - 0.6m)
- `WARNING_DISTANCE_THRESHOLD`: 距离警告阈值
- `SMOOTHING_WINDOW`: 距离平滑窗口大小

#### 性能参数
- `DEFAULT_FPS`: 默认帧率（30 FPS）
- `BUFFER_SIZE`: 缓冲区大小
- `TIMEOUT_MS`: 操作超时时间（毫秒）

#### 显示参数
- `HISTORY_DISPLAY_LENGTH`: 历史数据显示长度（100个数据点）
- `UI_UPDATE_INTERVAL`: UI更新间隔
- `VISUALIZATION_QUALITY`: 可视化质量设置

### 2. Logger (`logger.py`)
**功能**：统一的日志记录系统，提供分级日志和格式化输出。

**特性**：
- 多级日志支持（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 自动时间戳记录
- 文件和控制台双重输出
- 日志轮转机制
- 彩色控制台输出（可选）

**日志格式**：
```
[时间戳] [级别] [模块名] 消息内容
```

**使用示例**：
```python
from modules.core.logger import logger

logger.info("系统初始化完成")
logger.warning("检测到异常情况")
logger.error("处理失败", exc_info=True)
```

**日志配置**：
- 日志文件位置：`logs/aitable.log`
- 日志级别：可通过环境变量配置
- 文件大小限制：10MB（自动轮转）
- 保留文件数：5个

### 3. 工具函数
**功能**：提供通用的辅助功能。

**主要工具**：
- 时间戳管理
- 路径处理
- 配置加载
- 性能计时器
- 资源监控

## 设计原则

### 单一职责
每个组件只负责一个明确的功能领域：
- Constants：配置管理
- Logger：日志记录
- Tools：辅助功能

### 零依赖
Core模块不依赖其他业务模块，确保基础设施的独立性。

### 易于扩展
- 常量通过类属性定义，便于继承和覆盖
- 日志系统支持自定义格式器和处理器
- 工具函数采用模块化设计

## 配置管理

### 环境变量支持
支持通过环境变量覆盖默认配置：
```bash
# 设置日志级别
export AITABLE_LOG_LEVEL=DEBUG

# 设置性能模式
export AITABLE_PERFORMANCE_MODE=HIGH
```

### 配置文件
支持从配置文件加载参数：
```python
# config.yaml
detection:
  confidence_threshold: 0.6
  min_size: 50

distance:
  optimal_range: [0.3, 0.7]
  warning_threshold: 0.2
```

## 性能优化

### 日志性能
- 异步日志写入（可选）
- 批量刷新机制
- 条件日志（仅在特定级别启用）

### 内存管理
- 常量使用类属性，避免重复实例化
- 日志缓冲区自动管理
- 资源自动清理

## 错误处理

### 日志系统容错
- 日志文件写入失败时自动切换到控制台
- 磁盘空间不足时自动清理旧日志
- 权限问题自动处理

### 配置验证
- 自动验证配置参数范围
- 提供默认值fallback机制
- 配置错误详细提示

## 最佳实践

### 日志使用建议
1. **选择合适的日志级别**：
   - DEBUG：详细的调试信息
   - INFO：一般信息和里程碑
   - WARNING：潜在问题但不影响运行
   - ERROR：错误但可恢复
   - CRITICAL：严重错误需要立即处理

2. **包含上下文信息**：
```python
logger.info(f"处理帧 {frame_id}，检测到 {face_count} 个人脸")
```

3. **错误日志包含堆栈**：
```python
try:
    process_data()
except Exception as e:
    logger.error("数据处理失败", exc_info=True)
```

### 常量使用建议
1. 使用语义化命名
2. 提供单位说明（如_MS, _MM）
3. 分组相关常量
4. 添加注释说明用途

## 扩展点

### 添加新常量
```python
class Constants:
    # 现有常量...
    
    # 新功能常量组
    NEW_FEATURE_ENABLED = True
    NEW_FEATURE_THRESHOLD = 0.8
```

### 自定义日志格式
```python
import logging

custom_formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger.handlers[0].setFormatter(custom_formatter)
```

### 添加性能监控
```python
from modules.core.tools import Timer

with Timer("复杂操作"):
    # 执行操作
    result = complex_operation()
```

## 依赖关系
- **标准库**：
  - logging: 日志功能
  - os, sys: 系统交互
  - datetime: 时间处理
  - pathlib: 路径操作

## 维护指南

### 日志维护
- 定期检查日志文件大小
- 清理过期日志文件
- 监控日志级别分布

### 性能监控
- 跟踪日志写入延迟
- 监控内存使用
- 优化热点代码路径

### 版本管理
- 常量变更需要版本记录
- 保持向后兼容性
- 提供迁移指南