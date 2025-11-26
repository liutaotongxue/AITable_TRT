# TOF相机软重启功能说明

## 概述

TOF相机软重启功能提供了一种在相机出现异常时自动恢复的机制，无需重启整个系统即可恢复相机功能。

## 更新历史

### 2025-11-24 重要修复与优化
- 修复了状态返回优先级问题，确保软重启状态可见
- 修复了失败计数逻辑，所有失败情况都会被计入
- 添加了线程安全保护，避免并发访问冲突
- 修复了重启失败后的重试机制，系统不会卡死
- **新增指数退避策略**：频繁失败时自动延长重启间隔
- **增强线程安全**：cleanup() 方法增加锁保护
- **重启类型统计**：区分手动和自动重启次数

## 功能特性

### 自动触发条件
- **连续帧失败**: 当连续10帧获取失败时自动触发软重启
- **重启冷却时间**: 5秒内不会重复触发，避免频繁重启

### 手动触发
- **快捷键**: 按 `c` 键手动触发相机软重启
- **用途**: 用于调试或相机异常但未达到自动触发阈值时
- **优先级**: 手动重启忽略冷却时间和指数退避限制

### 指数退避策略（新增）
- **触发条件**: 连续3次自动重启失败
- **延迟计算**: 基础时间(10秒) × 2^失败次数，最大300秒
- **智能恢复**: 重启成功后自动重置退避状态
- **手动豁免**: 手动重启不受退避策略影响
- **非阻塞设计**: 使用后台Timer，不阻塞调用线程或主循环

## 状态转换

```
ok → error (检测到异常) → soft_restarting → recovering → ok/error
```

### 状态说明
- `ok`: 相机正常工作
- `error`: 相机出现错误
- `soft_restarting`: 正在执行软重启
- `recovering`: 正在恢复相机功能
- `unavailable`: SDK不可用（无法软重启）

## 软重启流程

1. **关闭相机**
   - 停止数据流
   - 关闭设备连接
   - 释放相机资源

2. **等待恢复**
   - 等待2秒让硬件重置

3. **重新初始化**
   - 重新扫描设备
   - 打开相机设备
   - 恢复数据流

## 统计信息

通过 `get_telemetry()` 方法可以获取以下信息：
- `total_soft_restarts`: 软重启总次数
- `total_manual_restarts`: 手动重启次数（新增）
- `total_auto_restarts`: 自动重启次数（新增）
- `consecutive_failures`: 当前连续失败次数
- `consecutive_restart_failures`: 连续重启失败次数（新增）
- `max_restart_failures`: 触发退避的失败阈值（新增）
- `backoff_active`: 是否处于退避状态（新增）
- `frame_age_ms`: 最后一帧的时间差（毫秒）
- `last_restart_time`: 最后一次重启时间

## 配置参数

在 `TOFCameraManager.__init__()` 中可调整：
```python
# 基础重启参数
self._max_consecutive_failures = 10  # 触发软重启的失败阈值
self._restart_cooldown = 5.0        # 重启冷却时间（秒）

# 指数退避参数（新增）
self._max_restart_failures = 3     # 触发退避的重启失败阈值
self._base_backoff_time = 10.0      # 基础退避时间（秒）
self._max_backoff_time = 300.0      # 最大退避时间（5分钟）
self._backoff_multiplier = 2.0      # 退避倍数
```

## 日志示例

### 自动软重启
```
[WARNING] Frame fetch failed: 0x80000001
[WARNING] Consecutive failures (10) exceeded threshold, triggering soft restart
[INFO] Starting soft restart #1...
[INFO] Soft restart step 1/3: Closing camera...
[INFO] Camera resources cleaned up
[INFO] Soft restart step 2/3: Reinitializing camera...
[INFO] TOF camera initialized successfully
[INFO] Soft restart step 3/3: Success! Camera reinitialized
```

### 手动软重启
```
[INFO] Manual camera soft restart triggered
[INFO] Manual soft restart requested
[INFO] Starting manual soft restart #2...
```

### 指数退避示例（新增）
```
[WARNING] Restart failure #1 (max: 3)
[WARNING] Restart failure #2 (max: 3)
[WARNING] Restart failure #3 (max: 3)
[WARNING] Too many consecutive restart failures (3), applying exponential backoff: 10.0s
[INFO] Exponential backoff completed, proceeding with restart...
[INFO] Starting automatic soft restart #6...
[WARNING] Restart failure #4 (max: 3)
[WARNING] Too many consecutive restart failures (4), applying exponential backoff: 20.0s
[INFO] Exponential backoff completed, proceeding with restart...
```

## 注意事项

1. **异步执行**: 软重启在后台线程执行，不会阻塞主程序
2. **状态监控**: 主程序会显示 "Camera recovering..." 提示
3. **失败处理**: 如果重启失败，状态变为 `error`，但会重置失败计数允许再次尝试
4. **线程安全**: 所有相机操作都使用线程锁保护，支持多线程环境（包括cleanup）
5. **失败计数**: 即使相机未初始化，fetch_frame调用仍会计入失败次数
6. **指数退避**: 自动重启频繁失败时，系统会延长重启间隔避免资源消耗
7. **重启类型**: 区分手动和自动重启，手动重启不受退避策略影响
8. **完全非阻塞**: 退避等待使用Timer后台执行，不会阻塞fetch_frame或主循环

## 故障排除

### 软重启频繁触发
- 检查USB连接是否稳定
- 确认相机供电是否充足
- 查看是否有其他程序占用相机

### 软重启失败
- 检查相机硬件连接
- 验证SDK路径配置
- 查看系统日志获取详细错误信息

### 禁用功能
如需临时禁用相关功能：
```python
# 禁用自动软重启
camera._max_consecutive_failures = 999999

# 禁用指数退避
camera._max_restart_failures = 999999

# 减少退避延迟（调试用）
camera._base_backoff_time = 1.0
camera._max_backoff_time = 5.0
```