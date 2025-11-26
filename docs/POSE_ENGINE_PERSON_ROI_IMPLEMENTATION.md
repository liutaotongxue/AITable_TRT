# PoseEngine Person ROI 支持 - 实施完成

> **实施日期**: 2025-01-21
> **目标**: 让 PoseEngine 接口支持外部提供的 person_roi，完全实现"所有模块分析同一主角"

---

## 📋 概述

本次实施完成了 **PoseEngine 外部 ROI 支持**，使姿态检测模块能够使用由 Face-Person 关联提供的 person_roi，确保情绪、疲劳、姿态三个模块分析的是**完全相同的主角**。

### 实施前后对比

#### 实施前 ❌

```
Face-Person 关联 → 主角选择 → 提取 face_roi + person_roi
    ↓
情绪模块 ← face_roi (主角A)
疲劳模块 ← face_roi (主角A)
姿态模块 ← 自己检测人体 (可能检测到背景人员B) ⚠️ 不一致!
```

**问题**: 姿态模块独立检测 person，可能检测到不同的人，导致结果不一致。

#### 实施后 ✅

```
Face-Person 关联 → 主角选择 → 提取 face_roi + person_roi
    ↓
情绪模块 ← face_roi (主角A)
疲劳模块 ← face_roi (主角A)
姿态模块 ← person_roi (主角A) ✅ 使用相同的 person!
```

**优势**: 所有模块分析同一个主角，结果一致性有保障。

---

## 🔧 修改内容

### 1. PoseEngine 接口增强

**文件**: `modules/engines/pose_engine.py`

#### 修改 1: `maybe_infer()` 方法签名

```python
def maybe_infer(
    self,
    rgb_frame,
    depth_frame,
    face_present: bool,
    global_frame_interval: float,
    frame_count: int,
    global_fps: float,
    person_roi=None,  #  新增参数
    person_bbox: Optional[Dict[str, int]] = None  #  新增参数
) -> PoseResult:
```

#### 修改 2: `_run_inference()` 方法

**新增逻辑**: 支持两种检测模式

```python
def _run_inference(
    self,
    rgb_frame,
    depth_frame,
    pose_age_ms: float,
    global_frame_interval: float,
    frame_count: int,
    global_fps: float,
    person_roi=None,  #  新增参数
    person_bbox: Optional[Dict[str, int]] = None  #  新增参数
) -> PoseResult:
```

**检测逻辑**:

```python
# 1. 检测 2D 关键点
#  支持外部提供的 person_roi（Face-Person 关联场景）
if person_roi is not None and hasattr(person_roi, 'roi_rgb'):
    # 模式 1: 使用外部提供的 person ROI（确保分析同一主角）
    detection_frame = person_roi.roi_rgb
    use_roi_mode = True
    roi_offset_x = person_bbox.get('x1', 0) if person_bbox else 0
    roi_offset_y = person_bbox.get('y1', 0) if person_bbox else 0
else:
    # 模式 2: 传统模式（在全帧上独立检测）
    detection_frame = rgb_frame
    use_roi_mode = False
    roi_offset_x = 0
    roi_offset_y = 0

raw_pose = self.pose_detector.detect_posture_with_yolo(detection_frame)
```

**坐标变换**:

```python
# 2. 提取关键点和质量标志
keypoints_2d = self.pose_detector.filter_keypoints_only(raw_pose)
quality_flags = self.pose_detector.extract_quality_flags(raw_pose)

#  坐标变换：如果使用 ROI 模式，需要将坐标转换回全帧坐标系
if use_roi_mode and (roi_offset_x != 0 or roi_offset_y != 0):
    keypoints_2d = {
        kp_name: (x + roi_offset_x, y + roi_offset_y)
        for kp_name, (x, y) in keypoints_2d.items()
    }
```

---

### 2. InferenceTask 数据结构增强

**文件**: `modules/core/async_engine.py`

```python
@dataclass
class InferenceTask:
    """推理任务封装"""
    task_id: int
    rgb_frame: Any
    depth_frame: Optional[Any]
    face_bbox: Optional[Dict]
    face_roi: Optional[Any]
    face_present: bool
    face_just_appeared: bool
    frame_count: int
    global_frame_interval: float
    global_fps: float
    submit_time: float
    person_roi: Optional[Any] = None  #  新增字段
    person_bbox: Optional[Dict] = None  #  新增字段
```

---

### 3. InferenceScheduler 调度器更新

**文件**: `modules/core/inference_scheduler.py`

#### 修改 1: `submit()` 方法签名

```python
def submit(
    self,
    rgb_frame: np.ndarray,
    depth_frame: Optional[np.ndarray],
    face_bbox: Optional[Dict],
    face_roi: Optional[Any],
    face_present: bool,
    face_just_appeared: bool,
    frame_count: int,
    global_frame_interval: float,
    global_fps: float,
    person_roi: Optional[Any] = None,  #  新增参数
    person_bbox: Optional[Dict] = None  #  新增参数
):
```

#### 修改 2: 任务创建

```python
task = InferenceTask(
    task_id=self._task_counter,
    rgb_frame=rgb_frame,
    depth_frame=depth_frame,
    face_bbox=face_bbox.copy() if face_bbox is not None else None,
    face_roi=face_roi,
    face_present=face_present,
    face_just_appeared=face_just_appeared,
    frame_count=frame_count,
    global_frame_interval=global_frame_interval,
    global_fps=global_fps,
    submit_time=submit_time,
    person_roi=person_roi,  #  传递 person ROI
    person_bbox=person_bbox.copy() if person_bbox is not None else None  #  传递 person bbox
)
```

#### 修改 3: `_infer_pose()` 方法

```python
def _infer_pose(self, task: InferenceTask):
    # ...
    #  传递 person_roi 和 person_bbox，确保分析同一主角
    result = self.pose_engine._run_inference(
        rgb_frame=rgb_frame_copy,
        depth_frame=depth_frame_copy,
        pose_age_ms=pose_age_ms,
        global_frame_interval=task.global_frame_interval,
        frame_count=task.frame_count,
        global_fps=task.global_fps,
        person_roi=task.person_roi,  #  使用主角的 person ROI
        person_bbox=task.person_bbox  #  使用主角的 person bbox
    )
```

---

### 4. DetectionOrchestrator 集成

**文件**: `modules/core/orchestrator.py`

#### 同步模式

```python
# 姿态检测（使用 PoseEngine）
#  传递 person_roi 和 person_bbox，确保分析同一主角
pose_results = None
if self.pose_engine:
    pose_result = self.pose_engine.maybe_infer(
        rgb_frame=rgb_frame,
        depth_frame=depth_frame,
        face_present=current_face_detected,
        global_frame_interval=self.global_frame_interval,
        frame_count=self.system.frame_count,
        global_fps=self.global_fps,
        person_roi=person_roi,  #  使用主角的 person ROI
        person_bbox=person_bbox  #  使用主角的 person bbox
    )
    pose_results = pose_result.data
```

#### 异步模式

```python
if self.async_enabled and self.inference_scheduler:
    # 异步模式：提交任务到调度器
    self.inference_scheduler.submit(
        rgb_frame=rgb_frame,
        depth_frame=depth_frame,
        face_bbox=face_bbox,
        face_roi=face_roi,
        face_present=current_face_detected,
        face_just_appeared=face_appeared,
        frame_count=self.system.frame_count,
        global_frame_interval=self.global_frame_interval,
        global_fps=self.global_fps,
        person_roi=person_roi,  #  传递 person ROI
        person_bbox=person_bbox  #  传递 person bbox
    )
```

---

## 🎯 实施效果

### 1. 一致性保障

**所有模块现在分析同一个主角**:

```python
# orchestrator.py 主循环
target_state = target_manager.update(face_person_pairs, depth_info)

if target_state and target_state.is_valid():
    # 提取同一主角的双 ROI
    rois = roi_manager.extract_dual(
        rgb_frame,
        face_bbox=target_state.face_bbox.to_dict(),
        person_bbox=target_state.person_bbox.to_dict() if target_state.person_bbox else None
    )
    face_roi = rois['face_roi']
    person_roi = rois['person_roi']

    # 所有模块使用相同主角的 ROI
    emotion_engine.maybe_infer(..., face_roi=face_roi)  # 主角A的脸
    fatigue_engine.infer(..., face_roi=face_roi)        # 主角A的脸
    pose_engine.maybe_infer(..., person_roi=person_roi) # 主角A的身体 ✅
```

### 2. 调试日志

启用调试模式查看 ROI 使用情况:

```bash
export AITABLE_DEBUG_POSE=1
```

**日志示例**:

```
[Pose] 使用外部 person_roi (offset_x=120, offset_y=80)
[Pose] 关键点坐标已转换到全帧坐标系 (offset: +120, +80)
```

### 3. 向后兼容

**如果不提供 person_roi**: PoseEngine 自动回退到传统模式（全帧检测）

```python
# 传统调用（仍然有效）
pose_result = pose_engine.maybe_infer(
    rgb_frame=rgb_frame,
    depth_frame=depth_frame,
    face_present=True,
    global_frame_interval=0.033,
    frame_count=100,
    global_fps=30.0
    # 不传递 person_roi/person_bbox → 自动使用全帧检测
)
```

---

## 🧪 测试建议

### 1. 单人场景测试

- **预期**: 所有模块的 `target_id` 一致
- **验证**: 检查日志中的 `[INCONSISTENT_TARGET]` 告警（应该为 0）

### 2. 多人场景测试

- **操作**: 主角前方有背景人员走过
- **预期**:
  - Face-Person 关联正确匹配主角的 face 和 person
  - 姿态检测使用主角的 person_roi，忽略背景人员
  - 背景人员深度超出 `[depth_min, depth_max]`，不会被选为主角

### 3. ROI 模式日志验证

```bash
# 启用 pose 调试日志
export AITABLE_DEBUG_POSE=1

# 运行系统
python main_gui.py

# 检查日志
grep "\[Pose\] 使用外部 person_roi" logs/aitable_*.log
```

**期望输出**:

```
[Pose] 使用外部 person_roi (offset_x=120, offset_y=80)
[Pose] 关键点坐标已转换到全帧坐标系 (offset: +120, +80)
```

### 4. 性能影响测试

**预期**: 使用 person_roi 后性能应有**轻微提升**

- **原因**: 在更小的 ROI 上检测（例如 200x400）比全帧检测（1280x720）更快
- **测试方法**: 比较姿态检测延迟（`latency_ms`）

```bash
# 查看姿态性能日志
grep "【姿态性能分析】" logs/aitable_*.log
```

---

## 🐛 已知限制

1. **ROI 边界处理**: 如果 person_bbox 紧贴图像边缘，可能裁剪不完整
   - **解决**: ROIManager 已有边界检查和 padding 处理

2. **坐标精度**: 从 ROI 坐标转换回全帧坐标时，采用整数偏移
   - **影响**: 可忽略（误差 < 1 像素）

3. **person_roi 为 None**: 如果 Face-Person 关联失败（未找到匹配的 person）
   - **行为**: PoseEngine 自动回退到全帧检测模式

---

## 📊 数据流总览

```
每帧处理流程:
1. 人脸检测（获取所有人脸）
   ↓
2. 人体检测（获取所有 person bbox）
   ↓
3.  Face-Person 关联（associate_face_and_person）
   - 输出: face_person_pairs (每个 face 对应一个 person)
   ↓
4. TargetPersonManager.update(face_person_pairs)
   - 选择主角（深度 gating + 跟踪连续性）
   - 输出: target_state (包含 face_bbox + person_bbox)
   ↓
5. ROIManager.extract_dual()
   - 输入: target_state.face_bbox, target_state.person_bbox
   - 输出: face_roi, person_roi
   ↓
6. 各模块推理（使用相同主角的 ROI）
   - 情绪: emotion_engine.maybe_infer(face_roi)
   - 疲劳: fatigue_engine.infer(face_roi)
   - 姿态: pose_engine.maybe_infer(person_roi, person_bbox) 
   ↓
7. 结果合并（所有模块的 target_id 一致）
```

---

## 🚀 未来优化方向

1. **智能降级策略**: 当 person_roi 提取失败时，根据置信度决定是否回退到全帧检测

2. **ROI 质量评估**: 检查 person_roi 是否包含完整的身体关键点（肩膀、臀部等）

3. **多人姿态支持**: 扩展到支持多个 person_roi（用于多目标跟踪场景）

4. **性能监控**: 在 TelemetryBuilder 中记录 ROI 使用率和性能对比

---

## 📚 相关文档

- [主角人物管理系统 - Face-Person 关联增强版](TARGET_PERSON_IMPLEMENTATION.md)
- [TensorRT-Only 架构说明](TENSORRT_ONLY_ARCHITECTURE.md)
- [架构决策记录](ARCHITECTURE_DECISION_ORCHESTRATOR.md)

---

## ✅ 实施清单

- [x] 修改 PoseEngine.maybe_infer() 接口（添加 person_roi/person_bbox 参数）
- [x] 修改 PoseEngine._run_inference() 实现（支持 ROI 模式 + 坐标转换）
- [x] 更新 InferenceTask 数据结构（添加 person_roi/person_bbox 字段）
- [x] 更新 InferenceScheduler.submit() 接口
- [x] 更新 InferenceScheduler._infer_pose() 实现
- [x] 更新 DetectionOrchestrator（同步模式）
- [x] 更新 DetectionOrchestrator（异步模式）
- [x] 编写实施文档

---

**最后更新**: 2025-01-21
**版本**: v1.0
**状态**: ✅ 实施完成
