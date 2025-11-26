# Orchestrator Architecture Decision

## Current State: Dual Main Loop Architecture

The codebase currently maintains **two parallel execution paths**, leading to duplicate maintenance burden:

### Path 1: Synchronous Main Loop (Currently Active)
**Location**: `main_gui.py:140-260` → `run_main_loop()`

**Characteristics**:
- Direct OpenCV window management
- Manual keyboard event handling
- Synchronous inference calls (blocking)
- Simple linear control flow
- No telemetry or performance monitoring
- Single-threaded execution

**Pros**:
- Simple and easy to understand
- Debuggable with standard tools
- Low latency for real-time GUI
- Currently working in production

**Cons**:
- Blocks on slow inference (e.g., emotion classifier)
- No performance telemetry
- Hard to add new modules without blocking
- Manual resource management

---

### Path 2: Async Orchestrator (Exists but Unused)
**Location**:
- `modules/core/orchestrator.py` - Multi-module coordinator
- `modules/core/inference_scheduler.py` - Async task scheduling
- `modules/core/async_engine.py` - Async inference wrapper
- `modules/core/telemetry.py` - Performance monitoring
- `modules/ui/window_manager.py` - Advanced window management

**Characteristics**:
- Async/await pattern for non-blocking inference
- Centralized orchestration of detection modules
- Built-in telemetry and performance tracking
- Queue-based inference scheduling
- Structured module lifecycle management

**Pros**:
- Non-blocking async inference
- Built-in telemetry (latency, FPS, GPU usage)
- Easy to add new modules
- Better resource management
- Future-proof for multi-camera/multi-model

**Cons**:
- More complex debugging
- Requires async/await understanding
- Higher initial latency overhead
- Never tested in production

---

## Problem Statement

**Maintenance Cost**: Maintaining two parallel architectures doubles the work:
- Bug fixes must be applied to both paths
- New features require dual implementation
- Configuration changes affect both systems
- Testing burden multiplied

**Resource Waste**: The orchestrator infrastructure (`orchestrator.py`, `inference_scheduler.py`, etc.) is fully implemented but never used, representing significant dead code.

---

## Decision Options

### Option A: Commit to Orchestrator (Recommended for Scale)

**When to Choose**:
- Planning to add more detection modules (e.g., gesture, object tracking)
- Need performance telemetry for production monitoring
- Want async inference to avoid GUI freezing
- Deploying on multi-camera setups

**Action Items**:
1. Create `main_orchestrator.py` as new entry point
2. Migrate `run_main_loop()` logic to orchestrator callbacks:
   ```python
   from modules.core.orchestrator import Orchestrator
   from modules.core.inference_scheduler import InferenceScheduler

   async def main():
       scheduler = InferenceScheduler()
       orchestrator = Orchestrator(scheduler, config)

       # Register modules
       orchestrator.register_module("face_detection", face_detector)
       orchestrator.register_module("emotion", emotion_classifier)
       orchestrator.register_module("fatigue", fatigue_detector)

       await orchestrator.run()
   ```
3. Update `run_aitable.sh` to use new entry point
4. Archive old `main_gui.py` as `main_gui_legacy.py`

**Migration Effort**: ~3-5 days
**Risk**: Medium (requires thorough testing)

---

### Option B: Keep Synchronous Path (Recommended for Simplicity)

**When to Choose**:
- Current performance is acceptable
- Single-camera, single-module deployment
- Team prefers simple, debuggable code
- No need for async capabilities

**Action Items**:
1. **Delete unused orchestrator files**:
   ```bash
   rm -rf modules/core/orchestrator.py
   rm -rf modules/core/inference_scheduler.py
   rm -rf modules/core/async_engine.py
   rm -rf modules/ui/window_manager.py
   ```
2. Document decision in `ARCHITECTURE.md`
3. Remove orchestrator references from docs

**Migration Effort**: ~1 day (cleanup only)
**Risk**: Low

---

### Option C: Hybrid Approach (Not Recommended)

Keep both paths but clearly separate them:
- `main_gui.py` - Synchronous, real-time GUI
- `main_orchestrator.py` - Async, headless batch processing

**When to Choose**:
- Need both GUI and headless modes
- Different deployment scenarios (dev vs production)

**Cons**:
- Still maintains duplicate logic
- Configuration must work for both
- Highest maintenance burden

---

## Recommendation

### For Jetson Orin Nano Production Deployment: **Option B (Keep Synchronous)**

**Rationale**:
1. **Current system works**: The synchronous path is production-tested
2. **Real-time constraints**: GUI responsiveness is critical for eye distance monitoring
3. **Single-camera deployment**: No need for multi-stream orchestration
4. **Team expertise**: Synchronous code is easier to debug on embedded systems
5. **TensorRT already fast**: Async overhead may not provide benefits

**Implementation**:
```bash
# Clean up unused orchestrator files
cd AITable_jerorin_TRT
rm modules/core/orchestrator.py
rm modules/core/inference_scheduler.py
rm modules/core/async_engine.py
rm modules/ui/window_manager.py

# Update documentation
echo "Architecture: Synchronous main loop (main_gui.py)" >> ARCHITECTURE.md
```

**Future Migration Path**: If requirements change (multi-camera, batch processing), orchestrator can be re-implemented based on the archived code.

---

## Resource Management Best Practices

Regardless of architecture choice, **always use explicit resource management**:

### TRTFaceDetector Usage

```python
# BAD: Relies on __del__ (unreliable)
detector = TRTFaceDetector("model.engine")
# ... use detector ...
# Hope __del__ gets called

# GOOD: Explicit cleanup
detector = TRTFaceDetector("model.engine")
try:
    # ... use detector ...
finally:
    detector.close()  # Guaranteed cleanup

# BEST: Context manager
with TRTFaceDetector("model.engine") as detector:
    # ... use detector ...
    pass
# Automatic cleanup via __exit__
```

### Main Loop Integration

```python
def run_main_system():
    # ... camera setup ...

    # Use context manager for TRT detector
    with TRTFaceDetector(model_path) as face_detector:
        system = EyeDistanceSystem(camera, face_detector=face_detector)

        try:
            run_main_loop(system)  # Main GUI loop
        except KeyboardInterrupt:
            logger.info("User interrupted - cleaning up...")
        finally:
            # Cleanup is automatic via __exit__
            pass
```

---

## Summary

| Aspect | Synchronous Path | Orchestrator Path |
|--------|------------------|-------------------|
| **Complexity** | Low | High |
| **Maintenance** | Easy | Complex |
| **Performance** | Good for single-stream | Better for multi-stream |
| **Debugging** | Easy | Harder |
| **Telemetry** | Manual | Built-in |
| **Future-proof** | Limited | Scalable |
| **Recommendation** | **For current deployment** | Consider for future scale |

**Decision**: Use **Option B (Synchronous)** for Jetson Orin Nano deployment, clean up orchestrator dead code to reduce maintenance burden.

---

## Change Log

- **2025-01-19**: Initial architecture decision document
- **2025-01-19**: Added explicit resource management recommendations (TRTFaceDetector.close())
