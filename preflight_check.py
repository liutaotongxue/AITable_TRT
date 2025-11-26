#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware/Dependency Preflight Check Script

Check system dependencies, model files and TOF camera SDK
"""

import sys
import os
from pathlib import Path

# Add modules to path
current_dir = Path(__file__).parent
modules_dir = current_dir / 'modules'
if str(modules_dir) not in sys.path:
    sys.path.insert(0, str(modules_dir))


def preflight_check(config_path=None):
    """
    Execute preflight checks

    Args:
        config_path: Configuration file path (optional)

    Returns:
        bool: Whether preflight passed
    """
    print("\n" + "="*70)
    print("Running system preflight checks...")
    print("="*70)

    # Check Python version
    print(f"\n[OK] Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"[OK] TensorRT {trt.__version__}")
    except ImportError:
        print("[FAIL] TensorRT not installed")
        return False

    # Check PyCUDA
    try:
        import pycuda.driver as cuda
        cuda.init()
        print("[OK] PyCUDA available")
    except Exception as e:
        print(f"[FAIL] PyCUDA initialization failed: {e}")
        return False

    # Check configuration file
    if config_path is None:
        # Auto-detect: try root directory first, then config/
        config_file = current_dir / "system_config.json"
        if not config_file.exists():
            config_file = current_dir / "config" / "system_config.json"
    else:
        config_file = Path(config_path)

    if config_file.exists():
        print(f"[OK] Config file: {config_file}")
    else:
        print(f"[FAIL] Config file not found: {config_file}")
        return False

    # Check model files
    models_dir = current_dir / "models"
    if models_dir.exists():
        engine_files = list(models_dir.glob("*.engine"))
        if engine_files:
            print(f"[OK] Found {len(engine_files)} TensorRT engine files")
            for engine in engine_files[:3]:  # Show first 3 only
                print(f"  - {engine.name}")
        else:
            print("[WARN] No .engine files in models/ directory")
    else:
        print("[WARN] models/ directory not found")

    # Check TOF Camera SDK and availability (critical check)
    print("\n[CHECKING] TOF Camera SDK and availability...")
    tof_check_passed = False
    config = None

    try:
        # Import configuration loader
        from modules.core.config_loader import get_config
        config = get_config(config_path=config_file)

        # Check SDK path configuration
        sdk_python_path = config.paths.get("sdk_python_path")
        sdk_lib_path = config.paths.get("sdk_lib_path_aarch64")

        print(f"  SDK Python path: {sdk_python_path}")
        print(f"  SDK library path: {sdk_lib_path}")

        # Try to initialize TOF camera manager
        from modules.camera.tof_manager import TOFCameraManager

        with TOFCameraManager(
            sdk_python_path=sdk_python_path,
            sdk_library_path=sdk_lib_path
        ) as camera:
            # Try to initialize camera
            if camera.initialize():
                print(f"[OK] TOF camera initialized successfully")
                tof_check_passed = True
            else:
                print("[FAIL] TOF camera initialization failed")

    except Exception as e:
        print(f"[FAIL] TOF camera check failed: {e}")

    # Check if required component based on configuration
    if not tof_check_passed:
        try:
            # Reuse config from above if available
            if config is None:
                from modules.core.config_loader import get_config
                config = get_config(config_path=config_file)

            tof_required = config.hardware.get("tof_camera", {}).get("required", True)

            if tof_required:
                print("\n[CRITICAL] TOF camera is REQUIRED but not available!")
                print("System cannot start without a working TOF camera.")
                print("\nTroubleshooting:")
                print("1. Check camera connection")
                print("2. Verify SDK paths in system_config.json")
                print("3. Ensure proper permissions")
                print("4. Check SDK compatibility with platform")
                return False
            else:
                print("[WARN] TOF camera not available (optional component)")
        except Exception as e:
            # Default assume camera is required
            print(f"\n[CRITICAL] TOF camera check failed and cannot determine if required: {e}")
            return False

    print("\n" + "="*70)
    print("Preflight checks completed successfully")
    print("="*70 + "\n")

    return True


def main():
    """Command line entry point"""
    success = preflight_check()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
