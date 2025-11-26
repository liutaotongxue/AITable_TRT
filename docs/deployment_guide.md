# AITable Jetson 部署指南 / Deployment Guide

## 配置文件位置 / Configuration File Location

The system automatically searches for `system_config.json` in the following order:

1. **Explicitly specified path** (via parameter or command line)
2. **Environment variable** `AITABLE_CONFIG`
3. **Root directory** `system_config.json` (recommended, matches current project)
4. **Config subdirectory** `config/system_config.json` (backward compatibility)

**Recommended location**: Place `system_config.json` in the project root directory.

```bash
AITable_jerorin_TRT/
├── system_config.json    # Place here (recommended)
├── main_gui.py
├── preflight_check.py
├── models/
└── modules/
```

Or use environment variable:

```bash
export AITABLE_CONFIG=/path/to/your/system_config.json
```

---

## 环境配置 / Environment Configuration

### UTF-8 编码配置（解决日志乱码问题）

项目日志使用 UTF-8 编码（包含中文和 emoji），如果在终端看到乱码（如 "馃摲""鈿狅笍""绯荤粺閿欒"），需要配置 UTF-8 环境。

#### 方法 1：临时设置（当前会话有效）

```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8
python main_gui.py
```

#### 方法 2：永久设置（推荐）

编辑 `~/.bashrc` 或 `~/.profile`：

```bash
echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
source ~/.bashrc
```

#### 方法 3：systemd 服务配置

如果使用 systemd 服务运行，在服务单元文件中添加：

```ini
[Service]
Environment="LANG=en_US.UTF-8"
Environment="LC_ALL=en_US.UTF-8"
Environment="PYTHONIOENCODING=utf-8"
```

完整示例：`/etc/systemd/system/aitable.service`

```ini
[Unit]
Description=AITable Monitoring Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/AITable_jerorin_TRT
Environment="LANG=en_US.UTF-8"
Environment="LC_ALL=en_US.UTF-8"
Environment="PYTHONIOENCODING=utf-8"
ExecStart=/usr/bin/python3 main_gui.py
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable aitable.service
sudo systemctl start aitable.service
sudo systemctl status aitable.service
```

#### 方法 4：使用纯英文日志（最佳兼容性）

如果无法修改系统环境配置，可以将日志改为纯英文。参考下文"日志语言切换"章节。

---

## 相机初始化配置 / Camera Initialization

### TOF 相机 SDK 兼容性

本项目已针对 Jetson 板上的旧版 SDK 进行兼容性调整：

- 使用 `MV3D_RGBD_GetParam` 代替 `MV3D_RGBD_GetFloatValue`（旧 SDK 兼容）
- 禁用 ImageAlign/OutputRgbd 设置（旧 SDK 不支持）
- 3 次重试机制启动相机（1 秒延迟）
- 移除暖机流程（与旧版本保持一致）
- 使用应用层内参进行 3D 坐标转换

### 运行前检查

```bash
# 检查相机连接
lsusb | grep -i camera

# 检查 USB 权限
ls -l /dev/video* /dev/bus/usb/

# 如果权限不足，添加用户到 video 组
sudo usermod -aG video $USER
# 重新登录生效
```

### 常见错误处理

#### Error: 0x80060005 (设备忙)

```
可能原因：
1. 其他程序正在使用相机
2. 权限不足
3. USB 连接不稳定

解决方案：
1. 检查其他占用相机的进程：lsof | grep video
2. 使用 sudo 运行程序（不推荐长期使用）
3. 重新插拔 USB 连接
4. 重启 Jetson 板
```

#### Resolution 0×0（分辨率无效）

```
原因：旧版 SDK 未返回有效分辨率

解决：代码已自动使用默认 640×480 分辨率，无需手动处理
```

---

## 日志语言切换 / Log Language Switch

### 选项 A：保持中文日志 + 配置 UTF-8 环境（推荐）

优点：日志信息更直观
缺点：需要配置系统环境

### 选项 B：切换为英文日志

如果需要，运行以下命令生成英文日志版本的文件：

```bash
# 将在下一版本中提供日志语言配置选项
# 或手动修改 logger.info/warning/error 调用
```

---

## 性能优化 / Performance Tuning

### Jetson 功耗模式

```bash
# 查看当前模式
sudo nvpmodel -q

# 设置最大性能模式（MAXN）
sudo nvpmodel -m 0

# 设置风扇（如果有）
sudo jetson_clocks --fan
```

### TensorRT 引擎优化

确保模型已转换为 TensorRT 引擎（`.engine` 文件）：

```bash
ls models/
# 应包含：
# - yolov8n-face.engine
# - emonet_fp16.engine
# - facemesh_fp16.engine
# - yolov8n-pose_fp16.engine
```

---

## 故障排查 / Troubleshooting

### 日志级别调整

编辑 `system_config.json`：

```json
{
  "logging": {
    "level": "DEBUG"  // INFO | DEBUG | WARNING | ERROR
  }
}
```

或使用环境变量：

```bash
export AITABLE_LOG_LEVEL=DEBUG
python main_gui.py
```

### 查看实时日志

```bash
# 如果使用 systemd 服务
sudo journalctl -u aitable.service -f

# 如果直接运行
python main_gui.py 2>&1 | tee aitable.log
```

---

## 更新日志 / Changelog

### v1.0.0 (当前版本)
- 兼容 Jetson 板旧版 TOF SDK
- 移除暖机流程
- 添加 3 次重试机制
- 支持 0×0 分辨率自动回退
- 禁用 SDK 对齐参数（使用应用层转换）

---

## 联系支持 / Support

如遇到问题，请提供以下信息：

1. Jetson 型号和 JetPack 版本：`cat /etc/nv_tegra_release`
2. Python 版本：`python --version`
3. 错误日志（最近 50 行）
4. 系统编码：`locale`
5. 相机型号和 SDK 版本
