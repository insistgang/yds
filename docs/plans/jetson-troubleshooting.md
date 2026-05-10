# Jetson 常见报错排查清单（对齐 AGENTS.md）

> 版本：对齐 PRD v2.2 / AGENTS.md
> 更新日期：2026-05-10

## 1. 目的

本清单用于排查 Jetson Orin Nano SUPER 在刷机、开机、环境安装和边缘端开发过程中最常见的问题。

**对齐 AGENTS.md 风险响应规范：**
- CSI 相机不出帧时，优先 fallback 到 USB/视频/图片序列（不卡死单点）
- TensorRT 未完成时，先用 PyTorch/Ultralytics 演示
- 自定义模型未完成时，先用 COCO 代理路障演示

适用范围：
- 刷卡后无法启动
- 进入系统后环境异常
- PyTorch / CUDA 不可用
- Ultralytics 运行失败
- CSI 相机不出帧（`/dev/video0` 存在但 JPEG 为 0 字节）
- 项目 mock 链路无法运行

---

## 2. 刷卡后无法启动

### 现象

- 黑屏
- 一直重启
- 卡在启动 Logo
- 显示器无输出

### 优先排查

1. `microSD` 是否刷写成功
2. 电源是否稳定
3. 是否需要先升级 JetPack 5.1.3 固件
4. 显示器输入源是否正确
5. `microSD` 卡本身是否质量过关

### 处理动作

- 重新格式化 `microSD`
- 重新用 `Balena Etcher` 刷镜像
- 确认使用的是官方镜像
- 回看官方 Getting Started 判断是否需要先更新固件

参考：

- <https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit>
- <https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/>

---

## 3. 可以开机，但系统很卡

### 现象

- 进入 Ubuntu 但非常卡
- 网络不稳定
- 软件包状态混乱
- 磁盘空间异常

### 排查命令

```bash
free -h
df -h
lsblk
```

---

## 4. PyTorch / CUDA 不可用

### 现象

- `import torch` 报错
- `torch.cuda.is_available()` 返回 False
- YOLO 推理时提示 CUDA 错误

### 排查

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 处理

- 确认安装的是 Jetson 专用 PyTorch（非 x86 版本）
- 检查 CUDA 版本与 PyTorch 版本匹配
- 参考：`https://elinux.org/Jetson_Zoo`

---

## 5. Ultralytics 运行失败

### 现象

- `from ultralytics import YOLO` 报错
- `yolo checks` 失败
- 模型推理时崩溃

### 排查

```bash
yolo checks
python3 -c "from ultralytics import YOLO; print('OK')"
```

### 处理

- 确认 ultralytics 版本兼容
- 检查模型文件路径是否正确
- 确认依赖包完整（opencv-python, numpy, pillow）

---

## 6. 项目 mock 链路无法运行

### 现象

- `python3 -m linkable_edge.demo` 报错
- 音频播报无输出
- 事件链路中断

### 排查步骤

1. 检查 PYTHONPATH：`export PYTHONPATH=/path/to/edge_p0/src`
2. 检查音频后端：`--audio print` 先验证文本输出
3. 检查模型文件是否存在：`yolo11n.pt` 或自定义模型
4. 检查 TTS 缓存：`~/.cache/linkable_edge/tts_minimax/`

---

## 7. CSI 相机不出帧（AGENTS.md 8.1）

### 判据

- `/dev/video0` 存在但 JPEG 为 0 字节
- Argus / V4L2 出现 timeout
- `nvarguscamerasrc` 报错 `Failed to create CaptureSession`
- `dmesg` 出现 `waitCsiFrameStart timeout`

### 处理顺序（不要卡死在 CSI）

1. 换排线（确认 15-pin -> 22-pin 方向、金手指朝向、压扣锁紧）
2. 换 CSI 口（确认 camera overlay 与实际接口匹配）
3. 换摄像头（已知可用的 IMX219）
4. **切 USB 摄像头**（`cv2.VideoCapture(0, cv2.CAP_V4L2)`）
5. **使用视频文件/图片序列**演示闭环（见 `video_demo.py`）

### 代码侧可继续推进

CSI 未恢复时，以下工作不阻塞：
- `python3 -m linkable_edge.video_demo --source video.mp4` 验证视频输入链路
- `python3 -m linkable_edge.image_demo --source image.jpg` 验证单图推理链路
- 继续采集/标注 P0 数据集
- 后续相机恢复后，只需替换 `FrameSource` 即可继续联调

### 参考

- AGENTS.md 第 8.1 节：CSI 相机不出帧
- `edge_p0/src/linkable_edge/inputs.py`：`auto_select_source()` 自动 fallback

---

> 最后更新：2026-05-10 | 对齐 PRD v2.2 / AGENTS.md
