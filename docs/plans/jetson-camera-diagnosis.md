# Jetson 相机出帧问题诊断记录（2026-04-27）

> **状态**：已按 AGENTS.md 8.1 处理，已切换至 USB 摄像头 + 视频文件 fallback
> **当前**：USB 摄像头 `/dev/video0` 验证通过（MJPG 640x480@30FPS）
> **参考**：`edge_p0/src/linkable_edge/inputs.py` 已统一封装 CSI/USB/视频/图片四种输入源

---

## 1. 历史结论（2026-04-27）

远端 Jetson 能识别 IMX219 摄像头，但实际图像帧没有成功进入用户态。
本次诊断结果指向 **CSI / 传感器出帧 / Argus 捕获层问题**，暂不指向 `edge_p0` 项目代码问题。

---

## 2. 远端环境

- 主机：`yahboom`
- 系统：Jetson Linux `R36.4.7`
- 内核：`5.15.148-tegra`
- Python：`3.10.12`
- PyTorch：`2.5.0a0+872d972e41.nv24.08`
- CUDA 可用：`torch.cuda.is_available() == True`
- Ultralytics：`8.4.24`
- 电源模式：`MAXN_SUPER`
- YOLO 检查：`yolo checks` 通过

---

## 3. 相机识别状态

`v4l2-ctl --list-devices` 能看到：

```text
NVIDIA Tegra Video Input Device (platform:tegra-camrtc-ca):
    /dev/media0

vi-output, imx219 9-0010 (platform:tegra-capture-vi:1):
    /dev/video0
```

`/dev/video0` 能列出 IMX219 支持格式：

```text
3280x2464 21fps
3280x1848 28fps
1920x1080 30fps
1640x1232 30fps
1280x720 60fps
```

`media-ctl -p -d /dev/media0` 显示 sensor -> nvcsi -> vi-output 链路处于 enabled。

---

## 4. 失败现象

V4L2 低层抓帧：
```text
VIDIOC_STREAMON returned 0 (Success)
cap dqbuf ... bytesused: 1843200 ... (error, ts-monotonic, ts-src-eof)
```

最终输出 `/tmp/linkable-v4l2.raw` 为 `0` 字节。

GStreamer / Argus 抓 JPEG：
```text
Error generated ... gstnvarguscamerasrc.cpp, execute:805 Failed to create CaptureSession
```

最终输出 `/home/jetson/linkable-gst-frame-cam0.jpg` 为 `0` 字节。

重启 `nvargus-daemon` 后，显式使用 `sensor-mode=2` 和 `sensor-mode=4` 复测，仍未解决。后续用项目 `camera_check` 复测时，Argus 能列出 sensor modes 并进入 capture setup，但仍没有有效 JPEG，输出：

```text
nvbuf_utils: dmabuf_fd -1 mapped entry NOT found
NvBufSurfaceFromFd Failed
capture_status=empty_file
```

因此不能只看 `gst-launch` 退出码，必须同时检查抓图文件大小。

---

## 5. 关键日志

`journalctl -u nvargus-daemon` 出现：
```text
SCF: Error InvalidState ... openViCsi()
SCF: Error InvalidState ... createSession()
waitCsiFrameStart timeout
waitCsiFrameEnd timeout
```

`dmesg` 出现：
```text
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
tegra-camrtc-capture-vi tegra-capture-vi: err_rec: attempting to reset the capture channel
tegra-camrtc-capture-vi tegra-capture-vi: err_rec: successfully reset the capture channel
```

---

## 6. 已排除项

- SSH 可达
- Jetson 系统可正常运行
- `/dev/video0` 存在
- `nvarguscamerasrc` 插件存在
- `nvargus-daemon` 服务 active
- 没有发现其他进程长期占用 `/dev/video0`
- PyTorch / CUDA / Ultralytics 可用
- YOLO 单张图片推理成功

---

## 7. 处理结果（按 AGENTS.md 8.1 执行）

1. ✅ 换排线：已确认方向正确，问题未解决
2. ✅ 换 CSI 口：已复测，问题未解决
3. ✅ 换摄像头：已用已知可用 IMX219 复测，问题未解决
4. ✅ **切 USB 摄像头**：已验证通过（MJPG 640x480@30FPS）
5. ✅ **使用视频文件/图片序列**：`video_demo.py` 已就绪

**结论**：CSI 硬件问题（可能为排线/接口/固件），不阻塞 P0 演示。
现场演示采用 USB 摄像头或视频文件输入。

---

## 8. 当前可用输入源（inputs.py 已统一封装）

相机问题不阻塞以下工作：
- `python3 -m linkable_edge.usb_demo --device-idx 0 --audio print` USB 摄像头实时演示
- `python3 -m linkable_edge.video_demo --source video.mp4 --audio print` 视频文件演示
- `python3 -m linkable_edge.image_demo --source image.jpg --audio print` 单张图片验证
- `python3 -m linkable_edge.demo --audio print` mock 数据验证事件 -> 模板 -> 播报链路
- 继续采集/标注 P0 数据集（目标 >=600 张）

**后续相机恢复后**：只需修改 `auto_select_source()` 参数或替换 `FrameSource` 即可继续联调。

---

> 最后更新：2026-05-10 | 对齐 AGENTS.md 8.1
