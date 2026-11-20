# 视频写入

## 需求
把一堆 PNG 帧序列编码成可播放的视频文件，并回读校验没有掉帧、没有画质塌陷。

## 作用

- **算法结果落盘**：所有"逐帧处理"的演示都要靠这一步变成可交付的视频。
- **数据集合成**：把标注帧合成训练用视频。
- **归档转码**：统一成 H.264 便于分享与二次压缩。

适用边界：`VideoWriter` 的编码器取决于本机 FFmpeg 构建；没有 H.264 时只能退到 mp4v。

## 核心函数
`cv2.VideoWriter()`（`VideoWriter_fourcc`）、`cv2.VideoCapture()`、`cv2.PSNR()`

## 关键步骤
1. 按文件名顺序读入帧序列
2. 用 mp4v 写临时文件，再交给 ffmpeg 转 H.264（`yuv420p`、`+faststart`）
3. 回读生成的文件，核对帧数/分辨率/帧率
4. 逐帧与原图算 PSNR，量化有损压缩损失；输出检查图

## 输入
`assets/frames/frame_000.png` … `frame_029.png`（640×360，30 帧，画面中有运动目标）

## 输出
H.264 视频与"首/中/末帧 + 参数摘要"检查图

## 素材
`assets/frames/`：从走廊监控片段中截取的 30 张连续帧（640×360），画面里目标在移动，
便于检查编码后运动是否连贯（早期素材 30 帧完全相同，视频看不出任何变化，已替换）。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --fps 30
```
依赖 OpenCV（含 contrib）+ NumPy + ffmpeg（系统已装）。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/frames_to_video.mp4` | H.264，**28 KB** |
| `output/frames_check.png` | 三帧缩略 + 编码参数 + PSNR |

实测：回读 **30 帧 / 30 fps / 640×360**，无掉帧；回读 PSNR **mean 37.87 dB**（min 37.56）。

## 实现要点
- **为什么中转一次**：`VideoWriter` 直接写 mp4v 兼容性一般（PPT/浏览器可能不认），
  转 H.264 后再发布最稳；ffmpeg 不可用时自动保留 mp4v。
- **必须回读校验**：编码参数写错（如宽高必须是偶数）会静默写出 0 帧文件。
- **PSNR 的解读**：37 dB 左右是有损压缩的正常水平，若低于 30 dB 通常是 CRF 设太高。
