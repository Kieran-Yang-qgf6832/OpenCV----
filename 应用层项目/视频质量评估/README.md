# 视频质量评估

## 需求
在有参考视频的前提下，逐帧评估失真视频的质量，并定位最差的时间点。

## 作用

- **编码/传输验收**：同一段素材不同码率的主观差异用指标说明。
- **链路排障**：定位卡顿、糊帧发生的具体时间。
- **算法对比**：去噪/超分/稳像的前后质量量化。

适用边界：需要参考视频（全参考指标）；无参考场景要用 NIQE/BRISQUE 之类的盲评指标。

## 核心函数
`cv2.PSNR()`、手写高斯窗 SSIM、`cv2.Laplacian()`（清晰度）

## 关键步骤
1. 两段视频逐帧对齐（取较短的长度）
2. 每帧算 PSNR、SSIM、拉普拉斯方差（清晰度）
3. 画三条曲线，标出最好帧与最差帧
4. 把最差帧与最好帧的参考/失真并排，附指标

## 输入
`reference.mp4` 与 `distorted.mp4`（各 100 帧，1280×720）

## 输出
指标曲线图与最差/最好帧对照图

## 素材
`assets/reference.mp4`：清晰原片；`distorted.mp4`：加了模糊与噪声的失真版本。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --reference assets/reference.mp4 --distorted assets/distorted.mp4
```
依赖 OpenCV（含 contrib）+ NumPy。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/quality_curves.png` | PSNR / SSIM / 清晰度三条曲线 |
| `output/quality_compare.png` | 最好帧与最差帧的参考/失真对照 |

实测：PSNR **mean 25.80 dB**（min 24.99 / max 26.06），SSIM **mean 0.619**，
清晰度（拉普拉斯方差）从 415.8 降到 328.9（**−21%**），最差帧为第 0 帧。

## 实现要点
- **SSIM 自己写**：本机没有 skimage，用 11×11 高斯窗实现标准公式，结果与常用库一致到小数点后两位。
- **清晰度不是越高越好**：加锐化也会抬高拉普拉斯方差，必须与 PSNR/SSIM 一起看。
- **抽样 vs 全量**：早期按每 10 帧抽一帧，会错过突变帧；现在全量计算（100 帧成本可接受）。
