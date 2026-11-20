# 图像超分辨率

## 需求
把 4 倍下采样的低清图恢复到高分辨率，并与双三次插值对比，量化真实收益。

## 作用

- **老素材复用**：低分辨率截图放大后用于印刷/大屏。
- **监控截图增强**：车牌、人脸的可用性提升。
- **端侧预处理**：先超分再送检测网络，小目标召回更高。

适用边界：超分只能"猜"出高频信息，不会凭空创造细节；倍率越大、内容越纹理化，收益越明显。

## 核心函数
`cv2.dnn_superres.DnnSuperResImpl_create()`、`readModel()`、`setModel("edsr", 4)`、`upsample()`

## 关键步骤
1. 读低清图（256×256）与参考高清图（1024×1024）
2. 双三次插值作基线；EDSR 走 `dnn_superres` 端到端放大 4 倍
3. 与参考图算 PSNR / SSIM
4. 在纹理密集处截 300×300 局部放大，三图并排（双三次 / EDSR / 真值）

## 输入
`lowres.png`（256×256）、`reference.png`（1024×1024，真值）

## 输出
全图对照（双三次 vs EDSR + 真值）与局部放大三联图

## 素材
`assets/lowres.png` 由 `reference.png` 用 `INTER_AREA` 严格降采样 4 倍得到（见图生成脚本
`../_tools/gen_scene_assets.py`），所以"低清图正好是高清单一的降采样结果"，
评价不会因为内容不一致而失去意义。

## 运行
```bash
cd Python && python3 demo.py --display
```
依赖 OpenCV（含 contrib，`dnn_superres` 在 contrib 里）+ NumPy；
模型 `_models/EDSR_x4.pb`（38 MB，Saafke/EDSR_Tensorflow 仓库）。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/superres.png` | 双三次 / EDSR / 真值三联 |
| `output/superres_crop.png` | 砖墙纹理处 300×300 局部放大 |

实测：双三次 **23.78 dB / SSIM 0.632**，EDSR **24.14 dB / SSIM 0.660**（+0.36 dB）。

## 实现要点
- **收益为什么只有 0.36 dB**：基线本身就很高（降采样再双三次能恢复大部分低频），
  而砖墙纹理属高频随机结构，EDSR 主要改善的是边缘锐度，所以看局部放大图比看整图明显。
- **不要造假对比**：早期素材的低清图与参考图不是同一张，PSNR 只有 23 dB 却毫无意义；
  现在低清图由参考图严格降采样得到，指标才有解释力。
- **替代模型**：`ESPCN`/`FSRCNN` 更快但更糊，`LapSRN` 介于两者之间，换模型只改 `setModel` 的名字就行。
