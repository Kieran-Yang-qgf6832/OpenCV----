# 文档扫描仪

## 需求
把手机翻拍的文档照片自动校正成平整、白底黑字的"扫描件"。

## 作用

- **随拍随扫**：不依赖扫描仪，拍完直接得到可存档的 PDF 级图像。
- **票据/合同归档**：自动裁边、拉正、去阴影。
- **OCR 前置**：拉正后的图像识别率显著更高。

适用边界：纸张四边要在画面内、背景与纸面有亮度差；带花纹背景或透明纸会找错角点。

## 核心函数
`cv2.findContours()`、`cv2.approxPolyDP()`、`cv2.getPerspectiveTransform()`、`cv2.adaptiveThreshold()`

## 关键步骤
1. 用亮度阈值把纸面从深色背景中分离（素材 alpha 全不透明，只能靠亮度）
2. 最大轮廓做多边形逼近取四角，逼近失败时用最小外接矩形兜底
3. 按真实长宽比做四点透视变换，拉到 700 px 宽
4. 光照均衡（除以大尺度模糊）+ 自适应阈值，得到扫描件

## 输入
3 张翻拍照片（`doc-01/02/03`，1920×2560 起）

## 输出
每组四联：原照片 / 角点检测 / 拉正 / 二值扫描件

## 素材
`assets/doc-01-flat-white-on-dark.png`（平放）、`doc-02-tilted-on-charcoal.png`（倾斜）、
`doc-03-receipt-on-charcoal.png`（小票，长宽比不同）。

## 运行
```bash
cd Python && python3 demo.py --display
```
依赖 OpenCV（含 contrib）+ NumPy。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/scan_result.png` | 三张素材 ×（原图/角点/拉正/二值）四联 |
| `output/scan_corners.png` | 角点检测放大图 |

实测：纸张占画面 **40%~49%**，拉正后宽度 700 px，长宽比与原纸一致（小票拉出 700×1318）。

## 实现要点
- **alpha 通道不可靠**：素材带 alpha 但全为 255，早期按 alpha 找纸张会把整幅图当成纸面；
  改用亮度阈值 + 形态学开闭后一次成功。
- **光照均衡比阈值更关键**：`gray / GaussianBlur(gray, 35)` 先抹掉渐变阴影，
  再自适应阈值，暗角处才不会变成黑块。
- **兜底路径**：`approxPolyDP` 找不到正好 4 个点时退回 `minAreaRect`，避免直接失败。
