# 二维码扫描器

## 需求
在图片与视频中检测二维码并解出内容，模拟扫码枪的工作过程。

## 作用

- **扫码入场/支付**：最典型的二维码落地场景。
- **设备配网**：扫设备上的码自动填参数。
- **票据核验**：票面二维码内容提取。

适用边界：需要足够的静止清晰度；强反光、弯曲、部分遮挡会导致解码失败。

## 核心函数
`cv2.QRCodeDetector()`、`detectAndDecode()`

## 关键步骤
1. 单张图调用 `detectAndDecode`，同时拿到内容、四个角点
2. 画角点与四边形，并在图下追加信息栏（内容 + 字节数）
3. 视频逐帧检测，成功帧画绿框并把内容压在画面上，失败帧显示搜索提示
4. 统计解码成功率并输出 H.264 视频

## 输入
`qrcode.png`（512×512）与 `scan.mp4`（100 帧，码面逐渐占满画面）

## 输出
标注图、扫描过程视频、关键帧

## 素材
`assets/qrcode.png`：内容为 `https://opencv.org/opencv-project-book`；
`assets/scan.mp4`：同一码面从远到近的推近镜头。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --image assets/qrcode.png --video assets/scan.mp4
```
依赖 OpenCV（含 contrib）+ NumPy。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/qr_detected.png` | 单图解码：内容 + 38 字节 + 四角标记 |
| `output/qr_scan.mp4` | 逐帧扫描，成功帧带绿框与内容 |
| `output/qr_frame.png` | 视频中间帧 |

实测：单图正确解出 **38 字节**内容，视频 **100/100 帧**全部解码成功。

## 实现要点
- **一次调用拿三样东西**：`detectAndDecode` 同时返回内容、角点、矫正后的码图，
  不必先 `detect` 再 `decode`。
- **多码场景**：一帧多码用 `detectAndDecodeMulti`，返回内容列表。
- **失败不抛异常**：解不出来时返回空串，画面切到"searching"提示，视频照常推进。
