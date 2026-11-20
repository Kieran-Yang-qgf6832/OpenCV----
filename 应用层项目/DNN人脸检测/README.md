# DNN 人脸检测

## 需求
用 DNN 在一张图里检出多张人脸并给出置信度，替代老旧的 Haar 级联。

## 作用

- **门禁/考勤**：先检测再识别。
- **照片整理**：人脸区域裁剪、打码。
- **直播美颜**：检测框是磨皮滤镜的作用域。

适用边界：侧脸、强逆光、极小脸会漏检；模型输入尺度对人脸大小敏感（见演示结果）。

## 核心函数
`cv2.dnn.readNetFromCaffe()`、`cv2.dnn.NMSBoxes()`

## 关键步骤
1. 加载 res10 SSD（300×300，Caffe）
2. 按训练均值 (104, 177, 123) 归一化后前向
3. 置信度过滤 + NMS（阈值 0.35）去掉重叠框
4. 在单人像与拼贴合影上各跑一遍，并对比 300/600 两种输入尺度

## 输入
`faces.png`（单人像）与 `group_photo.png`（三人拼贴合影）

## 输出
两张标注图（框 + 编号 + 置信度）

## 素材
`assets/faces.png`：单人正装照；
`assets/group_photo.png`：把三张现成人像拼到模糊街景背景上得到的合成合影（方便演示多人脸）。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --conf 0.7
```
依赖 OpenCV + NumPy；模型 `_models/res10_deploy.prototxt` 与 `res10_face.caffemodel`（10 MB）。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/faces_detected.png` | 单人像：1 张脸，置信度 1.000 |
| `output/faces_group.png` | 合影：3 张脸（1.000 / 0.875 / 0.854） |

输入尺度实验：300 px 时检出 3 张，**600 px 时检出 4 张** —— 小脸对输入分辨率很敏感。

## 实现要点
- **为什么不用 YuNet**：YuNet 的 ONNX 需要 OpenCV ≥ 4.8，本机 4.6 加载即报
  `Layer with requested id=-1 not found`，模型文件也不会被用到，已从 `_models/` 移除。
- **输入尺度是隐藏参数**：把 `input_size` 从 300 提到 600，最小的一张脸才被检出，
  代价是耗时约 3 倍。
- **NMS 别省**：SSD 会对同一张脸给出多个高置信框，不做 NMS 计数直接翻倍。
