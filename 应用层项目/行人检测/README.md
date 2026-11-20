# 行人检测

## 需求
在街景视频里检测行人并逐帧计数。

## 作用

- **客流统计**：门店、站台的实时人数。
- **安防布控**：重点区域出现行人时告警。
- **自动驾驶感知**：行人是最高优先级目标。

适用边界：MobileNet-SSD 是轻量模型，遮挡严重、极端小目标会漏检。

## 核心函数
`cv2.dnn.readNetFromCaffe()`、`cv2.dnn.blobFromImage()`、`net.forward()`

## 关键步骤
1. 加载 MobileNet-SSD（VOC 20 类，person 为第 15 类）
2. 300×300 blob（`0.007843` 缩放、`127.5` 均值）前向
3. 按置信度 0.35 过滤并还原到原图坐标
4. 画框 + 置信度 + 人数，同时统计全片人数曲线

## 输入
`pedestrians.mp4`（100 帧街景）与 `street.png`（单张场景图）

## 输出
检测视频、关键帧、静态图结果、人数曲线

## 素材
`assets/pedestrians.mp4`、`street.png`：真实街景，画面里有多位行人。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --conf 0.4
```
依赖 OpenCV + NumPy；模型 `_models/mobilenetssd_deploy.prototxt` 与 `mobilenetssd.caffemodel`（23 MB，
来自 chuanqi305/MobileNet-SSD 仓库）。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/pedestrians_detected.mp4` | 逐帧绿框 + 人数 |
| `output/pedestrian_frame.png` | 中间帧 |
| `output/pedestrian_still.png` | 静态图检出 4 人 |
| `output/pedestrian_count.png` | 逐帧人数曲线 |

实测：视频逐帧人数 **mean 3.38 / max 7**，检出率 **99%**。

## 实现要点
- **为什么不用 HOG+SVM**：`HOGDescriptor.getPeopleDetector64x128` 在真实街景上漏检严重，
  换成 DNN 后同一素材的检出率从个位数升到 99%。
- **类别过滤**：SSD 输出 20 类，必须只保留 class id 15，否则椅子、汽车也会被框。
- **置信度阈值**：0.35 是召回与误检的折中；统计口径要求严格时可提到 0.5。
