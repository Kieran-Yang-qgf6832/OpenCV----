# 实时姿态估计

## 需求
用 OpenPose 模型在图像/视频中估计人体关键点，并画出骨架。

## 作用

- **健身动作计数/纠正**：深蹲、俯卧撑的角度判定。
- **体感交互**：无需穿戴设备的动作输入。
- **人机安全**：判断人员是否进入危险姿态/区域。

适用边界：单人、全身入画效果最好；画面只拍到局部时只能检出可见关节。

## 核心函数
`cv2.dnn.readNetFromCaffe()`、`net.forward()`（18 张关键点热力图）

## 关键步骤
1. 加载 OpenPose COCO 模型（prototxt + 208 MB caffemodel）
2. 图像缩放到 288 宽后放到 288×288 画布左上角，`1/255` 归一化送网络
3. 输出 19 张热力图，取每张的峰值作为关节坐标，置信度低于 0.25 丢弃
4. 按 COCO 骨架拓扑连线（17 条边），并把坐标映射回原图

## 输入
`person.png`（全身人像）与 `pose.mp4`（100 帧，画面多为躯干特写）

## 输出
全身骨架图、骨架视频、视频关键帧

## 素材
`assets/person.png`：健身房全身人像（18 点齐全）；
`assets/pose.mp4`：同一地点的手持特写镜头，多数帧只拍到腰部以下。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --input-size 288 --conf 0.15
```
依赖 OpenCV + NumPy；模型 `_models/openpose_pose_coco.prototxt` 与 `openpose_pose_coco.caffemodel`
（208 MB，Tlaloc-Es/openpose 的 HuggingFace 镜像）。

## 演示结果
| 文件 | 内容 |
| --- | --- |
| `output/pose_still.png` | 全身人像 **18/18** 关节骨架 |
| `output/pose.mp4` | 视频逐帧骨架（平均 **8.0/18** 关节） |
| `output/pose_frame.png` | 视频中间帧 |

置信度最高的关节：Lknee 0.92、Lelb 0.85、Rknee 0.85、Rwri 0.85。

## 已知局限（如实说明）
- **视频素材是躯干特写**：头部与肩部根本不在画面内，所以只有 8/18 个关节，
  这是素材问题而不是模型问题。真正好看的骨架是 `pose_still.png` 那张全身图。
- 该模型是单人模型；多人场景需要先做人体检测再逐人裁剪。

## 实现要点
- **坐标映射的坑**：热力图是 36×36，换算回原图要用画布尺寸（288），
  不是缩放后图像的宽度；用错会让所有关节点整体偏移。
- **画布填充**：保持左上角对齐 + 黑底填充，与 OpenPose 官方预处理一致。
