# 光流法

## 需求
计算图像序列中像素的运动矢量

## 作用

- **运动估计**：算出相邻帧之间每个像素移动了多少。
- **目标跟踪**：给特征点算位移并串成轨迹。
- **视频稳像**：估计全局运动后反向补偿相机抖动。
- **运动分割**：把运动方向不同的区域分成独立目标。

适用边界：要求亮度恒定且位移较小，大位移需要金字塔或更高帧率。

## 核心函数
cv2.calcOpticalFlowPyrLK(), cv2.calcOpticalFlowFarneback()

## 关键步骤
1. 读取视频帧
2. 采样特征点
3. 计算光流矢量
4. 可视化运动

## 输入
连续视频帧

## 输出
光流矢量场

## 素材
`assets/`：`frame_01.png`、`frame_02.png`（单帧静图）、`flow.mp4`（100 帧 / 25 fps / 4 秒，匀速平移的连续帧序列）

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/optical_flow --points=120

# Python
cd Python && pip install -r requirements.txt && python optical_flow.py --points 120
```
参数：`-a/--frameA`、`-b/--frameB` 连续两帧（缺省读 `assets/`）、`-o/--output` 输出路径、`-p/--points` LK 最大特征点数、`-d/--display` 弹窗。
结果默认写入 `output/optical_flow.png`，终端打印 LK 平均位移与 Farneback 平均光流。若要用整段视频，可先自行从 `flow.mp4` 抽相邻两帧。
