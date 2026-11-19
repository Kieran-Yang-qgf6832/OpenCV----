# AKAZE跟踪

## 需求
使用AKAZE特征在视频中跟踪物体

## 作用

- **视频目标跟踪**：逐帧估计目标的位移与缩放。
- **电子稳像**：估计帧间变换后反向补偿抖动。
- **视频拼接**：把连续帧对齐到同一坐标系。
- **运动分析**：由单应分解得到相机运动趋势。

适用边界：目标大面积遮挡或快速运动时匹配数不足会跟丢。

## 核心函数
cv2.AKAZE_create(), cv2.BFMatcher()

## 关键步骤
1. 初始化参考帧
2. 逐帧匹配
3. 估计变换
4. 绘制轨迹

## 输入
视频序列

## 输出
跟踪结果

## 素材
`assets/`：`frame_01.png`、`frame_02.png`（单帧静图）、`tracking.mp4`（100 帧 / 25 fps / 4 秒，镜头平移并轻微缩小的连续序列）

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/akaze_tracking

# Python
cd Python && pip install -r requirements.txt && python akaze_tracking.py
```
参数：`-i/--input` 视频路径、`-o/--output` 输出目录、`-r/--ratio` 比率、`-d/--display` 弹窗。
输入为 `tracking.mp4`（100 帧），实测 99 帧全部跟踪成功（100%），首帧关键点 4818。
输出 `output/akaze_tracking.mp4`（带跟踪框的视频）与 `output/akaze_tracking.png`（4 帧采样拼接）。
