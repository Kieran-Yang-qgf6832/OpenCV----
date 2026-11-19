# AKAZE跟踪

## 需求
使用AKAZE特征在视频中跟踪物体

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
