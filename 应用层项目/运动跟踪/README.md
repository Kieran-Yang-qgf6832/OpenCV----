# 运动跟踪

## 需求
跟踪视频中多个运动目标

## 核心函数
cv2.MultiTracker_create(), cv2.BackgroundSubtractor()

## 关键步骤
1. 检测运动目标
2. 初始化多目标跟踪器
3. 逐帧更新
4. 轨迹绘制

## 输入
视频序列

## 输出
多目标跟踪结果
