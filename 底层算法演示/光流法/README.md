# 光流法

## 需求
计算图像序列中像素的运动矢量

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
