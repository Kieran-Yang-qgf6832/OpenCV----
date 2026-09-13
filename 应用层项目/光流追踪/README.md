# 光流追踪

## 需求
在视频中追踪目标或特征点轨迹

## 核心函数
cv2.goodFeaturesToTrack(), cv2.calcOpticalFlowPyrLK()

## 关键步骤
1. 初始化特征点
2. 逐帧计算光流
3. 筛选追踪点
4. 绘制轨迹

## 输入
视频序列

## 输出
带追踪轨迹的视频
