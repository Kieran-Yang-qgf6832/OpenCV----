# 视频稳像

## 需求
消除视频抖动保持画面稳定

## 核心函数
cv2.calcOpticalFlowFarneback(), cv2.warpAffine()

## 关键步骤
1. 运动估计
2. 运动平滑
3. 运动补偿
4. 边界填充

## 输入
抖动视频

## 输出
稳定视频
