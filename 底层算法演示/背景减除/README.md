# 背景减除

## 需求
从视频序列中分离前景运动物体

## 核心函数
cv2.createBackgroundSubtractorMOG2(), cv2.createBackgroundSubtractorKNN()

## 关键步骤
1. 创建背景减除器
2. 逐帧获取前景
3. 形态学优化
4. 提取目标

## 输入
视频序列

## 输出
前景掩码

## 素材
`assets/`：`frame.png`
