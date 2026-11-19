# 单应性

## 需求
计算两个平面之间的单应性变换矩阵

## 核心函数
cv2.findHomography(), cv2.warpPerspective()

## 关键步骤
1. 特征匹配
2. RANSAC估计矩阵
3. 透视变换
4. 图像拼接

## 输入
两组匹配点

## 输出
3x3单应矩阵

## 素材
`assets/`：`poster.png`、`poster_perspective.png`
