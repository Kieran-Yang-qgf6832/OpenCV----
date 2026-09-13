# 边缘检测

## 需求
使用Sobel、Laplacian和Canny算子检测图像边缘

## 核心函数
cv2.Sobel(), cv2.Laplacian(), cv2.Canny()

## 关键步骤
1. 灰度转换
2. 高斯降噪
3. 计算梯度幅值
4. 双阈值处理

## 输入
灰度或彩色图像

## 输出
边缘图像
