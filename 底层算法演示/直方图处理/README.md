# 直方图处理

## 需求
计算直方图并进行均衡化和CLAHE增强

## 核心函数
cv2.calcHist(), cv2.equalizeHist(), cv2.createCLAHE()

## 关键步骤
1. 计算直方图
2. 全局均衡化
3. 自适应CLAHE
4. 对比效果

## 输入
灰度/彩色图像

## 输出
增强后图像
