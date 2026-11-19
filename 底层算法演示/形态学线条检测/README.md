# 形态学线条检测

## 需求
使用形态学操作检测水平和垂直线条

## 核心函数
cv2.morphologyEx(), cv2.getStructuringElement()

## 关键步骤
1. 二值化
2. 创建方向结构元素
3. 开运算提取线条
4. 轮廓筛选

## 输入
灰度图像

## 输出
检测到的线条

## 素材
`assets/`：`table.png`
