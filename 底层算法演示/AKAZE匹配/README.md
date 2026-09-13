# AKAZE匹配

## 需求
使用AKAZE描述子进行特征匹配

## 核心函数
cv2.AKAZE_create(), cv2.BFMatcher()

## 关键步骤
1. 检测关键点和描述子
2. BFMatcher匹配
3. 比率测试筛选
4. 可视化

## 输入
两幅图像

## 输出
匹配点对
