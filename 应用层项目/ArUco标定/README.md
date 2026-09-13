# ArUco标定

## 需求
使用ArUco标定板进行相机标定

## 核心函数
cv2.aruco.calibrateCameraAruco()

## 关键步骤
1. 检测ArUco标定板
2. 收集多角度数据
3. 标定计算
4. 评估重投影误差

## 输入
多角度ArUco标定板图像

## 输出
相机参数
