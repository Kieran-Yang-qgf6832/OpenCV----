# AR标记叠加

## 需求
检测ArUco标记并叠加虚拟3D物体

## 核心函数
cv2.aruco.detectMarkers(), cv2.solvePnP()

## 关键步骤
1. 检测ArUco标记
2. 估计相机位姿
3. 投影3D模型
4. 实时叠加渲染

## 输入
摄像头视频流

## 输出
AR增强视频流
