# 平面物体检测

## 需求
在场景中检测平面物体并估计姿态

## 核心函数
cv2.findHomography(), cv2.perspectiveTransform()

## 关键步骤
1. 特征检测匹配
2. 计算单应性
3. 分解姿态
4. 绘制检测框

## 输入
场景图像+模板

## 输出
物体位置和姿态

## 素材
`assets/`：`scene.png`、`template.png`
