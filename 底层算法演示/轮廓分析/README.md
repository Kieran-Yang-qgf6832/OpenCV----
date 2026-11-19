# 轮廓分析

## 需求
检测并分析物体轮廓，计算面积周长等属性

## 核心函数
cv2.findContours(), cv2.contourArea(), cv2.boundingRect()

## 关键步骤
1. 二值化
2. 查找轮廓
3. 遍历计算属性
4. 绘制轮廓

## 输入
二值图像

## 输出
轮廓列表和可视化

## 素材
`assets/`：`binary.png`、`shapes.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/contour_analysis --minArea=500 --epsilon=0.02

# Python
cd Python && pip install -r requirements.txt && python contour_analysis.py --minArea 500 --epsilon 0.02
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-a/--minArea` 最小面积过滤、`-e/--epsilon` 多边形近似精度（占周长比例）、`-d/--display` 弹窗。
结果默认写入 `output/contour_analysis.png`，终端打印每个轮廓的面积、周长、圆度与近似顶点数。
