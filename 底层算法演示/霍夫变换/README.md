# 霍夫变换

## 需求
检测图像中的直线和圆等几何形状

## 核心函数
cv2.HoughLines(), cv2.HoughCircles()

## 关键步骤
1. 边缘检测
2. 霍夫空间累加
3. 峰值检测
4. 绘制形状

## 输入
边缘图像

## 输出
直线/圆参数

## 素材
`assets/`：`railway.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/hough_transform --linesThresh=400 --pThresh=180

# Python
cd Python && pip install -r requirements.txt && python hough_transform.py --linesThresh 400 --pThresh 180
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-t/--linesThresh` 标准霍夫累加器阈值、`-p/--pThresh` 概率霍夫累加器阈值、`-d/--display` 弹窗。
结果默认写入 `output/hough_transform.png`。素材只有直线，`HoughCircles` 需要另备含圆形的图像。
