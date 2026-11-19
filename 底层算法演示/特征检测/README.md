# 特征检测

## 需求
检测关键点并对比多种检测器

## 核心函数
cv2.Feature2D.detect()

## 关键步骤
1. 创建不同检测器
2. 检测关键点
3. 对比数量分布
4. 评估重复性

## 输入
灰度图像

## 输出
关键点列表

## 素材
`assets/`：`building.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/feature_detection --maxFeatures=800 --angle=5

# Python
cd Python && pip install -r requirements.txt && python feature_detection.py --maxFeatures 800 --angle 5
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-n/--maxFeatures` 特征点上限、`-a/--angle` 重复性评估的旋转角、`-t/--tolerance` 重复判定距离（像素）、`-d/--display` 弹窗。

重复率定义：把图像旋转 θ 后重新检测，所有点逆变换回原坐标系，若 3 像素内存在对应点即视为重复。
实测（building.png，旋转 5°）：ORB 800 点/89.9%，SIFT 800 点/72.0%，AKAZE 3963 点/93.7%，BRISK 13381 点/82.0%。
