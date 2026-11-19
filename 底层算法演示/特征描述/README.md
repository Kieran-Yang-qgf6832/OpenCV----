# 特征描述

## 需求
计算关键点描述子用于匹配

## 核心函数
cv2.Feature2D.compute()

## 关键步骤
1. 检测关键点
2. 计算描述子
3. 分析维度
4. 可视化描述子

## 输入
灰度图像+关键点

## 输出
描述子矩阵

## 素材
`assets/`：`church.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/feature_description --maxFeatures=500

# Python
cd Python && pip install -r requirements.txt && python feature_description.py --maxFeatures 500
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-n/--maxFeatures` 特征点上限、`-d/--display` 弹窗。

实测（church.png）：ORB 描述子 500×32（uint8 / 汉明距离），SIFT 描述子 501×128（float32 / 欧氏距离）；
图像旋转 15° 后 ORB 好匹配率 60.8%、SIFT 53.1%——维度更高不等于匹配率更高。
