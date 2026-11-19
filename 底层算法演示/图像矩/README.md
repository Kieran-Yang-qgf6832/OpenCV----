# 图像矩

## 需求
计算几何矩和Hu矩用于形状描述

## 核心函数
cv2.moments(), cv2.HuMoments()

## 关键步骤
1. 计算图像矩
2. 计算质心
3. 计算Hu矩
4. 形状匹配

## 输入
二值图像

## 输出
矩特征

## 素材
`assets/`：`binary.png`、`shapes.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_moments

# Python
cd Python && pip install -r requirements.txt && python image_moments.py
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-a/--minArea` 最小面积、`-d/--display` 弹窗。
实测 3 个轮廓：Hu 矩 log 值 0.70~0.80，`matchShapes` 得分 0.033 与 0.181，可用作形状相似度。
