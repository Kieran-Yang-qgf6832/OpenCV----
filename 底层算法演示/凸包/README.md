# 凸包

## 需求
计算点集的凸包用于形状分析

## 核心函数
cv2.convexHull(), cv2.convexityDefects()

## 关键步骤
1. 提取轮廓点集
2. 计算凸包
3. 计算凸缺陷
4. 可视化

## 输入
轮廓点集

## 输出
凸包和凸缺陷

## 素材
`assets/`：`binary.png`（16 个不规则石块轮廓）、`shapes.png`

> 该素材的凹陷较浅（实心度 ≈0.95），凸缺陷深度中位数只有 0.63 像素、最大 3.35 像素，
> 因此默认阈值取 `minDepth=1.0`；若用 5 像素会把所有缺陷都滤掉。

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/convex_hull --minArea=500 --minDepth=1.0

# Python
cd Python && pip install -r requirements.txt && python convex_hull.py --minArea 500 --minDepth 1.0
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-a/--minArea` 最小面积过滤、`-d/--minDepth` 凸缺陷最小深度（像素）、`-w/--display` 弹窗。

`convexityDefects` 返回的深度是定点数，代码里已除以 256 转成像素。实测 16 个轮廓、深度 ≥1 的缺陷共 91 个。
