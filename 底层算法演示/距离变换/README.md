# 距离变换

## 需求
计算像素到最近背景像素的距离

## 核心函数
cv2.distanceTransform()

## 关键步骤
1. 二值化
2. 计算距离变换
3. 归一化显示
4. 结合分水岭

## 输入
二值图像

## 输出
距离图

## 素材
`assets/`：`binary.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/distance_transform --seedRatio=0.5

# Python
cd Python && pip install -r requirements.txt && python distance_transform.py --seedRatio 0.5
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-s/--seedRatio` 前景种子阈值（占最大距离比例）、`-d/--display` 弹窗。
结果默认写入 `output/distance_transform.png`。素材为白底黑形状，程序内部会取反。实测最大距离 L1=121、L2=118.44、C=101，按 0.5*max 可提出 3 个物体种子。
