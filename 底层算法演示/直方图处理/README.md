# 直方图处理

## 需求
计算直方图并进行均衡化和CLAHE增强

## 作用

- **低对比度增强**：把挤在窄区间的灰度拉开。
- **逆光雾霾补偿**：用 CLAHE 局部恢复暗部细节。
- **医学影像增强**：让组织边界更易辨认。
- **输入归一化**：把不同曝光的输入统一到相近分布。

适用边界：均衡化会同时放大噪声，不能替代真正的曝光控制。

## 核心函数
cv2.calcHist(), cv2.equalizeHist(), cv2.createCLAHE()

## 关键步骤
1. 计算直方图
2. 全局均衡化
3. 自适应CLAHE
4. 对比效果

## 输入
灰度/彩色图像

## 输出
增强后图像

## 素材
`assets/`：`haze.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/histogram_processing

# Python
cd Python && pip install -r requirements.txt && python histogram_processing.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出路径、`-l/--clipLimit` CLAHE 限制、`-t/--tileSize` 网格数、`-d/--display` 弹窗。
实测均值/标准差：原图 93.2/57.0 → 全局均衡化 128.6/73.2 → CLAHE 109.3/57.8。
全局均衡化把对比度拉得更狠但会放大噪声，CLAHE 更温和。
