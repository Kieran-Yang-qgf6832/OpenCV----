# 直方图处理

## 需求
计算直方图并进行均衡化和CLAHE增强

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
