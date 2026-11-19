# 色彩空间转换

## 需求
在不同颜色空间之间转换并分析通道

## 作用

- **颜色分割**：在 HSV 里按色相分离特定颜色。
- **肤色与植被检测**：利用特定通道的分布特性做筛选。
- **亮度色度分离**：把亮度与颜色拆开分别处理。
- **显示与压缩**：转 YCrCb 后对亮度色度分别采样。

适用边界：转换不增加信息，阈值仍要按具体场景标定。

## 核心函数
cv2.cvtColor()

## 关键步骤
1. 读取图像
2. 转换目标空间
3. 分离通道
4. 分析特性

## 输入
彩色图像

## 输出
各通道图

## 素材
`assets/`：`fruits.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/color_space_conversion

# Python
cd Python && pip install -r requirements.txt && python color_space_conversion.py
```
参数：`-i/--input` 输入图像（缺省 `assets/fruits.png`）、`-o/--output` 输出路径、`-d/--display` 弹窗。
结果默认写入 `output/color_space_conversion.png`，含 HSV / Lab / YCrCb 各通道与 HSV 色相分割结果。
