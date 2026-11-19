# 图像滤波

## 需求
对图像进行平滑、锐化、降噪等滤波处理

## 作用

- **去噪预处理**：在检测前压掉传感器噪声。
- **保边平滑**：双边滤波在降噪时保住边缘。
- **椒盐噪声清除**：中值滤波对孤立黑白点最有效。
- **锐化增强**：用非锐化掩模突出细节。

适用边界：滤波都在丢失信息，平滑力度要靠下游效果反推。

## 核心函数
cv2.GaussianBlur(), cv2.medianBlur(), cv2.bilateralFilter()

## 关键步骤
1. 选择滤波器类型
2. 设定核大小
3. 卷积运算
4. 对比效果

## 输入
原始图像

## 输出
滤波后图像

## 素材
`assets/`：`texture.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_filtering

# Python
cd Python && pip install -r requirements.txt && python image_filtering.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出路径、`-k/--kernel` 核大小（奇数）、`-s/--sigma` 高斯标准差、`-e/--sigmaColor` 双边颜色 sigma、`-d/--display` 弹窗。
锐度（拉普拉斯方差）实测：原图 9774 → 加噪 34851 → 高斯 424 → 中值 691 → 双边 2342 → 锐化 24155。
双边滤波保留了更多结构（2342），远高于高斯的 424，这就是「保边」的量化表现。
