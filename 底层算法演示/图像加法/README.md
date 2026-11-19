# 图像加法

## 需求
实现图像叠加、混合和加权融合

## 作用

- **多帧平均降噪**：同一静态场景多张相加后取均值。
- **水印叠加**：按权重把标识混到原图上。
- **多重曝光合成**：把两张不同曝光的图合成一张。
- **亮度补偿**：给偏暗图像整体加一个偏置。

适用边界：`+` 会回绕到 0，必须用饱和的 add 或 addWeighted。

## 核心函数
cv2.add(), cv2.addWeighted()

## 关键步骤
1. 读取两幅图像
2. 直接相加
3. 加权混合
4. ROI融合

## 输入
两幅图像

## 输出
融合图像

## 素材
`assets/`：`image_a.png`、`image_b.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_addition --alpha=0.5

# Python
cd Python && pip install -r requirements.txt && python image_addition.py --alpha 0.5
```
参数：`-a/--imageA`、`-b/--imageB` 两幅输入（缺省读 `assets/`）、`-o/--output` 输出路径、`--alpha` 第一幅权重、`-d/--display` 弹窗。
结果默认写入 `output/image_addition.png`。注意 Python 必须用 `cv2.add`，numpy 的 `uint8` 相加会取模回绕而不是饱和。
