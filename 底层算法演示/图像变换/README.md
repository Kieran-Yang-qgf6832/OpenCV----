# 图像变换

## 需求
对数变换、伽马变换等非线性变换

## 作用

- **对比度增强**：用 Gamma 与拉伸改善过暗过亮画面。
- **显示适配**：把高位深图像映射到可显示的 8 位。
- **传感器校正**：用 LUT 补偿相机的非线性响应。
- **批量预处理**：把不同曝光的输入统一到同一范围。

适用边界：单调映射不增加信息，过曝区域拉伸后仍是死白。

## 核心函数
cv2.LUT()

## 关键步骤
1. 构建查找表
2. 对数变换
3. 伽马变换
4. 对比度拉伸

## 输入
灰度图像

## 输出
变换后图像

## 素材
`assets/`：`portrait.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_transform --gamma=0.4

# Python
cd Python && pip install -r requirements.txt && python image_transform.py --gamma 0.4
```
参数：`-i/--input` 输入图像（缺省 `assets/portrait.png`）、`-o/--output` 输出路径、`-g/--gamma` 伽马主参数、`-d/--display` 弹窗。
结果默认写入 `output/image_transform.png`，图中按 原图 / 对数 / 伽马 / 恒等 / 反伽马 / 百分位拉伸 六宫格排列。
