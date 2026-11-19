# 线性变换

## 需求
线性灰度变换调整对比度和亮度

## 作用

- **对比度调整**：用增益与偏置拉开灰度分布。
- **亮度校正**：给整幅图加上固定偏置。
- **传感器线性化**：用 alpha、beta 补偿响应曲线。
- **交互调参**：用滑动条找到合适参数再固化。

适用边界：`convertScaleAbs` 取绝对值，负增益会失真，需用截断版。

## 核心函数
cv2.convertScaleAbs()

## 关键步骤
1. 设定增益和偏置
2. 逐像素变换
3. 裁剪有效范围
4. 实时调节

## 输入
灰度图像

## 输出
变换后图像

## 素材
`assets/`：`low_contrast.png`、`portrait.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/linear_transform --alpha=1.6 --beta=0 --display=1

# Python
cd Python && pip install -r requirements.txt && python linear_transform.py --alpha 1.6 --beta 0 --display
```
参数：`-i/--input` 输入图像（缺省 `assets/low_contrast.png`）、`-r/--reference` 参考图像（缺省 `assets/portrait.png`）、`-o/--output` 输出路径、`-a/--alpha` 增益、`-b/--beta` 偏置、`-d/--display` 弹窗并启用滑动条实时调节。
结果默认写入 `output/linear_transform.png`。`convertScaleAbs` 语义是 `|alpha*x+beta|`，负增益会因取绝对值失真，输出图最后一格给出严格截断版本作对照。
