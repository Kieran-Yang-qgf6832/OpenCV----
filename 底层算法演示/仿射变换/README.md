# 仿射变换

## 需求
对图像进行旋转、缩放、平移等变换

## 作用

- **图像对齐**：把旋转或缩放后的图摆正回标准姿态。
- **数据增广**：训练前做随机旋转、缩放、平移扩样本。
- **ROI 提取**：把倾斜的车牌或标签旋到水平再切出。
- **坐标映射**：把一组点批量换算到另一套坐标系。

适用边界：只能表达平移、旋转、缩放、剪切，处理不了透视变形。

## 核心函数
cv2.getAffineTransform(), cv2.warpAffine()

## 关键步骤
1. 定义对应点
2. 计算仿射矩阵
3. 应用变换
4. 插值处理

## 输入
原始图像

## 输出
变换后图像

## 素材
`assets/`：`scene.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/affine_transform --angle=25 --scale=0.85

# Python
cd Python && pip install -r requirements.txt && python affine_transform.py --angle 25 --scale 0.85
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-a/--angle` 旋转角度、`-s/--scale` 旋转时的缩放、`-d/--display` 弹窗。
结果默认写入 `output/affine_transform.png`，包含旋转/缩放/平移/三点剪切/链式变换与两种插值方式的对比。
