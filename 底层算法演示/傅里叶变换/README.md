# 傅里叶变换

## 需求
频域分析和频域滤波

## 核心函数
cv2.dft(), cv2.idft(), cv2.magnitude()

## 关键步骤
1. 灰度转换
2. 离散傅里叶变换
3. 频谱中心化
4. 频域滤波

## 输入
灰度图像

## 输出
频谱图和滤波结果

## 素材
`assets/`：`texture.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/fourier_transform --radius=60

# Python
cd Python && pip install -r requirements.txt && python fourier_transform.py --radius 60
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-r/--radius` 低通截止半径、`-d/--display` 弹窗。
素材是周期性编织纹理，频谱上能看到成对的亮点。结果默认写入 `output/fourier_transform.png`。
