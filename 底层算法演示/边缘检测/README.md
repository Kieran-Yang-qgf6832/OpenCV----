# 边缘检测

## 需求
使用Sobel、Laplacian和Canny算子检测图像边缘

## 作用

- **缺陷检测**：提取划痕、裂纹、毛刺的边界位置。
- **车道线提取**：给霍夫变换提供候选边缘点。
- **尺寸测量**：用连续边缘拟合轮廓再折算实际长度。
- **版面分析**：定位表格框线与文字区域边界。

适用边界：对噪声敏感，必须先降噪；光照突变会产生成片伪边缘。

## 核心函数
cv2.Sobel(), cv2.Laplacian(), cv2.Canny()

## 关键步骤
1. 灰度转换
2. 高斯降噪
3. 计算梯度幅值
4. 双阈值处理

## 输入
灰度或彩色图像

## 输出
边缘图像

## 素材
`assets/`：`scene.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/edge_detection                      # 图片模式：输出四宫格对比图
./build/edge_detection --input=../../1789298663603..mp4   # 视频模式：写三段 mp4

# Python（已有的视频版脚本，输出到本目录）
cd Python && pip install -r requirements.txt && python edge_detection.py
```
参数：`-i/--input` 图片或视频、`-o/--output` 输出目录、`-l/--lowThresh`、`-u/--highThresh` 为 Canny 双阈值、`-d/--display` 弹窗。

C++ 版按输入自动切换模式：图片输入输出 `output/edge_detection.png`（灰度/Sobel/Laplacian/Canny 四宫格），
视频输入输出 `output/sobel.mp4`、`laplacian.mp4`、`canny.mp4`。
