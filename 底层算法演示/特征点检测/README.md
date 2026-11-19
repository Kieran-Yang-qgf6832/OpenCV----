# 特征点检测

## 需求
使用ORB/SIFT/FAST检测图像关键点

## 作用

- **跟踪初始化**：为光流挑选质量高的起始点。
- **纹理丰富度判断**：用关键点数量判断能否走特征法。
- **拼接预处理**：确认两图都有足够特征可供匹配。
- **算法选型**：按实时性要求在 ORB、FAST、SIFT 间权衡。

适用边界：只检测不描述，无法直接用于匹配与识别。

## 核心函数
cv2.ORB_create(), cv2.SIFT_create(), cv2.FastFeatureDetector_create()

## 关键步骤
1. 创建检测器
2. 检测关键点
3. 绘制关键点
4. 对比算法性能

## 输入
灰度图像

## 输出
关键点坐标和可视化

## 素材
`assets/`：`books.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/keypoint_detection --maxFeatures=1000

# Python
cd Python && pip install -r requirements.txt && python keypoint_detection.py --maxFeatures 1000
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-n/--maxFeatures` 特征点上限、`-d/--display` 弹窗。

实测（books.png）：ORB 1000 点 / 15 ms，SIFT 1000 点 / 220 ms，FAST 11254 点 / 3 ms。
FAST 数量最多但无尺度与方向信息，SIFT 最慢但具备多尺度不变性。
