# AKAZE匹配

## 需求
使用AKAZE描述子进行特征匹配

## 作用

- **图像配对**：判断两张图是否拍到同一场景。
- **重复图检索**：用匹配数量衡量两图相似程度。
- **拼接前置**：为单应估计提供可靠同名点。
- **变化检测**：对齐同机位不同时刻的两张图。

适用边界：低纹理或强重复纹理下，比率测试会筛掉大量真匹配。

## 核心函数
cv2.AKAZE_create(), cv2.BFMatcher()

## 关键步骤
1. 检测关键点和描述子
2. BFMatcher匹配
3. 比率测试筛选
4. 可视化

## 输入
两幅图像

## 输出
匹配点对

## 素材
`assets/`：`scene.png`、`scene_rotated.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/akaze_matching

# Python
cd Python && pip install -r requirements.txt && python akaze_matching.py
```
参数：`-a/--imageA`、`-b/--imageB` 两幅图像、`-o/--output` 输出路径、`-r/--ratio` 比率阈值、`-d/--display` 弹窗。
实测关键点 A 5040 / B 3879，描述子 61 字节；knn 首选 5040 组，比率 0.75 筛选后 1956 组（38.8%）。
