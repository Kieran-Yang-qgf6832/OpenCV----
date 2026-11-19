# 单应性

## 需求
计算两个平面之间的单应性变换矩阵

## 作用

- **透视矫正**：把斜拍的文档、车牌、二维码拉成正视图。
- **图像拼接**：把相邻两帧 warp 到同一坐标系后融合。
- **平面物体定位**：把模板四角变换过去画出物体边界。
- **AR 贴图**：让虚拟图案随真实平面一起透视变形。
- **位姿估计**：分解 H 得到相机旋转与平面法向。

适用边界：只对平面场景或纯旋转相机成立，有明显深度差会产生视差。

## 核心函数
cv2.findHomography(), cv2.warpPerspective()

## 关键步骤
1. 特征匹配
2. RANSAC估计矩阵
3. 透视变换
4. 图像拼接

## 输入
两组匹配点

## 输出
3x3单应矩阵

## 素材
`assets/`：`poster.png`、`poster_perspective.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/homography

# Python
cd Python && pip install -r requirements.txt && python homography.py
```
参数：`-r/--ref` 参考图、`-s/--scene` 场景图（缺省读 `assets/`）、`-o/--output` 输出路径、`-c/--ratio` Lowe 比率、`-t/--thresh` RANSAC 重投影阈值、`-d/--display` 弹窗。
结果默认写入 `output/homography.png`，并在终端打印匹配数、内点数与归一化后的 H 矩阵。
