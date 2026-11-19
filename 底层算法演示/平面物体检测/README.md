# 平面物体检测

## 需求
在场景中检测平面物体并估计姿态

## 作用

- **海报封面识别**：在场景里框出目标并给出姿态。
- **AR 锚定**：以识别到的平面建立虚拟物体坐标系。
- **货架盘点**：定位包装正面并矫正成正视图。
- **展板导览**：识别展板后在画面上叠加说明。

适用边界：目标需有足够纹理，且必须基本呈平面。

## 核心函数
cv2.findHomography(), cv2.perspectiveTransform()

## 关键步骤
1. 特征检测匹配
2. 计算单应性
3. 分解姿态
4. 绘制检测框

## 输入
场景图像+模板

## 输出
物体位置和姿态

## 素材
`assets/`：`scene.png`、`template.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/planar_object_detection --features=2000 --ratio=0.75

# Python
cd Python && pip install -r requirements.txt && python planar_object_detection.py --features 2000 --ratio 0.75
```
参数：`-s/--scene` 场景图、`-t/--template` 模板图、`-o/--output` 输出路径、`-n/--features` ORB 特征上限、`-r/--ratio` Lowe 比率、`-d/--display` 弹窗。

结果默认写入 `output/planar_object_detection.png`。实测 321 组好匹配、309 内点（96.3%），
`decomposeHomographyMat` 给出 4 组解。**注意**：无相机标定数据时使用「焦距 ≈ 1.2×长边」的虚拟内参，
旋转较可靠，平移量只有相对意义。
