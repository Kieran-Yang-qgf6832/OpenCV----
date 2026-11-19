# 点多边形测试

## 需求
判断点与多边形的位置关系

## 核心函数
cv2.pointPolygonTest()

## 关键步骤
1. 定义多边形
2. 测试多个点
3. 计算距离
4. 可视化分类

## 输入
轮廓+测试点

## 输出
位置关系和距离

## 素材
`assets/`：`binary.png`、`shapes.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/point_polygon_test --grid=40

# Python
cd Python && pip install -r requirements.txt && python point_polygon_test.py --grid 40
```
参数：`-b/--binary` 二值图、`-s/--shapes` 叠加用彩色图、`-o/--output` 输出路径、`-g/--grid` 测试点网格间距、`-p/--step` 距离场采样步长、`-d/--display` 弹窗。
素材为白底黑形状，程序内部会取反后再提取轮廓。结果默认写入 `output/point_polygon_test.png`。
