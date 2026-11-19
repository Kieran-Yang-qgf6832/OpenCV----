# 几何绘制

## 需求
在图像上绘制基本几何形状和文字

## 作用

- **结果标注**：在检测图上画框、箭头与文字说明。
- **数据可视化**：把轨迹、区域、统计值画到画面上。
- **调试叠加**：把中间结果叠回原图逐项核对。
- **版面拼接**：用空白画布拼多张结果图生成对照图。

适用边界：只负责绘制不做分析，亚像素标注需用 shift 定点数。

## 核心函数
cv2.line(), cv2.circle(), cv2.rectangle(), cv2.putText()

## 关键步骤
1. 创建画布
2. 绘制几何形状
3. 设置样式
4. 显示结果

## 输入
空白画布或图像

## 输出
绘制后图像

## 素材
`assets/`：`canvas.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/geometric_drawing --display=1

# Python
cd Python && pip install -r requirements.txt && python geometric_drawing.py --display
```
参数：`-i/--input` 画布素材（缺省 `assets/canvas.png`）、`-o/--output` 输出路径、`-w/-H` 空白画布尺寸、`-d/--display` 弹窗。
结果默认写入 `output/geometric_drawing.png`。
