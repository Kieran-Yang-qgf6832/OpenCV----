# 几何绘制

## 需求
在图像上绘制基本几何形状和文字

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
