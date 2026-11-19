# 形态学线条检测

## 需求
使用形态学操作检测水平和垂直线条

## 作用

- **表格线提取**：把表格的横线与竖线分开取出。
- **版面结构分析**：区分文字块与分隔线。
- **车道线增强**：拉长并连通断续的标线。
- **倾斜角估计**：由长直线的方向反推文档倾斜。

适用边界：只对方向明确的直线有效，倾斜线需用旋转核或改走霍夫。

## 核心函数
cv2.morphologyEx(), cv2.getStructuringElement()

## 关键步骤
1. 二值化
2. 创建方向结构元素
3. 开运算提取线条
4. 轮廓筛选

## 输入
灰度图像

## 输出
检测到的线条

## 素材
`assets/`：`table.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/line_detection_morphology

# Python
cd Python && pip install -r requirements.txt && python line_detection_morphology.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出路径、`-l/--lineLen` 结构元素长度、`-n/--minLen` 最小线长、`-d/--display` 弹窗。
自动取结构元素长度 40、最小线长 51；实测提取水平线 14 段、垂直线 16 段。
