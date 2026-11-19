# 连通域分析

## 需求
标记和分析二值图像中的连通区域

## 作用

- **目标计数**：数出二值图里有多少个独立目标。
- **杂质剔除**：按面积过滤掉噪点与碎屑。
- **焊点检测**：定位每个焊盘并统计面积与位置。
- **版面切分**：把文字块切成独立的行或词。

适用边界：粘连目标会被算成一个，需要先做分离。

## 核心函数
cv2.connectedComponents(), cv2.connectedComponentsWithStats()

## 关键步骤
1. 二值化
2. 连通域标记
3. 计算统计信息
4. 过滤区域

## 输入
二值图像

## 输出
标记图和统计信息

## 素材
`assets/`：`binary.png`（中文段落 + 4 个大小不同的圆点）、`text.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/connected_components --minArea=200 --connectivity=8

# Python
cd Python && pip install -r requirements.txt && python connected_components.py --minArea 200 --connectivity 8
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-a/--minArea` 最小面积过滤、`-c/--connectivity` 4 或 8 邻接、`-d/--display` 弹窗。

`binary.png` 共 267 个连通域（中文字符被拆成大量笔画连通域，面积中位数仅 9），
按面积 ≥200 过滤后只剩 3 个大圆点——正好演示「面积过滤剔除文字与噪点」。
