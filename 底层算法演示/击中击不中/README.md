# 击中击不中

## 需求
使用击中-击不中变换检测特定形状

## 作用

- **图案检索**：在二值图里定位固定尺寸的孤立形状。
- **字符缺陷检测**：找出缺笔或多点的印刷字符。
- **版式校验**：确认某个标记是否出现在规定位置。
- **噪声剔除**：只保留尺寸符合预期的目标。

适用边界：要求形状尺寸严格一致，轻微形变或缩放就会漏检。

## 核心函数
cv2.morphologyEx(MORPH_HITMISS)

## 关键步骤
1. 定义击中/击不中核
2. 应用变换
3. 定位匹配
4. 检测角点端点

## 输入
二值图像

## 输出
匹配位置标记

## 素材
`assets/`：`binary.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/hit_or_miss --small=17 --large=35

# Python
cd Python && pip install -r requirements.txt && python hit_or_miss.py --small 17 --large 35
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-s/--small` 小方块边长、`-l/--large` 大方块边长、`-d/--display` 弹窗。
素材由 35x35 方块、17x17 方块、59x9 横杠各 25 个组成，两个核各命中 25 处。结果默认写入 `output/hit_or_miss.png`。
