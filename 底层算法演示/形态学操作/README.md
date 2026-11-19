# 形态学操作

## 需求
对二值图像进行腐蚀、膨胀、开闭运算等处理

## 核心函数
cv2.erode(), cv2.dilate(), cv2.morphologyEx()

## 关键步骤
1. 定义结构元素
2. 腐蚀去噪
3. 膨胀填充
4. 开闭运算组合

## 输入
二值图像

## 输出
处理后图像

## 素材
`assets/`：`binary.png`、`text.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/morphology

# Python
cd Python && pip install -r requirements.txt && python morphology.py
```
参数：`-i/--input` 二值图、`-o/--output` 输出路径、`-k/--kernel` 结构元素尺寸、`-s/--shape` rect/ellipse/cross、`-d/--display` 弹窗。
前景像素数实测：原图 54551 → 腐蚀 1161 → 膨胀 162281 → 开运算 5093 → 闭运算 87785 → 梯度 161120。
腐蚀让前景大幅收缩、膨胀大幅扩张，开闭运算分别对应「去噪点」与「填孔洞」。
