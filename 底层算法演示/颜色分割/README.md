# 颜色分割

## 需求
基于颜色空间对特定颜色区域进行提取和分割

## 核心函数
cv2.cvtColor(), cv2.inRange(), cv2.bitwise_and()

## 关键步骤
1. 转换HSV/LAB颜色空间
2. 设定颜色阈值
3. 生成掩码提取目标
4. 形态学优化掩码

## 输入
彩色图像

## 输出
分割掩码和提取结果

## 素材
`assets/`：`apples.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/color_segmentation

# Python
cd Python && pip install -r requirements.txt && python color_segmentation.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出路径、`-n/--minSaturation`、`-v/--minValue` 饱和度/明度下限、`-d/--display` 弹窗。
红色跨色相 0 点，程序用 0~10 与 170~179 两段 `inRange` 合并。实测掩码占比 7.69%，形态学优化后 7.66%。
