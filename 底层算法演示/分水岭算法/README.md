# 分水岭算法

## 需求
基于形态学分离接触的物体

## 作用

- **粘连物体分离**：把重叠的硬币、细胞、颗粒逐个分开。
- **细胞计数**：在显微图像里按细胞核数量统计。
- **遥感地块分割**：把相邻的同色区域切成独立地块。
- **颗粒分析**：统计粘连颗粒的个数与单颗面积。

适用边界：对种子点敏感，标记不当会过分割或欠分割。

## 核心函数
cv2.watershed()

## 关键步骤
1. 二值化
2. 距离变换
3. 标记前景背景
4. 应用分水岭

## 输入
灰度图像

## 输出
分割标记图

## 素材
`assets/`：`coins.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/watershed_segmentation --fgRatio=0.4

# Python
cd Python && pip install -r requirements.txt && python watershed_segmentation.py --fgRatio 0.4
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-f/--fgRatio` 前景种子阈值（占最大距离的比例）、`-d/--display` 弹窗。
结果默认写入 `output/watershed_segmentation.png`，终端打印 Otsu 阈值与检测到的物体数。
