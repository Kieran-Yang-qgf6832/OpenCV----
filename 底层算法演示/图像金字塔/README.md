# 图像金字塔

## 需求
构建高斯和拉普拉斯金字塔

## 作用

- **多尺度检测**：大尺度找目标，小尺度提精度。
- **光流金字塔**：让 LK 能处理超出窗口的大位移。
- **图像融合**：按层融合不同曝光的图像。
- **尺度空间构造**：生成 SIFT 等算法所需的尺度序列。

适用边界：pyrUp 无法恢复已丢弃的高频，需靠拉普拉斯层补偿。

## 核心函数
cv2.pyrDown(), cv2.pyrUp()

## 关键步骤
1. 高斯模糊降采样
2. 上采样重建
3. 拉普拉斯计算
4. 图像融合

## 输入
原始图像

## 输出
金字塔层级图像

## 素材
`assets/`：`skyline.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_pyramid --levels=3

# Python
cd Python && pip install -r requirements.txt && python image_pyramid.py --levels 3
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-l/--levels` 金字塔层数（不含第 0 层）、`-d/--display` 弹窗。

输出六宫格：原图 / level1 / level2 / 拉普拉斯层 / 重建图 / 残差（放大 8 倍）。
实测重建 PSNR ≈ 97.5 dB，说明拉普拉斯金字塔可近无损重建。
**注意**：上采样在计算拉普拉斯与在重建阶段必须用同一精度（都走浮点或都走整数），
否则两处的舍入差异会把 PSNR 从 97 dB 拉到 56 dB。
