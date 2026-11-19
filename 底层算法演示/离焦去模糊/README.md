# 离焦去模糊

## 需求
使用维纳滤波复原离焦模糊图像

## 核心函数
自定义维纳滤波

## 关键步骤
1. 模拟离焦模糊
2. 估计点扩散函数
3. 维纳滤波复原
4. 评估质量

## 输入
模糊图像

## 输出
复原图像

## 素材
`assets/`：`blurred.png`、`sharp.png`

> 实测 `blurred.png` 由 `sharp.png` 经 σ≈2.5 的高斯模糊生成（以 σ=2.5 高斯模糊 sharp 后与 blurred
> 的 PSNR 达 56.9 dB），因此可用 σ 扫描反推真实离焦半径。

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/deblur_wiener

# Python
cd Python && pip install -r requirements.txt && python deblur_wiener.py
```
参数：`-i/--input` 模糊图像、`-r/--reference` 清晰参考图、`-o/--output` 输出路径、`-s/--sigma` PSF 标准差（0=自动搜索）、`-k/--K` 噪声功率比（0=自动搜索）、`-d/--display` 弹窗。

默认会扫描 σ∈{1.5,2,2.5,3,4} × K∈{1e-4,1e-2,1e-1}，取 PSNR 最高者。实测最优 σ=2.5、K=0.01：
PSNR 34.65 → **36.82 dB**；而 K 取 1e-5 时噪声被放大，PSNR 掉到 21.43 dB。
