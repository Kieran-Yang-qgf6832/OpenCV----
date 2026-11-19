# 各向异性分割

## 需求
各向异性扩散平滑图像同时保留边缘

## 核心函数
cv2.ximgproc.anisotropicDiffusion()

## 关键步骤
1. 设置扩散参数
2. 迭代扩散
3. 保留边缘降噪
4. 对比高斯滤波

## 输入
含噪图像

## 输出
边缘保持的平滑图像

## 素材
`assets/`：`noisy.png`、`smooth.png`

## 运行
```bash
# C++（参数需用 --key=value 形式，需要 opencv_contrib 的 ximgproc）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/anisotropic_diffusion --alpha=0.05 --K=0.1 --niters=10

# Python（需要 opencv-contrib-python，不是 opencv-python）
cd Python && pip install -r requirements.txt && python anisotropic_diffusion.py --alpha 0.05 --K 0.1 --niters 10
```
参数：`-i/--input` 含噪图、`-r/--reference` 干净参考图、`-o/--output` 输出路径、`-a/--alpha` 扩散速率（>0.25 会不稳定）、`-k/--K` 边缘敏感度（归一化尺度，常用 0.05~0.5）、`-n/--niters` 迭代次数、`-s/--sigma` 对照高斯 sigma（0 表示自动匹配噪声水平）、`-d/--display` 弹窗。
结果默认写入 `output/anisotropic_diffusion.png`，终端打印平坦区噪声、梯度 99 分位与 PSNR。
