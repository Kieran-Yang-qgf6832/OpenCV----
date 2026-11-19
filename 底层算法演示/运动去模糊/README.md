# 运动去模糊

## 需求
复原运动模糊图像

## 作用

- **抖动画面补救**：恢复相机移动造成的拖影。
- **车载无人机清晰化**：缓解平台运动带来的模糊。
- **车牌可读性提升**：把带拖影的车牌复原到可辨识。
- **识别前处理**：先去模糊再识别以提高命中率。

适用边界：需要估计运动方向与长度，估偏会让结果比不处理更差。

## 核心函数
自定义运动模糊核+维纳滤波

## 关键步骤
1. 模拟运动模糊
2. 估计运动核
3. 逆滤波复原
4. 复原图像

## 输入
运动模糊图像

## 输出
复原图像

## 素材
`assets/`：`motion_blurred.png`、`sharp.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/motion_deblur

# Python
cd Python && pip install -r requirements.txt && python motion_deblur.py
```
参数：`-i/--input` 模糊图、`-r/--reference` 清晰参考图、`-o/--output` 输出路径、`-k/--K` 噪声功率比、`-d/--display` 弹窗。
采用两阶段扫描（粗扫角度步长 15°/长度 10-30，再在最优附近细化）。
实测最优核约 165°~170°/长度 20，PSNR 19.08 → 20.05 dB（C++）/ 21.39 dB（Python）。
**注意**：单级粗扫的量化误差会让复原结果比不处理还差（18.56 dB），必须细化。
