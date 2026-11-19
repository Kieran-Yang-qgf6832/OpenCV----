# 角点亚像素

## 需求
将角点检测精度提升到亚像素级别

## 核心函数
cv2.cornerSubPix()

## 关键步骤
1. 粗检测角点
2. 设定搜索窗口
3. 亚像素精化
4. 对比精度

## 输入
灰度图像+初始角点

## 输出
亚像素角点坐标

## 素材
`assets/`：`checkerboard.png`

> **素材说明**：棋盘格由 4× 超采样后面积下采样生成，边缘有真实灰度过渡（256 级，
> 过渡像素约 30%），整幅图经亚像素级透视变换，内部角点真值均落在非整数像素上。
> 实测 49 个角点：整数角点相对真值平均误差 0.396 px，cornerSubPix 精化后降至
> 0.080 px（中位 0.081、最大 0.121），修正量互不相同（x 方向标准差 0.356 px），
> 可直接演示「像素级 → 亚像素级」的提升。项目脚本另做 ±1.5 px 抖动初值收敛检验，
> 实测收敛差 0.0001 px。

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/corner_subpixel --quality=0.40 --minDistance=60 --winSize=5

# Python
cd Python && pip install -r requirements.txt && python corner_subpixel.py --quality 0.40 --minDistance 60 --winSize 5
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-n/--maxCorners` 粗检测上限、`-q/--quality` 角点质量阈值（0.40 时正好得到上面用于标定的 49 个角点）、`-m/--minDistance` 角点最小间距、`-w/--winSize` 搜索窗口半径、`-j/--jitter` 收敛性检验的抖动幅度、`-d/--display` 弹窗。
结果默认写入 `output/corner_subpixel.png`，含整数角点、亚像素角点与 12 倍放大对照。

程序输出三个量化指标（默认参数实测）：

1. **整数 → 亚像素位移**：均值 0.859 px、最大 1.294 px，说明粗检测确实偏离真实角点。
2. **抖动收敛误差**：初值加 ±1.5 px 随机抖动后精化，与不抖动结果差 0.00015 px（最大 0.00098），说明解稳定。
3. **梯度正交残差**：`cornerSubPix` 的目标函数值，从 190626 降到 83526（改善 56.2%）。
