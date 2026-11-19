# 背景减除

## 需求
从视频序列中分离前景运动物体

## 作用

- **固定机位监控**：从画面里分离出走动的人或车。
- **客流与车流统计**：给后续跟踪计数提供干净的二值掩膜。
- **周界入侵报警**：只在出现运动目标时才唤醒后续识别。
- **产线监控**：检出传送带上缺失或错位的工件。

适用边界：要求相机静止、光照稳定；相机一动整幅都会判成前景。

## 核心函数
cv2.createBackgroundSubtractorMOG2(), cv2.createBackgroundSubtractorKNN()

## 关键步骤
1. 创建背景减除器
2. 逐帧获取前景
3. 形态学优化
4. 提取目标

## 输入
视频序列（固定机位）

> **为什么必须是视频而不是单张图片**：背景减除是时间维建模——算法为每个像素维护一组高斯分布，
> 随着帧不断到来才逐步"学会"背景。单帧调用时背景模型尚不存在，整幅画面会被判为前景；
> 单张图跑完一次更新后模型退化成"这一帧自身"，再相减得到的是空掩膜。
> OpenCV 官方教程也把输入定义为视频文件（`-vid`）或图像序列（`-img`），二者都要求是**连续帧**。

## 输出
前景掩码（逐帧二值视频）

## 素材
`assets/`：`car_passing.mp4`（1280×720 / 24 fps / 241 帧 / 10 秒，固定机位拍摄的空旷道路，
红色轿车在前段空场景之后驶过画面，正好演示「背景模型先收敛、后检出目标」的完整过程）

## 运行
```bash
# C++（参数需用 --key=value 形式，且必须 Release 构建）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/background_subtraction
./build/background_subtraction --method=KNN --display=1

# Python
cd Python && pip install -r requirements.txt && python background_subtraction.py
python background_subtraction.py --method KNN --display
```
参数：`-i/--input` 输入视频（缺省 `assets/car_passing.mp4`）、`-m/--method` 算法 `MOG2` 或 `KNN`、
`-o/--output` 输出视频路径、`-d/--display` 弹窗实时显示。

结果默认写入 `output/mask_MOG2.mp4`（选 KNN 时为 `mask_KNN.mp4`）。

## 实现要点
- **去阴影**：两个减除器都开启 `detectShadows`，阴影被标为 127、确定前景为 255，
  因此 `refineMask` 先用 `threshold(mask, 200, 255)` 丢掉阴影，只保留真实目标。
- **形态学优化**：3×3 椭圆核开运算去孤立噪点，闭运算（2 次迭代）填补目标内部空洞。
- **面积过滤**：轮廓面积小于 500 像素的视为噪声。
