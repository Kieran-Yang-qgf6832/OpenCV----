# 随机数与文本

## 需求
生成随机数并绘制随机文本图形

## 作用

- **测试图生成**：造随机噪声图验证算法鲁棒性。
- **合成数据**：生成标注坐标已知的测试样本。
- **结果标注**：把参数与指标写到输出图上。
- **可视化调试**：随机配色区分不同类别或实例。

适用边界：cv::RNG 与 numpy 序列不同，跨语言结果不一致。

## 核心函数
cv2.randu(), cv2.putText()

## 关键步骤
1. 创建RNG对象
2. 生成随机数
3. 绘制随机文本
4. 随机图形绘制

## 输入
随机种子

## 输出
随机生成图像

## 素材
`assets/`：`canvas.png`（纯白画布）

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/random_and_text --seed=20261119 --count=40

# Python
cd Python && pip install -r requirements.txt && python random_and_text.py --seed 20261119 --count 40
```
参数：`-i/--input` 画布素材、`-o/--output` 输出路径、`-s/--seed` 随机种子、`-n/--count` 图形与文本数量、`-d/--display` 弹窗。

种子相同则**同一实现内**结果完全可复现；C++ 的 `cv::RNG` 与 numpy 的 PCG64 算法不同，
两种语言的随机序列并不相同（分布统计量一致：均匀分布均值≈128、标准差≈74；正态≈128/40）。
结果默认写入 `output/random_and_text.png`，含均匀噪声、正态噪声、随机图形与随机文本四格。
