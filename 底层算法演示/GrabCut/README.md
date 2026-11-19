# GrabCut

## 需求
交互式前景提取，通过矩形框分割前景

## 核心函数
cv2.grabCut()

## 关键步骤
1. 定义初始矩形
2. 迭代优化模型
3. 涂鸦修正
4. 提取前景

## 输入
彩色图像+矩形框

## 输出
前景分割结果

## 素材
`assets/`：`subject.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/grabcut

# Python
cd Python && pip install -r requirements.txt && python grabcut.py
```
参数：`-i/--input` 输入图、`-o/--output` 输出路径、`-m/--margin` 矩形留白比例、`-n/--iters` 迭代次数、`-d/--display` 弹窗。
矩形由留白比例自动生成（默认 10%），实测前景占比 26.44%。
