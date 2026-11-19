# 图像阈值化

## 需求
全局/自适应/大津阈值分割

## 核心函数
cv2.threshold(), cv2.adaptiveThreshold()

## 关键步骤
1. 灰度转换
2. Otsu全局阈值
3. 自适应阈值
4. 多阈值对比

## 输入
灰度图像

## 输出
二值图像

## 素材
`assets/`：`paper.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_thresholding --block=31 --constant=5

# Python
cd Python && pip install -r requirements.txt && python image_thresholding.py --block 31 --constant 5
```
参数：`-i/--input` 输入图像（缺省 `assets/paper.png`）、`-o/--output` 输出路径、`-b/--block` 自适应邻域尺寸（须为大于 1 的奇数）、`-c/--constant` 常数偏移 C、`-d/--display` 弹窗。
结果默认写入 `output/image_thresholding.png`。素材纸面光照不均，全局阈值（含 Otsu）会整片丢失暗部，自适应阈值能保留。
