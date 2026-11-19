# 角点检测

## 需求
Harris和Shi-Tomasi角点检测

## 作用

- **特征跟踪**：为光流或匹配挑出稳定的跟踪点。
- **相机标定**：在棋盘格上检出角点完成标定。
- **图像配准**：用角点对齐两张有微小位移的图像。
- **场景分析**：用角点密度区分纹理丰富与平坦区域。

适用边界：尺度变化和模糊会显著减少角点，密集纹理容易过检。

## 核心函数
cv2.cornerHarris(), cv2.goodFeaturesToTrack()

## 关键步骤
1. 灰度转换
2. 计算角点响应
3. 阈值筛选
4. 非极大值抑制

## 输入
灰度图像

## 输出
角点位置

## 素材
`assets/`：`brick.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/corner_detection --harrisThresh=0.5

# Python
cd Python && pip install -r requirements.txt && python corner_detection.py --harrisThresh 0.5
```
参数：`-i/--input` 输入图像、`-o/--output` 输出路径、`-t/--harrisThresh` Harris 相对阈值（占归一化响应比例）、`-m/--maxCorners` Shi-Tomasi 角点上限、`-q/--minDistance` 最小间距、`-d/--display` 弹窗。
砖墙纹理极密，Harris 的 NMS 窗口取 9x9、相对阈值 0.5 才能压到百级数量。结果默认写入 `output/corner_detection.png`。
