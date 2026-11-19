# 模板匹配

## 需求
在图像中搜索模板，支持多尺度匹配

## 核心函数
cv2.matchTemplate(), cv2.minMaxLoc()

## 关键步骤
1. 读取模板和目标
2. 选择匹配方法
3. 滑动窗口匹配
4. 定位最佳位置

## 输入
目标图像+模板

## 输出
匹配位置和热力图

## 素材
`assets/`：`scene.png`、`template.png`

> 实测模板是场景在 (359,359) 处的精确裁剪（`TM_CCOEFF_NORMED` 峰值 1.0000），可直接作为定位真值。

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/template_matching --minScale=0.5 --maxScale=1.5 --steps=11

# Python
cd Python && pip install -r requirements.txt && python template_matching.py --minScale 0.5 --maxScale 1.5 --steps 11
```
参数：`-s/--scene` 场景图、`-t/--template` 模板图、`-o/--output` 输出路径、`-a/--minScale`、`-b/--maxScale` 缩放范围、`-n/--steps` 扫描点数、`-d/--display` 弹窗。

结果默认写入 `output/template_matching.png`，含检测框、响应热力图与匹配到的图块。
实测尺度 1.00 峰值 1.0000，尺度 0.5 与 1.5 时峰值分别降到 0.76 与 0.63——多尺度扫描的必要性一目了然。
