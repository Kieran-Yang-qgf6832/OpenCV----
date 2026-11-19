# 周期性噪声

## 需求
检测并去除周期性噪声

## 作用

- **扫描条纹去除**：去掉扫描仪或传感器造成的规律条纹。
- **摩尔纹消除**：处理拍摄屏幕产生的周期干扰。
- **电子干扰抑制**：去掉固定频率的传感器噪声。
- **干扰源分析**：在频域定位干扰的频率特征。

适用边界：陷波半径过大会削掉真实结构，需先确认峰值确为噪声。

## 核心函数
cv2.dft(), 频域掩码

## 关键步骤
1. 傅里叶变换
2. 定位噪声峰值
3. 陷波滤波器
4. 逆变换复原

## 输入
含噪图像

## 输出
去噪图像

## 素材
`assets/`：`periodic_noise.png`、`photo.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/periodic_noise

# Python
cd Python && pip install -r requirements.txt && python periodic_noise.py
```
参数：`-i/--input` 含噪图、`-r/--reference` 干净参考图、`-o/--output` 输出路径、`-p/--peaks` 峰值对数量、`-n/--notchRadius` 陷波半径、`-d/--display` 弹窗。
实测检出峰值对 (427,597)、(462,562) 等，PSNR 22.92 → 32.91 dB。
**注意**：低频屏蔽半径必须足够大（本实现用 40 px），否则会把图像自身结构当成噪声峰，陷波后反而更差（曾降到 27.61 dB）。
