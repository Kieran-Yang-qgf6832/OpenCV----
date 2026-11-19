# 图像修复

## 需求
修复图像中的划痕和损坏区域

## 作用

- **划痕修复**：去掉老照片与扫描件上的划痕污点。
- **遮挡去除**：抹掉画面里不需要的水印或文字。
- **坏点修补**：补上传感器固定位置的缺像素。
- **目标移除**：把物体抹掉后填补背景。

适用边界：只适合小面积缺损，大洞会被填成糊状。

## 核心函数
cv2.inpaint()

## 关键步骤
1. 创建损坏掩码
2. 选择修复算法
3. 执行修复
4. 评估质量

## 输入
损坏图像+掩码

## 输出
修复后图像

## 素材
`assets/`：`damaged.png`、`mask.png`、`photo.png`

## 运行
```bash
# C++（参数需用 --key=value 形式）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/image_inpainting

# Python
cd Python && pip install -r requirements.txt && python image_inpainting.py
```
参数：`-i/--input` 损坏图、`-m/--mask` 掩码、`-r/--reference` 干净参考图、`-o/--output` 输出路径、`-a/--radius` 邻域半径、`-d/--display` 弹窗。
掩码占比 1.10%。全图 PSNR：TELEA 48.12 dB、NS 48.39 dB；
掩码区平均绝对误差：TELEA 0.05、NS 0.05（掩码面积小，全图 PSNR 区分度低，掩码区误差更有意义）。
