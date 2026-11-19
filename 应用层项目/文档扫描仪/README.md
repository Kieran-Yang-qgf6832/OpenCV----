# 文档扫描仪

## 需求
将手机文档照片自动矫正为正面扫描件

## 核心函数
cv2.Canny(), cv2.findContours(), cv2.warpPerspective()

## 关键步骤
1. 边缘检测
2. 查找文档轮廓
3. 透视变换矫正
4. 自适应二值化

## 输入
文档照片

示例素材（`assets/`）：
- `assets/doc-01-flat-white-on-dark.png`：常规微倾斜拍摄，适合基础对比
- `assets/doc-02-tilted-on-charcoal.png`：斜俯拍、梯形透视明显，适合演示四点检测
- `assets/doc-03-receipt-on-charcoal.png`：带折痕、边缘微卷的小票，适合测鲁棒性

## 输出
矫正扫描件

演示结果（`assets/demo/`）：
- `*-contour.png`：四点轮廓检测结果
- `*-warped.png`：透视矫正结果
- `*-binary.png`：自适应二值化结果
