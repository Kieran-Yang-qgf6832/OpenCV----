# 图像分类

## 需求
用预训练 CNN 对单张图片做 ImageNet 1000 类分类，输出 Top-5 概率。

## 作用

- **来料/分拣分类**：粗粒度品类判断。
- **相册整理**：按内容打标签。
- **模型落地验证**：检验 `cv2.dnn` 能否跑通某个 ONNX 模型。

适用边界：ImageNet 只有 1000 类，业务类别必须落在这 1000 类里才有意义，否则要自己微调。

## 核心函数
`cv2.dnn.readNetFromONNX()`、`cv2.dnn.blobFromImage()`、`net.forward()`、softmax

## 关键步骤
1. 加载 MobileNetV2 ONNX 与 ImageNet 标签
2. 预处理：缩放到 224、按 ImageNet 均值/标准差归一化、RGB 通道交换
3. 前向得到 1000 维 logits，手写 softmax 转概率
4. 取 Top-5 画概率条；再做一次水平翻转推理检查结论稳定性

## 输入
`image.png`（1024×1024 的猫照片）

## 输出
左侧原图 + 右侧 Top-5 概率条的信息图

## 素材
`assets/image.png`；模型 `_models/mobilenetv2-7.onnx`（14 MB）与 `imagenet_classes.txt`。

## 运行
```bash
cd Python && python3 demo.py --display
```
依赖 OpenCV + NumPy（ONNX 由 `cv2.dnn` 直接加载，不需要 onnxruntime）。

## 演示结果
`output/classification.png`：Top-5 依次为
**Egyptian cat 0.395 / tiger cat 0.369 / tabby 0.223** / lynx 0.005 / Persian cat 0.004。

## 实现要点
- **必须做 softmax**：这个 ONNX 导出的输出是 logits（原始峰值 14.2），直接当概率看会误判置信度。
- **归一化的坑**：`blobFromImage` 只支持整体缩放与单一均值，per-channel 的 std 要自己再除一遍。
- **翻转鲁棒性**：水平翻转后 Top-1 从 Egyptian cat 变成 tiger cat，
  说明猫的这几个近邻类别本来就难分，涉及细分类时应看 Top-5 而不是只看 Top-1。
