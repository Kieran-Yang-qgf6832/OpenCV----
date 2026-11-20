# -*- coding: utf-8 -*-
"""图像分类：MobileNetV2(ONNX) 前向 + softmax，输出 Top-5 类别与置信度。

思路：
    1. readNetFromONNX 加载 MobileNetV2，输入 224x224，按训练配置做归一化（mean/std）；
    2. forward 得到 1000 维 logits，手动 softmax 转成概率；
    3. 取 Top-5 与 ImageNet 标签对齐，图右侧画横向概率条；
    4. 同时做一次水平翻转推理，看结论是否稳定（简单的鲁棒性检查）。

核心函数：cv2.dnn.readNetFromONNX / blobFromImage / net.forward / softmax
输出：output/classification.png
"""

import argparse
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖：pip install opencv-contrib-python numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ASSETS = os.path.join(PROJ, "assets")
OUT = os.path.join(PROJ, "output")
MODELS = os.path.join(os.path.dirname(PROJ), "_models")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_model():
    model = os.path.join(MODELS, "mobilenetv2-7.onnx")
    labels = os.path.join(MODELS, "imagenet_classes.txt")
    if not (os.path.exists(model) and os.path.exists(labels)):
        sys.exit("缺少模型/标签：_models/mobilenetv2-7.onnx、imagenet_classes.txt（见 README）")
    net = cv2.dnn.readNetFromONNX(model)
    names = [line.strip() for line in open(labels, encoding="utf-8") if line.strip()]
    return net, names


def predict(net, names, img, topk=5):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (SIZE, SIZE)), 1.0 / 255.0, (SIZE, SIZE), MEAN, swapRB=True)
    # blobFromImage 只支持整体均值/缩放，这里再按通道除以标准差
    blob = blob / np.array(STD, np.float32).reshape(1, 3, 1, 1)
    net.setInput(blob)
    logits = net.forward().ravel()
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    idx = np.argsort(probs)[::-1][:topk]
    return [(names[i] if i < len(names) else str(i), float(probs[i])) for i in idx]


def render(img, top, extra_lines=()):
    """左侧原图 + 右侧 Top-5 概率条。"""
    h, w = img.shape[:2]
    panel_w = max(int(w * 0.8), 520)
    canvas = np.full((h, w + panel_w + 20, 3), 22, np.uint8)
    canvas[:, :w] = cv2.resize(img, (w, h))
    x0 = w + 20
    cv2.putText(canvas, "MobileNetV2  Top-5 (softmax)", (x0, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    step = max(int(h * 0.13), 90)
    for i, (name, prob) in enumerate(top):
        y = 110 + i * step
        bar_w = int(prob * (panel_w - 80))
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + 34), (60, 170, 240), -1)
        cv2.putText(canvas, "%.3f" % prob, (x0 + bar_w + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, name[:30], (x0, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    for i, line in enumerate(extra_lines):
        cv2.putText(canvas, line, (x0, canvas.shape[0] - 30 - 26 * (len(extra_lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="MobileNetV2 图像分类")
    ap.add_argument("--input", default=os.path.join(ASSETS, "image.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    net, names = load_model()
    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)

    top = predict(net, names, img)
    flipped = predict(net, names, cv2.flip(img, 1))
    print("Top-5：")
    for name, prob in top:
        print("   %-38s %.4f" % (name, prob))
    print("水平翻转后 Top-1：%s（%.3f）%s" % (flipped[0][0], flipped[0][1],
                                         "与正立一致" if flipped[0][0] == top[0][0] else "发生变化"))

    canvas = render(img, top, ["flip test: %s %.3f" % (flipped[0][0][:26], flipped[0][1])])
    cv2.imwrite(out_path("classification.png"), canvas)
    print("已保存 output/classification.png")

    if args.display:
        cv2.imshow("classify", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
