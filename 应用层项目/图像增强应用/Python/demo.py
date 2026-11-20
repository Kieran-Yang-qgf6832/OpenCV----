# -*- coding: utf-8 -*-
"""图像增强应用：暗光图的三种增强方案对比，用亮度分布与信息熵量化提升。

思路：
    1. LAB 空间只对亮度通道做 CLAHE，避免色偏；
    2. Gamma 校正（γ<1）抬升暗部，并做一次亮度直方图均衡作对照；
    3. 用平均亮度、亮度标准差、信息熵、暗部像素占比量化增强前后；
    4. 输出四宫格对比图与直方图对照，说明各自适用场景。

核心函数：cv2.createCLAHE / cv2.LUT（gamma）/ cv2.calcHist
输出：output/enhance.png、enhance_hist.png
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

GAMMA = 0.45


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stats(img, tag):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = float((gray < 60).mean() * 100)
    print("%-22s 平均亮度 %6.1f  对比度(std) %5.1f  熵 %4.2f  暗部占比 %4.1f%%"
          % (tag, gray.mean(), gray.std(), entropy(img), dark))
    return gray.mean(), gray.std(), entropy(img), dark


def gamma_correct(img, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def clahe_lab(img, clip=3.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def blend_gamma(img, alpha=0.55):
    """gamma 提亮 + 原图按比例混合，暗部提上来又不至于把亮部冲爆。"""
    return cv2.addWeighted(gamma_correct(img, GAMMA), alpha, img, 1 - alpha, 0)


def label(img, text, size=(400, 300)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def hist_image(img, size=(500, 260)):
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = np.log1p(hist) / max(np.log1p(hist).max(), 1e-6)
    pts = [(int(20 + i * (w - 40) / 255), int(h - 20 - hist[i] * (h - 50))) for i in range(256)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (60, 90, 200), 1)
    cv2.putText(canvas, "brightness histogram (log)", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="暗光图像增强对比")
    ap.add_argument("--input", default=os.path.join(ASSETS, "lowlight.png"))
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)

    results = {
        "original": img,
        "CLAHE(LAB)": clahe_lab(img),
        "gamma %.2f" % args.gamma: gamma_correct(img, args.gamma),
        "gamma+blend": blend_gamma(img),
    }
    for tag, out in results.items():
        stats(out, tag)

    keys = list(results)
    grid = np.vstack([
        np.hstack([label(results[keys[0]], keys[0]), label(results[keys[1]], keys[1])]),
        np.hstack([label(results[keys[2]], keys[2]), label(results[keys[3]], keys[3])]),
    ])
    cv2.imwrite(out_path("enhance.png"), grid)

    hist = np.hstack([hist_image(results["original"])])
    hist = np.vstack([hist, hist_image(results["CLAHE(LAB)"]), hist_image(results["gamma+blend"])])
    cv2.imwrite(out_path("enhance_hist.png"), hist)
    print("已保存 output/enhance.png、enhance_hist.png")

    if args.display:
        cv2.imshow("enhance", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
