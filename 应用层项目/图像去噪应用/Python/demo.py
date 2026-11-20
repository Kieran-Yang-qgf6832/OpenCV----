# -*- coding: utf-8 -*-
"""图像去噪应用：非局部均值 / 双边 / 中值三种方案对比，用 PSNR·SSIM 与局部放大图评判。

思路：
    1. 干净图与加噪图成对给出，所以可以量化评价，而不只是“看着舒服”；
    2. 分别跑 fastNlMeansDenoisingColored、bilateralFilter、medianBlur；
    3. 逐张算 PSNR/SSIM 与残留噪声标准差，并截局部纹理区域做放大对照；
    4. 输出一张“原图/加噪/三种结果”的对比图与数值表。

核心函数：cv2.fastNlMeansDenoisingColored / cv2.bilateralFilter / cv2.medianBlur
输出：output/denoise.png、denoise_crop.png
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


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def ssim(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu1, mu2 = cv2.GaussianBlur(ga, k, 1.5), cv2.GaussianBlur(gb, k, 1.5)
    s1 = cv2.GaussianBlur(ga * ga, k, 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(gb * gb, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(ga * gb, k, 1.5) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return float(m.mean())


def label(img, text, size=(400, 300)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="三种经典去噪算法对比")
    ap.add_argument("--clean", default=os.path.join(ASSETS, "clean.png"))
    ap.add_argument("--noisy", default=os.path.join(ASSETS, "noisy.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    clean = cv2.imread(args.clean)
    noisy = cv2.imread(args.noisy)
    if clean is None or noisy is None:
        sys.exit("读不到 clean.png / noisy.png")
    if clean.shape[:2] != noisy.shape[:2]:
        noisy = cv2.resize(noisy, (clean.shape[1], clean.shape[0]))

    results = {
        "noisy": noisy,
        "NLM": cv2.fastNlMeansDenoisingColored(noisy, None, 7, 7, 7, 21),
        "bilateral": cv2.bilateralFilter(noisy, 9, 60, 60),
        "median": cv2.medianBlur(noisy, 5),
    }

    print("%-10s %8s %8s %8s" % ("method", "PSNR", "SSIM", "noise_std"))
    for name, img in results.items():
        noise_std = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                                 - cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)))
        print("%-10s %8.2f %8.4f %8.2f" % (name, cv2.PSNR(clean, img), ssim(clean, img), noise_std))

    h, w = clean.shape[:2]
    cx, cy, cs = int(w * 0.30), int(h * 0.45), int(min(h, w) * 0.22)

    def crop(img):
        return img[cy : cy + cs, cx : cx + cs]

    row1 = np.hstack([label(clean, "clean"), label(noisy, "noisy %.2f dB" % cv2.PSNR(clean, noisy))])
    row2 = np.hstack([label(results["NLM"], "NLM %.2f dB" % cv2.PSNR(clean, results["NLM"])),
                      label(results["bilateral"], "bilateral %.2f dB" % cv2.PSNR(clean, results["bilateral"])),
                      label(results["median"], "median %.2f dB" % cv2.PSNR(clean, results["median"]))])
    row1 = cv2.resize(row1, (row2.shape[1], int(row1.shape[0] * row2.shape[1] / row1.shape[1])))
    cv2.imwrite(out_path("denoise.png"), np.vstack([row1, row2]))

    zoom = np.hstack([label(crop(noisy), "noisy", (300, 300)),
                      label(crop(results["NLM"]), "NLM", (300, 300)),
                      label(crop(clean), "clean", (300, 300))])
    cv2.imwrite(out_path("denoise_crop.png"), zoom)
    print("已保存 output/denoise.png、denoise_crop.png")

    if args.display:
        cv2.imshow("denoise", zoom)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
