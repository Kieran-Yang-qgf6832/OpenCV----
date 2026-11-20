# -*- coding: utf-8 -*-
"""视频质量评估：逐帧算 PSNR / SSIM / 清晰度，并定位质量最差的时间点。

思路：
    1. 参考视频与失真视频逐帧对齐；
    2. 每帧算 PSNR、SSIM（高斯窗），以及拉普拉斯方差（清晰度）与高频能量；
    3. 画四条曲线并把最差帧、最好帧摆出来做对照；
    4. 打印统计摘要，便于写进报告。

核心函数：cv2.PSNR / 高斯滤波手写 SSIM / cv2.Laplacian
输出：output/quality_curves.png、quality_compare.png
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


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def ssim(a, b):
    """11x11 高斯窗 SSIM（灰度），返回整幅均值。"""
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


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def draw_curves(series, size=(1100, 520)):
    """两栏四曲线：上排 PSNR/SSIM，下排清晰度。"""
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    panels = [("PSNR (dB)", series["psnr"], (200, 90, 30)),
              ("SSIM", series["ssim"], (30, 150, 60)),
              ("sharpness (Laplacian var)", series["sharp"], (180, 80, 160))]
    for pi, (title, values, color) in enumerate(panels):
        y_off = 30 + pi * (h // 3)
        ph = h // 3 - 50
        cv2.putText(canvas, "%s  min %.2f / mean %.2f" % (title, min(values), float(np.mean(values))),
                    (16, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-6)
        pts = [(int(40 + i * (w - 80) / max(len(values) - 1, 1)),
                int(y_off + 20 + ph - (values[i] - lo) * ph / span)) for i in range(len(values))]
        cv2.line(canvas, (40, y_off + 20 + ph), (w - 40, y_off + 20 + ph), (180, 180, 180), 1)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="视频质量客观指标评估")
    ap.add_argument("--reference", default=os.path.join(ASSETS, "reference.mp4"))
    ap.add_argument("--distorted", default=os.path.join(ASSETS, "distorted.mp4"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    ref = read_video(args.reference)
    dis = read_video(args.distorted)
    n = min(len(ref), len(dis))
    if n == 0:
        sys.exit("视频读不到帧")
    print("参考 %d 帧，失真 %d 帧，取前 %d 帧对比" % (len(ref), len(dis), n))

    psnr, ssims, sharp_ref, sharp_dis = [], [], [], []
    for i in range(n):
        a, b = ref[i], dis[i]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        psnr.append(float(cv2.PSNR(a, b)))
        ssims.append(ssim(a, b))
        sharp_ref.append(sharpness(a))
        sharp_dis.append(sharpness(b))

    worst = int(np.argmin(psnr))
    best = int(np.argmax(psnr))
    print("PSNR : mean %.2f dB, min %.2f, max %.2f" % (np.mean(psnr), min(psnr), max(psnr)))
    print("SSIM : mean %.4f, min %.4f" % (np.mean(ssims), min(ssims)))
    print("清晰度（拉普拉斯方差）：参考 mean %.1f / 失真 mean %.1f（下降 %.0f%%）"
          % (np.mean(sharp_ref), np.mean(sharp_dis),
             100.0 * (1 - np.mean(sharp_dis) / max(np.mean(sharp_ref), 1e-6))))
    print("最差帧 #%d（PSNR %.2f dB），最好帧 #%d（%.2f dB）" % (worst, psnr[worst], best, psnr[best]))

    canvas = draw_curves({"psnr": psnr, "ssim": ssims, "sharp": sharp_dis})
    cv2.imwrite(out_path("quality_curves.png"), canvas)

    def column(idx, tag):
        a, b = cv2.resize(ref[idx], (560, 315)), cv2.resize(dis[idx], (560, 315))
        pair = np.vstack([a, b])
        cv2.putText(pair, "%s  ref #%d" % (tag, idx), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(pair, "PSNR %.1f dB  SSIM %.3f" % (psnr[idx], ssims[idx]), (12, 315 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return pair

    compare = np.hstack([column(best, "best"), column(worst, "worst")])
    cv2.imwrite(out_path("quality_compare.png"), compare)
    print("已保存 output/quality_curves.png、quality_compare.png")

    if args.display:
        cv2.imshow("quality", compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
