# -*- coding: utf-8 -*-
"""图像修复应用：按掩膜把划痕/破损区域补上，对比 Telea 与 Navier-Stokes 两种算法。

思路：
    1. 读入受损老照片与人工掩膜（白色为待修复区域）；
    2. cv2.inpaint 分别用 INPAINT_TELEA 与 INPAINT_NS 修复，半径可调；
    3. 把掩膜膨胀一点做“边缘过渡”，避免修补区出现硬接缝；
    4. 输出三宫格对比图与局部放大图，并打印掩膜占比与耗时。

核心函数：cv2.inpaint / cv2.dilate
输出：output/inpaint.png、inpaint_crop.png、inpaint_mask.png
"""

import argparse
import os
import sys
import time

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


def label(img, text, size=(400, 400)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="掩膜驱动的小区域图像修复")
    ap.add_argument("--input", default=os.path.join(ASSETS, "damaged.png"))
    ap.add_argument("--mask", default=os.path.join(ASSETS, "mask.png"))
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        sys.exit("读不到 damaged.png / mask.png")
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    ratio = float((mask > 0).mean() * 100)
    print("待修复像素占比 %.2f%%，修复半径 %d" % (ratio, args.radius))

    outs = {}
    for name, flag in (("Telea", cv2.INPAINT_TELEA), ("Navier-Stokes", cv2.INPAINT_NS)):
        t0 = time.time()
        outs[name] = cv2.inpaint(img, mask, args.radius, flag)
        print("  %-14s 耗时 %.3f s" % (name, time.time() - t0))

    masked = img.copy()
    masked[mask > 0] = (0, 0, 255)
    grid = np.hstack([label(img, "damaged"),
                      label(masked, "mask %.2f%%" % ratio),
                      label(outs["Telea"], "inpaint Telea"),
                      label(outs["Navier-Stokes"], "inpaint NS")])
    cv2.imwrite(out_path("inpaint.png"), grid)

    # 用积分图找破损最密集的窗口，放大图才看得出修复效果
    integral = cv2.integral((mask > 0).astype(np.float32))
    cs = 320
    best, cy, cx = -1, mask.shape[0] // 2, mask.shape[1] // 2
    for y in range(0, mask.shape[0] - cs, 40):
        for x in range(0, mask.shape[1] - cs, 40):
            total = integral[y + cs, x + cs] - integral[y, x + cs] - integral[y + cs, x] + integral[y, x]
            if total > best:
                best, cy, cx = total, y + cs // 2, x + cs // 2
    cs = 260
    x0, y0 = max(0, cx - cs // 2), max(0, cy - cs // 2)
    x1, y1 = min(img.shape[1], x0 + cs), min(img.shape[0], y0 + cs)
    zoom = np.hstack([label(img[y0:y1, x0:x1], "damaged", (300, 300)),
                      label(outs["Telea"][y0:y1, x0:x1], "Telea", (300, 300)),
                      label(outs["Navier-Stokes"][y0:y1, x0:x1], "NS", (300, 300))])
    cv2.imwrite(out_path("inpaint_crop.png"), zoom)
    cv2.imwrite(out_path("inpaint_mask.png"), mask)
    print("已保存 output/inpaint.png、inpaint_crop.png、inpaint_mask.png")

    if args.display:
        cv2.imshow("inpaint", zoom)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
