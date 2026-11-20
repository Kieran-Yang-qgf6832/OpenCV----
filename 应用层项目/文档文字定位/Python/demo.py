# -*- coding: utf-8 -*-
"""文档文字定位：在扫描文档里定位整页、文字行与单字，输出三级框选结果。

说明：
    本机没有 Tesseract 引擎，Python 版 OpenCV 也不带 OCR 识别模型，
    所以这个选题只做“把字找出来并切分”，不做字符识别——标题也据此改名。

思路：
    1. 灰度 + 自适应二值化（反向）得到墨迹；
    2. 水平投影切出文字行，垂直投影在行内切出单字；
    3. 用最小外接矩形估计整页倾斜角并校正，保证行框水平；
    4. 输出三级框选图与逐行统计（行数、字数、行高）。

核心函数：cv2.adaptiveThreshold / cv2.projections / cv2.getRotationMatrix2D
输出：output/text_layout.png、text_binary.png
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


def estimate_skew(binary):
    """用最大文字块的 minAreaRect 估计倾斜角，范围限制在 ±10°。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in contours if cv2.contourArea(c) > 50])
    if len(pts) < 10:
        return 0.0
    angle = cv2.minAreaRect(pts)[2]
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    return float(np.clip(angle, -10, 10))


def segment(binary, min_gap=6, min_height=8):
    """水平投影切行，行内垂直投影切字。"""
    rows = np.where(binary.sum(axis=1) > 0)[0]
    if len(rows) == 0:
        return []
    gaps = np.where(np.diff(rows) > min_gap)[0]
    starts = np.r_[rows[0], rows[gaps + 1]]
    ends = np.r_[rows[gaps], rows[-1]]
    lines = []
    for y0, y1 in zip(starts, ends):
        if y1 - y0 < min_height:
            continue
        band = binary[y0 : y1 + 1]
        cols = np.where(band.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        cgaps = np.where(np.diff(cols) > 3)[0]
        cs = np.r_[cols[0], cols[cgaps + 1]]
        ce = np.r_[cols[cgaps], cols[-1]]
        chars = [(int(a), int(y0), int(b - a), int(y1 - y0)) for a, b in zip(cs, ce) if b - a >= 3]
        lines.append((int(cols[0]), int(y0), int(cols[-1] - cols[0]), int(y1 - y0), chars))
    return lines


def main():
    ap = argparse.ArgumentParser(description="文档文字行/字切分（不含字符识别）")
    ap.add_argument("--input", default=os.path.join(ASSETS, "document.png"))
    ap.add_argument("--deskew", action="store_true", default=True)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    angle = estimate_skew(binary) if args.deskew else 0.0
    if abs(angle) > 0.3:
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        binary = cv2.warpAffine(binary, M, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REPLICATE)
    print("整页倾斜角估计：%.2f 度" % angle)

    lines = segment(binary)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    total_chars = 0
    for idx, (x, y, w, h, chars) in enumerate(lines):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(canvas, "L%d" % (idx + 1), (max(x - 42, 2), y + h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        for cx, cy, cw, ch in chars:
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 1)
        total_chars += len(chars)

    print("文字行 %d 行，切出单字 %d 个" % (len(lines), total_chars))
    for idx, (x, y, w, h, chars) in enumerate(lines[:6]):
        print("  第 %d 行：宽 %d px，高 %d px，含 %d 个字符块" % (idx + 1, w, h, len(chars)))

    cv2.imwrite(out_path("text_layout.png"), canvas)
    cv2.imwrite(out_path("text_binary.png"), binary)
    print("已保存 output/text_layout.png、text_binary.png")

    if args.display:
        cv2.imshow("layout", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
