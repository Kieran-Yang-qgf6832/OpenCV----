# -*- coding: utf-8 -*-
"""文本检测：用 OpenCV 的 ERFilter（极值区域）在自然场景图里找出文字块。

思路：
    1. 转灰度并做对比度归一化，让深底白字/白底深字都能提取；
    2. cv2.text.createERFilterNM1/NM2 组成的两级滤波器逐层筛掉非文字区域；
    3. 相邻文字块按水平距离聚类成“行”，再用文字行框把零散块合并；
    4. 输出文字块（绿）与文字行（橙）两层框，并打印块数/行数。

核心函数：cv2.MSER_create / cv2.morphologyEx / cv2.boundingRect

注：OpenCV 的 ERFilter 需要 trained_classifierNM1/NM2.xml 分类器文件，
    这里没有随包提供，因此改用内置 MSER 候选 + 形态学行聚合，完全自包含。
输出：output/text_regions.png、text_mask.png
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


def morph_text_blocks(img):
    """无分类器时的替代方案：MSER 候选 + 形态学行合并（纯 OpenCV 内置算子）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, np.uint8)
    boxes = []
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        if h < 10 or h > 200 or w < 4 or w / h > 6 or h / w > 6:
            continue
        boxes.append((x, y, w, h))
        mask[y : y + h, x : x + w] = 255
    # 水平膨胀把同一行的字连起来
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    lines = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]
    return boxes, line_boxes, mask


def main():
    ap = argparse.ArgumentParser(description="场景文字检测（MSER 候选 + 行聚类）")
    ap.add_argument("--input", default=os.path.join(ASSETS, "signboard.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)

    boxes, line_boxes, mask = morph_text_blocks(img)
    canvas = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for x, y, w, h in line_boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 3)
        cv2.putText(canvas, "%dx%d" % (w, h), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    print("候选文字块 %d 个，聚合为 %d 行" % (len(boxes), len(line_boxes)))
    for i, (x, y, w, h) in enumerate(sorted(line_boxes, key=lambda b: b[1])[:8]):
        print("  第 %d 行：%d x %d px @ (%d, %d)" % (i + 1, w, h, x, y))

    cv2.imwrite(out_path("text_regions.png"), canvas)
    cv2.imwrite(out_path("text_mask.png"), mask)
    print("已保存 output/text_regions.png、text_mask.png")

    if args.display:
        cv2.imshow("text", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
