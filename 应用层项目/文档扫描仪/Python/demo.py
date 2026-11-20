# -*- coding: utf-8 -*-
"""文档扫描仪：自动找纸张四角、透视校正、光照均衡，把翻拍照片变成"扫描件"。

思路：
    1. 素材是带 alpha 的纸张照片，用 alpha 与灰度联合确定纸面区域；
    2. 最大轮廓做多边形逼近得到四个角点（不足则用最小外接矩形兜底）；
    3. 按目标长宽比做四点透视变换，把纸面拉正；
    4. 自适应阈值 + 去阴影（同态式的除法归一化）得到白底黑字的扫描件；
    5. 三张素材各生成一组“原图 → 拉正 → 扫描件”的对照拼图。

核心函数：cv2.findContours / cv2.approxPolyDP / cv2.getPerspectiveTransform / cv2.adaptiveThreshold
输出：output/scan_result.png、scan_corners.png
"""

import argparse
import glob
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

TARGET_W = 700   # 拉正后的目标宽度


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_page(path):
    """读入素材：alpha 通道用作纸面掩膜，返回 (bgr, 纸面掩膜)。"""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    bgr = raw[:, :, :3].copy() if raw.ndim == 3 and raw.shape[2] == 4 else raw.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 素材的 alpha 是全不透明（纸张靠亮度区别于背景），所以用亮度阈值找纸面
    mask = (gray > 140).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return bgr, mask


def find_quad(bgr, mask):
    """在掩膜上找最大四边形，返回 4 个角点（顺时针）。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


def order_points(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def warp(bgr, quad, width=TARGET_W):
    tl, tr, br, bl = order_points(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    height = int(width * h / max(w, 1.0))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(bgr, M, (width, height))


def to_scan(page):
    """光照均衡 + 二值化：先除以大尺度模糊去掉阴影，再自适应阈值。"""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 35)
    norm = cv2.divide(gray, blur, scale=255)
    bin_ = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)
    bin_ = cv2.medianBlur(bin_, 3)
    return cv2.cvtColor(bin_, cv2.COLOR_GRAY2BGR)


def labeled(img, text, size=(360, 270)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="翻拍文档自动校正为扫描件")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ASSETS, "doc-0*.png")))
    if not files:
        sys.exit("assets/ 下没有 doc-0*.png")

    rows, corners_preview = [], []
    for path in files:
        bgr, mask = load_page(path)
        if bgr is None:
            continue
        quad = find_quad(bgr, mask)
        if quad is None:
            print("跳过（找不到纸张轮廓）：", os.path.basename(path))
            continue
        height_mm_ratio = None
        flat = warp(bgr, quad)
        scan = to_scan(flat)
        area_ratio = abs(cv2.contourArea(quad)) / float(bgr.shape[0] * bgr.shape[1])
        print("%-34s 纸张占画面 %.0f%%，拉正后 %dx%d" %
              (os.path.basename(path), 100 * area_ratio, flat.shape[1], flat.shape[0]))

        marked = bgr.copy()
        cv2.polylines(marked, [quad.astype(np.int32)], True, (0, 255, 0), 6)
        corners_preview.append(marked)
        rows.append(np.hstack([labeled(bgr, "photo"), labeled(marked, "detected corners"),
                               labeled(flat, "rectified"), labeled(scan, "binarized")]))

    if not rows:
        sys.exit("没有成功处理的素材")
    cv2.imwrite(out_path("scan_result.png"), np.vstack(rows))
    cv2.imwrite(out_path("scan_corners.png"), np.vstack([labeled(c, "corners", (600, 450)) for c in corners_preview]))
    print("已保存 output/scan_result.png、scan_corners.png")

    if args.display:
        cv2.imshow("scan", cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
