# -*- coding: utf-8 -*-
"""尺寸测量：用画面里的 ArUco 标记做比例尺，测出目标的真实尺寸（毫米）。

思路：
    1. ArUco 标记的物理边长已知（本例按 50 mm），由它在像素上的边长得到 px/mm；
    2. 目标按颜色分割（蓝色方块），用最小外接矩形求出长边/短边的像素长度；
    3. 除以比例尺即得毫米尺寸，并把量得的尺寸直接标在图上。

核心函数：cv2.aruco.detectMarkers / cv2.minAreaRect / cv2.boxPoints
输出：output/measure_result.png
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

MARKER_MM = 50.0          # ArUco 标记的物理边长（mm）
# 目标是一块深蓝色卡片，背景是同色系的浅蓝渐变，因此除了色相还要卡住明度上限
HUE_RANGE = (95, 130)
SAT_MIN = 80
VAL_MAX = 95


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("已保存", os.path.relpath(path, PROJ))
    return path


def find_marker(img):
    """返回 (marker 四角像素坐标, 标记 id)；找不到时返回 (None, None)。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
    if ids is None or len(corners) == 0:
        return None, None
    return corners[0].reshape(4, 2), int(ids.ravel()[0])


def px_per_mm(corner):
    """标记边长取四边平均，换算出每毫米多少像素。"""
    sides = [np.linalg.norm(corner[i] - corner[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides)) / MARKER_MM


def find_target(img):
    """按颜色找出蓝色目标，返回最小外接矩形的 (中心, (宽, 高), 角度)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((hue > HUE_RANGE[0]) & (hue < HUE_RANGE[1]) & (sat > SAT_MIN) & (val < VAL_MAX)) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not contours:
        return None, mask
    contour = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(contour), mask


def draw_size(img, rect, scale):
    """画出外接矩形、尺寸箭头与毫米读数。"""
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
    (cx, cy), (w, h), angle = rect
    long_side, short_side = max(w, h), min(w, h)
    long_mm, short_mm = long_side / scale, short_side / scale

    # 沿长边画一根带端帽的尺寸线，读数写在线的中点
    pts = box.astype(np.float32)
    edge_len = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    i = int(np.argmax(edge_len))
    p1, p2 = pts[i], pts[(i + 1) % 4]
    cv2.arrowedLine(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), 2, tipLength=0.04)
    cv2.arrowedLine(img, tuple(p2.astype(int)), tuple(p1.astype(int)), (0, 255, 255), 2, tipLength=0.04)
    mid = ((p1 + p2) / 2).astype(int)
    label = f"%.1f x %.1f mm" % (long_mm, short_mm)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (mid[0] - tw // 2 - 8, mid[1] - th - 12), (mid[0] + tw // 2 + 8, mid[1] + 6), (0, 0, 0), -1)
    cv2.putText(img, label, (mid[0] - tw // 2, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return long_mm, short_mm


def main():
    ap = argparse.ArgumentParser(description="基于 ArUco 参照物的尺寸测量")
    ap.add_argument("--input", default=os.path.join(ASSETS, "measure_scene.png"))
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM, help="参照标记的实际边长（毫米）")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)

    corner, marker_id = find_marker(img)
    if corner is None:
        sys.exit("画面里没有检测到 ArUco 标记，无法建立比例尺")
    scale = px_per_mm(corner)
    print("检测到标记 id=%d，按边长 %.1f mm 换算：%.3f px/mm" % (marker_id, args.marker_mm, scale))

    rect, mask = find_target(img)
    if rect is None:
        sys.exit("没有分割出目标色块")

    canvas = img.copy()
    cv2.aruco.drawDetectedMarkers(canvas, [corner.reshape(1, 4, 2).astype(np.float32)], np.array([[marker_id]]))
    long_mm, short_mm = draw_size(canvas, rect, scale)

    info = "scale %.3f px/mm  marker id %d" % (scale, marker_id)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(canvas, info, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("目标尺寸：%.1f mm x %.1f mm（像素外接矩形 %.1f x %.1f）" % (long_mm, short_mm, max(rect[1]), min(rect[1])))
    save("measure_result.png", canvas)
    save("measure_mask.png", mask)

    if args.display:
        cv2.imshow("measure", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
