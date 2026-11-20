# -*- coding: utf-8 -*-
"""ArUco 检测：检测标记、解出 ID，并估计每个标记在相机坐标系下的位姿。

思路：
    1. detectMarkers 拿到四角像素坐标与 ID；
    2. 用“标记边长已知 + 假定的相机内参”调 solvePnP，得到 rvec/tvec；
    3. drawFrameAxes 把位姿画成坐标轴，把 ID 与距离标在标记中心。

核心函数：cv2.aruco.detectMarkers / cv2.solvePnP / cv2.drawFrameAxes
输出：output/aruco_detected.png
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

MARKER_MM = 40.0  # 假定标记边长（mm）


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("已保存", os.path.relpath(path, PROJ))


def assumed_camera(img):
    """没有标定文件时，用图像宽度估一个合理内参（f≈1.2W，主点在中心）。"""
    h, w = img.shape[:2]
    f = 1.2 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def marker_object_points(size_mm):
    half = size_mm / 2.0
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)


def main():
    ap = argparse.ArgumentParser(description="ArUco 标记检测与位姿估计")
    ap.add_argument("--input", default=os.path.join(ASSETS, "aruco_scene.png"))
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
    if ids is None or len(ids) == 0:
        sys.exit("未检测到 ArUco 标记")

    canvas = img.copy()
    cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
    K = assumed_camera(img)
    objp = marker_object_points(args.marker_mm)
    axis_len = args.marker_mm / 2.0

    detections = []
    for corner, mid in zip(corners, ids.ravel()):
        pts = corner.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(objp, pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        distance = float(np.linalg.norm(tvec))
        detections.append((int(mid), distance))
        try:
            cv2.drawFrameAxes(canvas, K, None, rvec, tvec, axis_len, 2)
        except cv2.error:
            proj, _ = cv2.projectPoints(np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0]]), rvec, tvec, K, None)
            proj = proj.reshape(-1, 2).astype(int)
            cv2.line(canvas, tuple(proj[0]), tuple(proj[1]), (0, 0, 255), 2)
            cv2.line(canvas, tuple(proj[0]), tuple(proj[2]), (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, "#%d %.0fmm" % (mid, distance), (center[0] - 40, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, "#%d %.0fmm" % (mid, distance), (center[0] - 40, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    detections.sort()
    print("共检测到 %d 个标记" % len(detections))
    for mid, dist in detections:
        print("  id=%-2d 距离 %.1f mm" % (mid, dist))
    print("ID 列表：", [d[0] for d in detections])

    save("aruco_detected.png", canvas)
    if args.display:
        cv2.imshow("aruco", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
