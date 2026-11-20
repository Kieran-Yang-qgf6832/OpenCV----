# -*- coding: utf-8 -*-
"""PnP 位姿估计：由棋盘格角点的 2D-3D 对应求相机相对标定板的 6DoF 位姿。

思路：
    1. findChessboardCorners + cornerSubPix 拿到 9x6 个角点的亚像素坐标；
    2. 标定板方格边长已知，角点在世界系下是规则的 3D 点阵；
    3. solvePnP 解 rvec/tvec，再用 LM 精修，最后把坐标轴和一个立方体投回图像验证；
    4. 用重投影误差量化位姿精度。

核心函数：cv2.solvePnP / cv2.solvePnPRefineLM / cv2.projectPoints
输出：output/pnp_axes.png
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

PATTERN = (9, 6)
SQUARE_MM = 25.0


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("已保存", os.path.relpath(path, PROJ))


def assumed_camera(img):
    h, w = img.shape[:2]
    f = 1.2 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def cube_points(square_mm, cols, rows, height_mm):
    """立方体 8 个顶点：贴着标定板的一个“盒子”，用来直观检查位姿。"""
    w = (cols - 1) * square_mm
    h = (rows - 1) * square_mm
    x0, y0 = 0.0, -h
    return np.float32([
        [x0, y0, 0], [x0 + w, y0, 0], [x0 + w, y0 + h, 0], [x0, y0 + h, 0],
        [x0, y0, -height_mm], [x0 + w, y0, -height_mm], [x0 + w, y0 + h, -height_mm], [x0, y0 + h, -height_mm],
    ])


def draw_cube(img, K, rvec, tvec, pts):
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    p = proj.reshape(-1, 2).astype(int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 200, 255), 2)
    for i, (x, y) in enumerate(p):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser(description="棋盘格 PnP 位姿估计")
    ap.add_argument("--input", default=os.path.join(ASSETS, "board.png"))
    ap.add_argument("--square-mm", type=float, default=SQUARE_MM)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit("读不到输入图：%s" % args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if not ok:
        sys.exit("没找到 %dx%d 棋盘格" % PATTERN)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))

    cols, rows = PATTERN
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    K = assumed_camera(img)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        sys.exit("solvePnP 失败")
    rvec, tvec = cv2.solvePnPRefineLM(objp, corners, K, None, rvec, tvec)

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))

    canvas = img.copy()
    cv2.drawChessboardCorners(canvas, PATTERN, corners, True)
    try:
        cv2.drawFrameAxes(canvas, K, None, rvec, tvec, args.square_mm * 2, 3)
    except cv2.error:
        pass
    draw_cube(canvas, K, rvec, tvec, cube_points(args.square_mm, cols, rows, args.square_mm * 2))

    rmat, _ = cv2.Rodrigues(rvec)
    euler = cv2.RQDecomp3x3(rmat)[0]
    text = "t=(%.1f, %.1f, %.1f)mm  ang=(%.1f, %.1f, %.1f)deg  reproj=%.2fpx" % (
        tvec[0, 0], tvec[1, 0], tvec[2, 0], euler[0], euler[1], euler[2], err)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

    print("旋转向量 rvec =", np.array2string(rvec.ravel(), precision=3))
    print("平移向量 tvec =", np.array2string(tvec.ravel(), precision=2), "(单位与方格边长一致)")
    print("欧拉角 =", np.array2string(np.array(euler), precision=2), "度")
    print("重投影误差 =%.3f px" % err)
    save("pnp_axes.png", canvas)

    if args.display:
        cv2.imshow("pnp", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
