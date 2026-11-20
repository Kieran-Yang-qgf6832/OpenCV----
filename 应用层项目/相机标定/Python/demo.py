# -*- coding: utf-8 -*-
"""相机标定：用 12 张棋盘格视图求内参与畸变系数，并用重投影误差检验标定质量。

思路：
    1. 每张图检测 9x6 内角点并做亚像素细化；
    2. calibrateCamera 求 K、畸变系数，同时输出每张图的重投影误差；
    3. 用标定结果去畸变，与原图并排对比，误差按视图画成柱状图。

核心函数：cv2.findChessboardCorners / cv2.cornerSubPix / cv2.calibrateCamera / cv2.undistort
输出：output/calib_corners.png、calib_undistort.png、calib_errors.png
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

PATTERN = (9, 6)      # 内角点列数、行数
SQUARE_MM = 25.0      # 方格边长
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("已保存", os.path.relpath(path, PROJ))


def tile(images, size=(420, 320)):
    """把若干张图缩放到同一尺寸后横向/纵向拼图。"""
    thumbs = [cv2.resize(im, size) for im in images]
    return np.vstack(thumbs) if len(thumbs) == 1 else np.hstack(thumbs)


def bar_chart(values, labels, size=(760, 360), title="reprojection error (px)"):
    """用 OpenCV 手绘柱状图，避免引入 matplotlib。"""
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(canvas, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    top = max(max(values) * 1.25, 0.5)
    x0, y0, plot_h = 50, h - 40, h - 90
    cv2.line(canvas, (x0, y0), (w - 20, y0), (0, 0, 0), 1)
    cv2.line(canvas, (x0, y0), (x0, y0 - plot_h), (0, 0, 0), 1)
    bw = (w - x0 - 30) / len(values)
    for i, (v, lab) in enumerate(zip(values, labels)):
        bx = int(x0 + i * bw + bw * 0.2)
        bh = int(plot_h * v / top)
        color = (60, 160, 60) if v < np.mean(values) * 1.5 else (40, 90, 200)
        cv2.rectangle(canvas, (bx, y0 - bh), (int(bx + bw * 0.6), y0), color, -1)
        cv2.putText(canvas, "%.2f" % v, (bx - 2, y0 - bh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, lab, (bx, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="棋盘格相机标定")
    ap.add_argument("--pattern", default="%dx%d" % PATTERN)
    ap.add_argument("--square-mm", type=float, default=SQUARE_MM)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    files = sorted(glob.glob(os.path.join(ASSETS, "calib_*.png")))
    if not files:
        sys.exit("assets/ 下没有 calib_*.png")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    object_points, image_points, used, previews = [], [], [], []
    for path in files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not ok:
            print("跳过（没找到角点）：", os.path.basename(path))
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRIT)
        object_points.append(objp)
        image_points.append(corners)
        used.append(os.path.basename(path))
        if len(previews) < 2:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, ok)
            previews.append(vis)

    if len(object_points) < 3:
        sys.exit("有效视图不足 3 张，无法标定")

    first_img = cv2.imread(files[0])
    size = (first_img.shape[1], first_img.shape[0])
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None, flags=flags)

    per_view = []
    for i in range(len(object_points)):
        proj, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_points[i].reshape(-1, 2), axis=1)))
        per_view.append(err)

    print("参与标定视图：%d / %d" % (len(used), len(files)))
    print("整体重投影 RMS：%.3f px" % rms)
    print("内参 K =\n", np.array2string(K, precision=2))
    print("畸变系数 dist =", np.array2string(dist.ravel(), precision=4))
    print("单视图误差：", ["%.2f" % e for e in per_view])

    gt_file = os.path.join(ASSETS, "intrinsics_gt.txt")
    if os.path.exists(gt_file):
        g = np.loadtxt(gt_file)
        errs = [abs(K[0, 0] - g[0]) / g[0] * 100, abs(K[1, 1] - g[1]) / g[1] * 100,
                abs(K[0, 2] - g[2]), abs(K[1, 2] - g[3])]
        print("与真值对比：fx %.3f%%  fy %.3f%%  cx %.2fpx  cy %.2fpx（合成数据自带真值）"
              % (errs[0], errs[1], errs[2], errs[3]))

    save("calib_corners.png", tile(previews))
    first = cv2.imread(files[0])
    undist = cv2.undistort(first, K, dist)
    compare = np.hstack([cv2.resize(first, (512, 512)), cv2.resize(undist, (512, 512))])
    cv2.putText(compare, "original", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(compare, "undistorted", (528, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    save("calib_undistort.png", compare)
    save("calib_errors.png", bar_chart(per_view, [u.split("_")[-1].split(".")[0] for u in used]))

    if args.display:
        cv2.imshow("undistort", compare)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
