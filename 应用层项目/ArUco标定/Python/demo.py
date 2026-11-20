# -*- coding: utf-8 -*-
"""ArUco 标定：用多视角的 ArUco 标定板求相机内参与畸变。

思路：
    1. 标定板是 3x3 个 DICT_6X6_250 标记（id 0~8），标记边长与间隔已知；
    2. 按 id 把每个标记的四角映射到板坐标系，拼成一张大点集；
    3. 12 个视角联合 calibrateCamera，得到 K、畸变系数与重投影误差；
    4. 用标定结果去畸变，并拼一张视图概览。

核心函数：cv2.aruco.detectMarkers / cv2.calibrateCamera / cv2.undistort
输出：output/calib_undistorted.png、calib_montage.png
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

MARKER_MM = 40.0   # 单个标记边长
GAP_MM = 12.0      # 标记之间的间隔
GRID = 3           # 3x3 个标记


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print("已保存", os.path.relpath(path, PROJ))


def board_object_points(marker_mm=MARKER_MM, gap_mm=GAP_MM):
    """按 id 生成整块标定板的角点：id 从左上开始按行编号。"""
    step = marker_mm + gap_mm
    half = marker_mm / 2.0
    per_marker = {}
    for mid in range(GRID * GRID):
        row, col = divmod(mid, GRID)
        cx, cy = col * step, -row * step
        per_marker[mid] = np.array(
            [[cx - half, cy + half, 0], [cx + half, cy + half, 0],
             [cx + half, cy - half, 0], [cx - half, cy - half, 0]], np.float32)
    return per_marker


def main():
    ap = argparse.ArgumentParser(description="ArUco 标定板多视图标定")
    ap.add_argument("--marker-mm", type=float, default=MARKER_MM)
    ap.add_argument("--gap-mm", type=float, default=GAP_MM)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    obj_map = board_object_points(args.marker_mm, args.gap_mm)

    files = sorted(glob.glob(os.path.join(ASSETS, "board_view_*.png")))
    if not files:
        sys.exit("assets/ 下没有 board_view_*.png")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    object_points, image_points, previews = [], [], []
    for path in files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) < 4:
            print("跳过（标记不足 4 个）：", os.path.basename(path))
            continue
        obj_pts, img_pts = [], []
        for corner, mid in zip(corners, ids.ravel()):
            if int(mid) not in obj_map:
                continue
            obj_pts.append(obj_map[int(mid)])
            img_pts.append(corner.reshape(4, 2).astype(np.float32))
        object_points.append(np.vstack(obj_pts))
        image_points.append(np.vstack(img_pts))
        if len(previews) < 4:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            previews.append(vis)

    if len(object_points) < 3:
        sys.exit("有效视图不足 3 张")

    h, w = cv2.imread(files[0]).shape[:2]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (w, h), None, None, flags=flags)
    print("参与标定视图：%d / %d，每视图角点 %d" % (len(object_points), len(files), object_points[0].shape[0]))
    print("重投影 RMS：%.3f px" % rms)
    print("K =\n", np.array2string(K, precision=2))
    print("dist =", np.array2string(dist.ravel(), precision=4))

    gt_file = os.path.join(ASSETS, "intrinsics_gt.txt")
    if os.path.exists(gt_file):
        g = np.loadtxt(gt_file)
        errs = [abs(K[0, 0] - g[0]) / g[0] * 100, abs(K[1, 1] - g[1]) / g[1] * 100,
                abs(K[0, 2] - g[2]), abs(K[1, 2] - g[3])]
        print("与真值对比：fx %.3f%%  fy %.3f%%  cx %.2fpx  cy %.2fpx（合成数据自带真值）"
              % (errs[0], errs[1], errs[2], errs[3]))

    first = cv2.imread(files[0])
    undist = cv2.undistort(first, K, dist)
    pair = np.hstack([cv2.resize(first, (512, 512)), cv2.resize(undist, (512, 512))])
    cv2.putText(pair, "original", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(pair, "undistorted", (528, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    save("calib_undistorted.png", pair)

    thumbs = [cv2.resize(v, (420, 420)) for v in previews]
    while len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
    save("calib_montage.png", np.vstack(rows))

    if args.display:
        cv2.imshow("undistort", pair)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
