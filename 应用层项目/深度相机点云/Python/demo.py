# -*- coding: utf-8 -*-
"""深度相机点云：用内参把 RGB-D 反投影成三维点云，并渲染两个视角观察结构。

思路：
    1. 读彩色图与 16 位毫米深度图，按内参反投影：X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=depth/1000；
    2. 丢掉无效深度与过远点，按步长降采样控制点数；
    3. 用画家算法把点按深度从远到近投影到画布上，颜色取原图颜色；
    4. 输出两个视角（正面 + 侧转 35°）的点云，并打印点云包围盒尺寸。

核心函数：np.meshgrid 反投影 / cv2.projectPoints 渲染
输出：output/pointcloud.png、pointcloud_front.png
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


def load_intrinsics():
    path = os.path.join(ASSETS, "depth_intrinsics.txt")
    if os.path.exists(path):
        fx, fy, cx, cy = np.loadtxt(path)
        return float(fx), float(fy), float(cx), float(cy)
    rgb = cv2.imread(os.path.join(ASSETS, "depth_rgb.png"))
    h, w = rgb.shape[:2]
    f = 1.2 * w
    return f, f, w / 2, h / 2


def backproject(rgb, depth_mm, intrinsics, step=2, z_max=8.0):
    fx, fy, cx, cy = intrinsics
    h, w = depth_mm.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    z = depth_mm[::step, ::step].astype(np.float32) / 1000.0
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb[::step, ::step].reshape(-1, 3)
    keep = (pts[:, 2] > 0.2) & (pts[:, 2] < z_max)
    return pts[keep], colors[keep]


def render(points, colors, yaw_deg=0.0, size=(760, 560), focal=520.0):
    """画家算法点云渲染：按相机系深度排序后逐点画圆。"""
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = points @ R.T
    order = np.argsort(-p[:, 2])
    p, c = p[order], colors[order]
    w, h = size
    u = (p[:, 0] / np.maximum(p[:, 2], 1e-3) * focal + w / 2).astype(np.int32)
    v = (-p[:, 1] / np.maximum(p[:, 2], 1e-3) * focal + h / 2).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    canvas = np.full((h, w, 3), 18, np.uint8)
    radius = 1 if len(p) > 40000 else 2
    for x, y, col in zip(u[inside], v[inside], c[inside]):
        cv2.circle(canvas, (int(x), int(y)), radius, tuple(int(v2) for v2 in col), -1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="RGB-D 反投影点云与三维渲染")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    rgb = cv2.imread(os.path.join(ASSETS, "depth_rgb.png"))
    depth = cv2.imread(os.path.join(ASSETS, "depth_map.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        sys.exit("缺少 depth_rgb.png / depth_map.png")
    intrinsics = load_intrinsics()
    print("内参：fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % intrinsics)

    pts, colors = backproject(rgb, depth, intrinsics, args.step)
    bbox = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(3)
    print("点云 %d 点，包围盒 %.2f x %.2f x %.2f m，深度范围 %.2f~%.2f m"
          % (len(pts), bbox[0], bbox[1], bbox[2], pts[:, 2].min(), pts[:, 2].max()))

    # 按深度上色（近红远蓝），结构比彩色纹理更容易看清
    dnorm = np.clip((pts[:, 2] - pts[:, 2].min()) / max(pts[:, 2].ptp(), 1e-6), 0, 1)
    depth_colors = cv2.applyColorMap((dnorm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    depth_colors = depth_colors.reshape(-1, 3)
    front = render(pts, depth_colors, 0.0)
    side = render(pts, depth_colors, 35.0)
    cv2.putText(front, "front view  %d pts  (color = depth)" % len(pts), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(side, "yaw 35 deg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(out_path("pointcloud_front.png"), front)
    cv2.imwrite(out_path("pointcloud.png"), np.hstack([front, side]))
    print("已保存 output/pointcloud.png、pointcloud_front.png")

    if args.display:
        cv2.imshow("cloud", np.hstack([front, side]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
