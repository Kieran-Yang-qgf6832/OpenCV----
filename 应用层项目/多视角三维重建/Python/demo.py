# -*- coding: utf-8 -*-
"""多视角三维重建：相邻视角位姿链式估计 + 尺度对齐 + 三角化稀疏重建。

流程：
    1. 每张视图提 SIFT 特征，相邻两帧做比值检验匹配（相邻视角重叠最大，最稳）；
    2. findEssentialMat + recoverPose 得到相邻帧的相对位姿（平移只有方向）；
    3. 用“同一批点在前后两段三角化里的模长比”把每段的平移尺度换算到同一基准；
    4. 沿链把位姿累乘到第 1 个相机的世界系，三角化所有内点并按重投影误差筛除离群点；
    5. 渲染两个视角的点云，并与合成真值对照相机夹角（每档 14°）。

核心函数：cv2.SIFT_create / cv2.findEssentialMat / cv2.recoverPose / cv2.triangulatePoints
输出：output/recon_cloud.png、recon_views.png
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
GT_STEP_DEG = 14.0


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_intrinsics():
    path = os.path.join(ASSETS, "intrinsics_gt.txt")
    fx, fy, cx, cy = np.loadtxt(path)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def ratio_match(desc_a, desc_b, ratio=0.8):
    knn = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    return [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < ratio * n.distance]


def triangulate(P_a, P_b, p_a, p_b):
    X = cv2.triangulatePoints(P_a, P_b, p_a.T, p_b.T)
    w = X[3]
    ok = np.abs(w) > 1e-9
    X = (X[:3] / np.where(ok, w, 1.0)).T
    return X, ok


def render(points, colors, yaw_deg=0.0, size=(760, 560), focal=620.0, radius=2):
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = points - np.median(points, axis=0)
    p = p @ R.T
    order = np.argsort(-p[:, 2])
    p, c = p[order], colors[order]
    w, h = size
    dist = float(np.percentile(np.abs(p[:, :2]), 95)) or 1.0
    u = (p[:, 0] / (p[:, 2] + dist) * focal + w / 2).astype(np.int32)
    v = (-p[:, 1] / (p[:, 2] + dist) * focal + h / 2).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    canvas = np.full((h, w, 3), 14, np.uint8)
    for x, y, col in zip(u[inside], v[inside], c[inside]):
        cv2.circle(canvas, (int(x), int(y)), radius, tuple(int(t) for t in col), -1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="多视图稀疏重建（相邻帧链式位姿）")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ASSETS, "view_*.png")))
    if len(files) < 3:
        sys.exit("至少需要 3 张 view_*.png")
    images = [cv2.imread(p) for p in files]
    K = load_intrinsics()
    print("视图 %d 张，内参 fx=%.1f cx=%.1f cy=%.1f" % (len(images), K[0, 0], K[0, 2], K[1, 2]))

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.008, edgeThreshold=12)
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in images]
    pts2d = [np.float32([k.pt for k in f[0]]) for f in feats]
    colors2d = [np.array([images[i][int(k.pt[1]), int(k.pt[0])] for k in f[0]], np.uint8)
                for i, f in enumerate(feats)]

    steps = []
    for i in range(len(images) - 1):
        pairs = ratio_match(feats[i][1], feats[i + 1][1])
        if len(pairs) < 30:
            print("  视角 %d-%d 匹配太少（%d），链断裂" % (i + 1, i + 2, len(pairs)))
            steps.append(None)
            continue
        idx_a = np.array([p[0] for p in pairs])
        idx_b = np.array([p[1] for p in pairs])
        pa, pb = pts2d[i][idx_a], pts2d[i + 1][idx_b]
        E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        ret, R, t, _ = cv2.recoverPose(E, pa, pb, K, mask=mask)
        keep = mask.ravel() > 0
        idx_a, idx_b, pa, pb = idx_a[keep], idx_b[keep], pa[keep], pb[keep]

        P_a = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_b = K @ np.hstack([R, t])
        X_unit, ok = triangulate(P_a, P_b, pa, pb)
        steps.append({"R": R, "t": t.ravel(), "idx_a": idx_a, "idx_b": idx_b,
                      "X": X_unit, "ok": ok, "pa": pa, "pb": pb})
        print("  视角 %d-%d：匹配 %d，内点 %d，夹角 %.1f 度"
              % (i + 1, i + 2, len(pairs), keep.sum(),
                 np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))

    # 尺度链：后一段与前一段在公共视角上的 3D 模长比
    scales = []
    for i, st in enumerate(steps):
        if st is None:
            scales.append(1.0)
            continue
        if i == 0 or steps[i - 1] is None:
            scales.append(1.0)
            continue
        prev = steps[i - 1]
        # 前一段在公共视角 i 下观测到的 3D 点：X' = R_{i-1} X + t_{i-1}
        Xa = prev["X"] @ prev["R"].T + prev["t"]
        key_prev = {int(b): j for j, b in enumerate(prev["idx_b"])}
        key_cur = {int(a): j for j, a in enumerate(st["idx_a"])}
        common = [k for k in key_cur if k in key_prev]
        ratios = []
        for k in common:
            n_cur = np.linalg.norm(st["X"][key_cur[k]])
            n_prev = np.linalg.norm(Xa[key_prev[k]])
            if n_prev > 1e-6 and np.isfinite(n_cur) and np.isfinite(n_prev):
                ratios.append(n_cur / n_prev)
        r = float(np.median(ratios)) if len(ratios) >= 8 else 1.0
        scales.append(scales[-1] * r if abs(r - 1.0) > 1e-3 else scales[-1])
        print("  尺度链：视角 %d 与前一阶段公共点 %d 个，尺度比 %.3f -> 累计 %.3f"
              % (i + 1, len(common), r, scales[-1]))

    # 位姿链：X_i = R_{i-1} X_{i-1} + t_{i-1} * s_{i-1}
    Rw, tw = [np.eye(3)], [np.zeros(3)]
    for i, st in enumerate(steps):
        if st is None:
            Rw.append(Rw[-1])
            tw.append(tw[-1])
            continue
        s = scales[i]
        Rw.append(st["R"] @ Rw[-1])
        tw.append(st["R"] @ tw[-1] + st["t"] * s)

    clouds, cols = [], []
    for i, st in enumerate(steps):
        if st is None:
            continue
        s = scales[i]
        Xw = (st["X"] * s - tw[i]) @ Rw[i]          # 转到世界系
        ok = st["ok"] & np.isfinite(Xw).all(axis=1) & (Xw[:, 2] > 0.1)
        # 用该视角自己的位姿投影回去，与它的 2D 点比较，剔除三角化离群点
        P_i = K @ np.hstack([Rw[i], tw[i].reshape(3, 1)])
        proj = (P_i @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T
        proj = proj[:, :2] / np.where(np.abs(proj[:, 2:3]) > 1e-9, proj[:, 2:3], 1.0)
        err = np.linalg.norm(proj - pts2d[i][st["idx_a"]], axis=1)
        ok &= err < 4.0
        clouds.append(Xw[ok])
        cols.append(colors2d[i][st["idx_a"][ok]])

    pts = np.vstack(clouds)
    cols_ = np.vstack(cols)
    inl_all = sum(len(c) for c in clouds)
    print("三角化 %d 点，通过检验 %d 点" % (sum(int(s["ok"].sum()) for s in steps if s), inl_all))
    if len(pts) == 0:
        sys.exit("没有重建出有效三维点")

    # 去掉深度离群点（三角化会在极远处产生少量噪点），否则渲染时整团点被压扁
    lo, hi = np.percentile(pts[:, 2], [1, 97])
    keep = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
    pts, cols_ = pts[keep], cols_[keep]
    print("剔除深度离群点后保留 %d 点（深度 %.2f~%.2f）" % (len(pts), pts[:, 2].min(), pts[:, 2].max()))

    angles = [np.degrees(np.arccos(np.clip((np.trace(Rw[i]) - 1) / 2, -1, 1))) for i in range(len(Rw))]
    print("视角 1 与各视角夹角估计：%s" % ["%.1f" % a for a in angles[1:]])
    print("合成真值：%s（每档 %.0f°）" % (["%.0f" % (GT_STEP_DEG * i) for i in range(1, len(angles))], GT_STEP_DEG))
    print("点云深度范围 %.2f~%.2f（相对尺度）" % (pts[:, 2].min(), pts[:, 2].max()))

    # 逐视角渲染：分段点云各自成型，拼合误差一眼可见（README 有说明）
    per_view = []
    for i, (c, col) in enumerate(zip(clouds, cols)):
        if len(c) < 20:
            continue
        img = render(c, col, 0.0, size=(380, 300), focal=420.0)
        cv2.putText(img, "view %d  %d pts" % (i + 1, len(c)), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        per_view.append(img)
    if per_view:
        while len(per_view) % 3:
            per_view.append(np.zeros_like(per_view[0]))
        cv2.imwrite(out_path("recon_perview.png"), np.vstack([np.hstack(per_view[i:i + 3])
                                                             for i in range(0, len(per_view), 3)]))

    front = render(pts, cols_, 0.0)
    side = render(pts, cols_, 40.0)
    cv2.putText(front, "reconstruction  %d pts" % len(pts), (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(side, "yaw 40 deg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(out_path("recon_cloud.png"), np.hstack([front, side]))

    thumbs = [cv2.resize(im, (266, 200)) for im in images]
    while len(thumbs) % 3:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(out_path("recon_views.png"), np.vstack([np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)]))
    print("已保存 output/recon_cloud.png、recon_views.png")

    if args.display:
        cv2.imshow("recon", np.hstack([front, side]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
