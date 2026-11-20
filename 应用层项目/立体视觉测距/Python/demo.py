# -*- coding: utf-8 -*-
"""立体视觉测距：SGBM 求视差、换算深度，并用场景真值检验误差。

思路：
    1. 左右视图严格配准（合成时右相机沿 +X 平移基线 b），做 SGBM 匹配；
    2. 视差 d 与深度满足 Z = f*b/d，f、b 从 stereo_params.txt 读取；
    3. 与真值视差图逐像素比较，统计 bad-1px / bad-2px 与有效像素率；
    4. 在近处方块区域取中位数深度作为“测距结果”，与真值比出相对误差。

核心函数：cv2.StereoSGBM_create / compute / cv2.reprojectImageTo3D
输出：output/disparity.png、depth_compare.png、depth_map.png
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


def load_params():
    path = os.path.join(ASSETS, "stereo_params.txt")
    if not os.path.exists(path):
        sys.exit("缺少 assets/stereo_params.txt（focal 与 baseline）")
    f, b = np.loadtxt(path)
    return float(f), float(b)


def load_gt():
    path = os.path.join(ASSETS, "stereo_disp_gt.png")
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 16.0


def compute_disparity(left, right, block=5):
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=120, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    return disp


def colored(disp, vmax=40):
    vis = np.clip(disp / vmax * 255, 0, 255).astype(np.uint8)
    vis[disp <= 0] = 0
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def panel(img, text, size=(640, 360)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="SGBM 双目测距")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    left = cv2.imread(os.path.join(ASSETS, "stereo_left.png"))
    right = cv2.imread(os.path.join(ASSETS, "stereo_right.png"))
    if left is None or right is None:
        sys.exit("缺少 stereo_left.png / stereo_right.png")
    f, b = load_params()
    print("相机参数：f = %.1f px，baseline = %.3f m" % (f, b))

    disp = compute_disparity(left, right, args.block)
    valid = disp > 0
    depth = np.where(valid, f * b / np.maximum(disp, 1e-6), 0)
    print("有效视差像素 %.1f%%，视差范围 %.1f~%.1f px" % (100 * valid.mean(), disp[valid].min(), disp[valid].max()))

    gt = load_gt()
    if gt is not None:
        common = valid & (gt > 0)
        err = np.abs(disp[common] - gt[common])
        print("与真值对比（%d 个像素）：bad-1px %.2f%%，bad-2px %.2f%%，平均误差 %.2f px"
              % (common.sum(), 100 * (err > 1).mean(), 100 * (err > 2).mean(), err.mean()))
        # 取近处方块区域（真值视差最大的一段）评估测距精度
        band = common & (gt >= np.percentile(gt[gt > 0], 92))
        est_m = float(np.median(depth[band]))
        gt_m = float(np.median(f * b / np.maximum(gt[band], 1e-6)))
        print("近处目标测距：估计 %.3f m，真值 %.3f m，相对误差 %.2f%%"
              % (est_m, gt_m, 100 * abs(est_m - gt_m) / gt_m))

        gt_depth = np.where(gt > 0, f * b / np.maximum(gt, 1e-6), 0)
        diff = np.where(common, np.abs(depth - gt_depth), 0)
        diff_vis = np.clip(diff / 0.5 * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path("depth_compare.png"), np.hstack([
            panel(colored(disp), "estimated disparity"),
            panel(colored(gt), "ground-truth disparity"),
            panel(cv2.applyColorMap(diff_vis, cv2.COLORMAP_MAGMA), "depth error (<0.5m)")]))

    cv2.imwrite(out_path("disparity.png"), colored(disp))
    dn = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(out_path("depth_map.png"), cv2.applyColorMap(dn, cv2.COLORMAP_JET))
    print("已保存 output/disparity.png、depth_map.png" + ("、depth_compare.png" if gt is not None else ""))

    if args.display:
        cv2.imshow("disparity", colored(disp))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
