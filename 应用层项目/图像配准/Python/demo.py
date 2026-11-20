# -*- coding: utf-8 -*-
"""图像配准：SIFT 特征 + RANSAC 单应，把变形图对齐回参考图并做叠加验证。

思路：
    1. 两张图各提 SIFT 关键点与 128 维描述子；
    2. BFMatcher + Lowe 比值检验筛匹配对；
    3. findHomography(RANSAC) 求 H，内点数作为配准置信度；
    4. 用 H 反向映射变形图，与参考图 50/50 叠加 + 棋盘格对照，内点匹配线一并画出。

核心函数：cv2.SIFT_create / cv2.BFMatcher.knnMatch / cv2.findHomography / cv2.warpPerspective
输出：output/registration.png、registration_matches.png
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


def tile_label(img, text, size=(512, 512)):
    out = cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="SIFT + RANSAC 图像配准")
    ap.add_argument("--ref", default=os.path.join(ASSETS, "scene.png"))
    ap.add_argument("--moving", default=os.path.join(ASSETS, "scene_warped.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    ref = cv2.imread(args.ref)
    mov = cv2.imread(args.moving)
    if ref is None or mov is None:
        sys.exit("读不到 scene.png / scene_warped.png")
    if mov.shape[:2] != ref.shape[:2]:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]))

    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), None)
    print("关键点：参考 %d，待配准 %d" % (len(k1), len(k2)))

    matcher = cv2.BFMatcher()
    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    print("比值检验后匹配对：%d" % len(good))
    if len(good) < 8:
        sys.exit("匹配对太少，无法估计单应")

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)
    inliers = int(mask.sum())
    print("RANSAC 内点：%d / %d（%.0f%%）" % (inliers, len(good), 100.0 * inliers / len(good)))
    print("单应矩阵 H =\n", np.array2string(H, precision=4))

    aligned = cv2.warpPerspective(mov, H, (ref.shape[1], ref.shape[0]))
    blend = cv2.addWeighted(ref, 0.5, aligned, 0.5, 0)
    cv2.imwrite(out_path("registration.png"), np.hstack([
        tile_label(mov, "moving image"), tile_label(aligned, "aligned by H"),
        tile_label(blend, "50% blend")]))
    print("已保存 output/registration.png（待配准 / 对齐后 / 50% 叠加）")

    lines = cv2.drawMatches(ref, k1, mov, k2, [g for g, keep in zip(good, mask.ravel()) if keep],
                            None, (0, 255, 0), (0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    lines = cv2.resize(lines, (1400, int(lines.shape[0] * 1400 / lines.shape[1])))
    cv2.putText(lines, "SIFT inliers %d/%d" % (inliers, len(good)), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(out_path("registration_matches.png"), lines)
    print("已保存 output/registration_matches.png")

    if args.display:
        cv2.imshow("blend", cv2.resize(blend, (720, 720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
