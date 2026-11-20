# -*- coding: utf-8 -*-
"""全景图像拼接：用 Stitcher 把三张有重叠的视图拼成全景，并给出接缝检查图。

思路：
    1. Stitcher(SCANS) 内部做特征匹配 + 相机参数估计 + 接缝搜索 + 多频段融合；
    2. 拼接成功时输出全景图，并把每张输入按顺序排在下方作为来源对照；
    3. 额外输出一张“接缝带”灰度梯度图，便于检查融合有没有重影；
    4. 失败时打印 Stitcher 返回状态码，便于定位是哪一步不满足。

核心函数：cv2.Stitcher_create / stitcher.stitch
输出：output/panorama.png、panorama_sources.png、panorama_seam.png
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

CODES = {0: "OK", 1: "需要更多图像", 2: "图像互相不匹配", 3: "相机参数估计失败"}


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def main():
    ap = argparse.ArgumentParser(description="多视图全景拼接")
    ap.add_argument("--mode", default="SCANS", choices=["SCANS", "PANORAMA"])
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ASSETS, "pan_*.png")))
    if len(files) < 2:
        sys.exit("至少需要两张 pan_*.png")
    images = [cv2.imread(p) for p in files]
    print("输入 %d 张，尺寸 %s" % (len(images), [im.shape[1::-1] for im in images]))

    mode = cv2.Stitcher_SCANS if args.mode == "SCANS" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(mode)
    status, pano = stitcher.stitch(images)
    print("拼接状态码 %d（%s）" % (status, CODES.get(status, "未知")))
    if status != 0 or pano is None:
        sys.exit("拼接失败，检查输入是否有足够重叠")

    print("全景尺寸 %dx%d，放大 %.1f 倍" % (pano.shape[1], pano.shape[0],
                                          pano.shape[1] / float(images[0].shape[1])))
    cv2.imwrite(out_path("panorama.png"), pano)

    thumbs = [cv2.resize(im, (pano.shape[1] // len(images), int(im.shape[0] * (pano.shape[1] / len(images)) / im.shape[1])))
              for im in images]
    row = np.hstack(thumbs)
    for i, im in enumerate(thumbs):
        cv2.putText(im, "#%d" % (i + 1), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(out_path("panorama_sources.png"), np.vstack([cv2.resize(pano, (row.shape[1], int(pano.shape[0] * row.shape[1] / pano.shape[1]))), row]))

    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    seam = cv2.normalize(np.abs(edges), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(out_path("panorama_seam.png"), cv2.applyColorMap(seam, cv2.COLORMAP_INFERNO))
    print("已保存 output/panorama.png、panorama_sources.png、panorama_seam.png")

    if args.display:
        cv2.imshow("panorama", cv2.resize(pano, None, fx=0.6, fy=0.6))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
