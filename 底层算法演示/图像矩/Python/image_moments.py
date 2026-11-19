# -*- coding: utf-8 -*-
"""
@file    image_moments.py
@brief   图像矩演示：计算轮廓的几何矩、质心与 Hu 矩，并用 Hu 矩做形状匹配。
@details 流程：提取轮廓 -> 计算图像矩 -> 计算质心 -> 计算 Hu 矩 -> 形状匹配。
@note    依赖：opencv-python、numpy
         Hu 矩有 7 个分量，对平移、缩放、旋转不变；数值跨越多阶数量级，
         比较前通常取 -sign(h)·log10|h|，本演示打印时直接给出 log 值。
"""

import argparse
import math
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30


def locate_asset(file_name):
    """@brief 在若干候选位置中查找素材，便于从不同工作目录运行。"""
    candidates = [
        os.path.join("..", "..", "assets", file_name),
        os.path.join("..", "assets", file_name),
        os.path.join("assets", file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def output_dir_for(input_path):
    """@brief 由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。"""
    directory = os.path.dirname(input_path)
    if os.path.basename(directory) == "assets":
        directory = os.path.dirname(directory)
    return os.path.join(directory, "output")


def fit_panel(image, title):
    """@brief 等比缩放并居中留白到固定面板。"""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    scale = min(PANEL_WIDTH / bgr.shape[1], PANEL_HEIGHT / bgr.shape[0])
    scaled = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.full((PANEL_HEIGHT + TITLE_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    x = (PANEL_WIDTH - scaled.shape[1]) // 2
    y = TITLE_HEIGHT + (PANEL_HEIGHT - scaled.shape[0]) // 2
    panel[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def montage(panels, cols):
    """@brief 按列数拼接等尺寸面板。"""
    rows = (len(panels) + cols - 1) // cols
    h, w = panels[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        canvas[r * h:r * h + panel.shape[0], c * w:c * w + panel.shape[1]] = panel[:h, :w]
    return canvas


def main():
    """@brief 主流程：逐轮廓计算矩特征并与第一个轮廓做形状匹配。"""
    parser = argparse.ArgumentParser(
        description="图像矩演示：moments / centroid / HuMoments / matchShapes")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_moments.png")
    parser.add_argument("-a", "--minArea", type=float, default=500,
                        help="最小面积过滤阈值")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：提取轮廓。素材白底黑形状，取反后形状成为前景。
    _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= args.minArea]
    if not kept:
        raise RuntimeError("没有面积达到阈值的轮廓")

    binary_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    moment_view = binary_view.copy()

    print("输入: {}, 有效轮廓: {}".format(input_path, len(kept)))
    print("{:<4}{:<10}{:<20}{:<10}{:<10}{}".format(
        "id", "area", "centroid", "hu[0]", "hu[1]", "match"))

    # 步骤 2~4：图像矩、质心、Hu 矩、形状匹配。
    for index, contour in enumerate(kept):
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        hu = cv2.HuMoments(moments).ravel()
        # Hu 矩跨数量级，取 -sign(h)·log10|h| 才便于比较与显示。
        hu0 = -math.copysign(1.0, hu[0]) * math.log10(abs(hu[0]) + 1e-30)
        hu1 = -math.copysign(1.0, hu[1]) * math.log10(abs(hu[1]) + 1e-30)
        # matchShapes 越小越相似；这里统一和第一个轮廓比较。
        match = cv2.matchShapes(contour, kept[0], cv2.CONTOURS_MATCH_I1, 0.0)

        print("{:<4}{:<10}{:<20}{:<10.2f}{:<10.2f}{:.4f}".format(
            index, int(cv2.contourArea(contour)),
            "{}, {}".format(int(cx), int(cy)), hu0, hu1, match))

        cv2.drawContours(moment_view, kept, index, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(moment_view, (int(round(cx)), int(round(cy))), 6, (0, 0, 255),
                   cv2.FILLED, cv2.LINE_AA)
        cv2.putText(moment_view, "#{}".format(index),
                    (int(round(cx)) + 8, int(round(cy)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # 第三格：按 matchShapes 得分着色（越相似越偏绿）。
    match_view = binary_view.copy()
    for index, contour in enumerate(kept):
        match = cv2.matchShapes(contour, kept[0], cv2.CONTOURS_MATCH_I1, 0.0)
        green = max(0, 255 - int(match * 120.0))
        cv2.drawContours(match_view, kept, index, (0, green, 255 - green), cv2.FILLED,
                         cv2.LINE_AA)

    panels = [
        fit_panel(binary, "binary input"),
        fit_panel(moment_view, "centroids & Hu"),
        fit_panel(match_view, "matchShapes score"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_moments.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image moments", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
