# -*- coding: utf-8 -*-
"""
@file    hough_transform.py
@brief   霍夫变换演示：标准霍夫直线变换与概率霍夫直线变换。
@details 流程：边缘检测 -> 霍夫空间累加 -> 峰值检测 -> 绘制直线/线段。
@note    依赖：opencv-python、numpy
         标准 HoughLines 输出 (rho, theta) 参数对，得到的是贯穿全图的直线；
         HoughLinesP 直接给出线段端点，工程上更常用。
         霍夫圆需要画面中存在圆形目标，本素材（铁路）只有直线，故只演示直线。
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
    """@brief 主流程：边缘检测 + 两种霍夫直线变换。"""
    parser = argparse.ArgumentParser(
        description="霍夫变换演示：HoughLines + HoughLinesP")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/railway.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/hough_transform.png")
    parser.add_argument("-t", "--linesThresh", type=float, default=400,
                        help="标准霍夫的累加器阈值")
    parser.add_argument("-p", "--pThresh", type=float, default=180,
                        help="概率霍夫的累加器阈值")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("railway.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：边缘检测。霍夫变换的输入必须是边缘图，阈值的选取直接影响结果数量。
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 步骤 2~4：累加、取峰值并绘制。标准霍夫得到的是无限长直线的参数对。
    lines = cv2.HoughLines(edges, 1, np.pi / 180, args.linesThresh)
    line_view = src.copy()
    count = 0 if lines is None else len(lines)
    for rho, theta in ([] if lines is None else lines[:, 0, :]):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x0, y0 = cos_t * rho, sin_t * rho
        # 沿直线方向取足够长的两点，保证画到画面之外。
        p1 = (int(x0 + 2000 * (-sin_t)), int(y0 + 2000 * cos_t))
        p2 = (int(x0 - 2000 * (-sin_t)), int(y0 - 2000 * cos_t))
        cv2.line(line_view, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, args.pThresh,
                               minLineLength=150, maxLineGap=8)
    segment_view = src.copy()
    seg_count = 0 if segments is None else len(segments)
    for x1, y1, x2, y2 in ([] if segments is None else segments[:, 0, :]):
        cv2.line(segment_view, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    print("标准霍夫直线数: {}".format(count))
    print("概率霍夫线段数: {}".format(seg_count))
    for rho, theta in ([] if lines is None else lines[:5, 0, :]):
        print("  rho={:.1f}, theta={:.1f} 度".format(rho, math.degrees(theta)))

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(edges, "canny edges"),
        fit_panel(line_view, "HoughLines: {}".format(count)),
        fit_panel(segment_view, "HoughLinesP: {}".format(seg_count)),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "hough_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("hough transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
