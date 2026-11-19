# -*- coding: utf-8 -*-
"""
@file    corner_detection.py
@brief   角点检测演示：Harris 响应与 Shi-Tomasi（goodFeaturesToTrack）对比。
@details 流程：灰度转换 -> 计算角点响应 -> 阈值筛选 -> 非极大值抑制 -> 标注角点。
@note    依赖：opencv-python、numpy
         Harris 用响应图 R = det(M) - k·trace(M)²，需要自己阈值 + 非极大值抑制；
         Shi-Tomasi 直接取 M 的较小特征值，goodFeaturesToTrack 内部已做 NMS 与最小间距。
"""

import argparse
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


def harris_corners(gray, threshold):
    """
    @brief  Harris 角点检测：响应阈值 + 非极大值抑制。
    @param  gray      灰度图。
    @param  threshold 归一化后响应的相对阈值（占最大值的比例）。
    @return 角点坐标数组 (N, 2)。
    """
    response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    normalized = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)
    # 非极大值抑制：只有本像素等于邻域最大值时才保留，避免角点成片。
    # 窗口取 9x9：砖墙纹理极密，3x3 窗口压不住，会留下上万个角点。
    dilated = cv2.dilate(normalized, np.ones((9, 9), np.uint8))
    mask = (normalized > threshold) & (normalized >= dilated)
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys])


def main():
    """@brief 主流程：两种角点检测算法的结果对比。"""
    parser = argparse.ArgumentParser(
        description="角点检测演示：Harris 与 Shi-Tomasi 对比")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/brick.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/corner_detection.png")
    parser.add_argument("-t", "--harrisThresh", type=float, default=0.5,
                        help="Harris 相对阈值（占归一化响应的比例）")
    parser.add_argument("-m", "--maxCorners", type=int, default=200,
                        help="Shi-Tomasi 最大角点数")
    parser.add_argument("-q", "--minDistance", type=float, default=10,
                        help="Shi-Tomasi 角点最小间距")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("brick.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：灰度转换。
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 2~4：Harris 响应、阈值、非极大值抑制。
    harris = harris_corners(gray, args.harrisThresh)
    harris_view = src.copy()
    for x, y in harris:
        cv2.circle(harris_view, (int(x), int(y)), 3, (0, 0, 255), 1, cv2.LINE_AA)

    response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    response_view = cv2.applyColorMap(
        cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)

    # Shi-Tomasi：goodFeaturesToTrack 内部已完成质量筛选与最小间距抑制。
    points = cv2.goodFeaturesToTrack(
        gray, max(1, args.maxCorners), args.harrisThresh, args.minDistance)
    points = np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)
    shi_view = src.copy()
    for x, y in points:
        cv2.circle(shi_view, (int(round(x)), int(round(y))), 4, (0, 255, 0), 2, cv2.LINE_AA)

    print("输入: {} ({}x{})".format(input_path, gray.shape[1], gray.shape[0]))
    print("Harris 角点数: {} (阈值 {})".format(len(harris), args.harrisThresh))
    print("Shi-Tomasi 角点数: {} (上限 {})".format(len(points), args.maxCorners))

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(response_view, "harris response"),
        fit_panel(harris_view, "harris NMS: {}".format(len(harris))),
        fit_panel(shi_view, "shi-tomasi: {}".format(len(points))),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "corner_detection.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("corner detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
