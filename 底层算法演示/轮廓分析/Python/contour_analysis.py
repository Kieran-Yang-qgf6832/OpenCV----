# -*- coding: utf-8 -*-
"""
@file    contour_analysis.py
@brief   轮廓分析演示：查找轮廓并计算面积、周长、圆度、多边形近似与包围盒。
@details 流程：二值化 -> 查找轮廓 -> 遍历计算属性 -> 绘制轮廓与标注。
@note    依赖：opencv-python、numpy
         面积用 contourArea（格林公式），周长用 arcLength；
         圆度 4πA/P² 对完美圆为 1，可用于形状分类。
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
    """@brief 主流程：提取轮廓、计算属性并可视化。"""
    parser = argparse.ArgumentParser(
        description="轮廓分析演示：面积/周长/圆度/多边形近似")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/contour_analysis.png")
    parser.add_argument("-a", "--minArea", type=float, default=500,
                        help="最小面积过滤阈值")
    parser.add_argument("-e", "--epsilon", type=float, default=0.02,
                        help="多边形近似的精度，占周长的比例")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景。
    _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

    # 步骤 2：查找外轮廓。
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    approx_view = contour_view.copy()
    box_view = contour_view.copy()

    # 步骤 3：遍历计算属性。
    rng = np.random.default_rng(20261119)
    kept = 0
    print("输入: {}, 轮廓总数: {}".format(input_path, len(contours)))
    print("{:<4}{:<10}{:<10}{:<10}{:<10}{}".format(
        "id", "area", "perim", "circular", "vertices", "bbox"))
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < args.minArea:
            continue
        kept += 1
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        approx = cv2.approxPolyDP(contour, args.epsilon * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        color = tuple(int(v) for v in rng.integers(40, 255, 3))

        cv2.drawContours(contour_view, contours, index, color, 2, cv2.LINE_AA)
        cv2.drawContours(approx_view, [approx], -1, (0, 0, 255), 2, cv2.LINE_AA)
        for pt in approx.reshape(-1, 2):
            cv2.circle(approx_view, (int(pt[0]), int(pt[1])), 4, (0, 255, 0),
                       cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(box_view, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(box_view, "A={}".format(int(area)), (x + 2, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if kept <= 12:
            print("{:<4}{:<10}{:<10.1f}{:<10.3f}{:<10}{}".format(
                index, int(area), perimeter, circularity, len(approx),
                "{},{},{},{}".format(x, y, w, h)))
    print("  面积 >= {} 的轮廓: {} 个".format(args.minArea, kept))

    panels = [
        fit_panel(binary, "binary input"),
        fit_panel(contour_view, "contours: {}".format(kept)),
        fit_panel(approx_view, "approxPolyDP (eps {:.2f})".format(args.epsilon)),
        fit_panel(box_view, "bounding boxes"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "contour_analysis.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("contour analysis", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
