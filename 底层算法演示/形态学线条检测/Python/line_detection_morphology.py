# -*- coding: utf-8 -*-
"""
@file    line_detection_morphology.py
@brief   形态学线条检测演示：用方向结构元素提取表格的水平线与垂直线。
@details 流程：二值化 -> 创建方向结构元素 -> 开运算提取线条 -> 轮廓筛选。
@note    依赖：opencv-python、numpy
         核心思想：长度 1×N 的结构元素只能「容纳」水平方向的线条，开运算后其余结构被腐蚀掉；
         同理 1×M 的竖条只保留垂直线。结构元素长度要大于线宽、小于线长。
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


def filter_by_length(lines, min_len, horizontal):
    """
    @brief  过滤掉过短的线段，只保留真正的表格线。
    @param  lines      候选线条二值图。
    @param  min_len    最小长度。
    @param  horizontal 是否为水平方向（决定用宽还是高做判据）。
    @return 过滤后的二值图。
    """
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(lines)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = w if horizontal else h
        if length >= min_len:
            # 注意终点要减 1，否则相邻线段会连成一片，连通域计数会偏少
            cv2.rectangle(filtered, (x, y), (x + w - 1, y + h - 1), 255, cv2.FILLED)
    return filtered


def main():
    """@brief 主流程：提取水平线与垂直线并叠加显示。"""
    parser = argparse.ArgumentParser(
        description="形态学线条检测演示：方向结构元素 + 开运算")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/table.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/line_detection_morphology.png")
    parser.add_argument("-l", "--lineLen", type=int, default=0,
                        help="结构元素长度，0 表示自动取宽(高)的 1/25")
    parser.add_argument("-n", "--minLen", type=int, default=0,
                        help="线条最小长度，0 表示自动取宽(高)的 1/20")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("table.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：二值化。表格线通常是深色，取反后成为前景。
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, -2)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 步骤 2：创建方向结构元素。长度取画面尺寸的 1/25 左右即可滤掉文字。
    line_len = args.lineLen if args.lineLen > 0 else max(
        10, min(gray.shape[0], gray.shape[1]) // 25)
    horizontal_element = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vertical_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))

    # 步骤 3：开运算提取线条。
    horizontal = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horizontal_element)
    vertical = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical_element)

    # 步骤 4：轮廓筛选，去掉过短的残留。
    min_len = args.minLen if args.minLen > 0 else max(
        20, min(gray.shape[0], gray.shape[1]) // 20)
    horizontal = filter_by_length(horizontal, min_len, True)
    vertical = filter_by_length(vertical, min_len, False)

    horizontal_count = cv2.connectedComponents(horizontal)[0] - 1
    vertical_count = cv2.connectedComponents(vertical)[0] - 1
    print("输入: {}, 结构元素长度: {}, 最小线长: {}".format(input_path, line_len, min_len))
    print("  水平线段: {}, 垂直线段: {}".format(horizontal_count, vertical_count))

    # 叠加显示：水平线绿色，垂直线红色。
    overlay = src.copy()
    overlay[horizontal > 0] = (0, 255, 0)
    overlay[vertical > 0] = (0, 0, 255)
    combined = cv2.bitwise_or(horizontal, vertical)

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(cleaned, "binary"),
        fit_panel(horizontal, "horizontal"),
        fit_panel(vertical, "vertical"),
        fit_panel(combined, "combined"),
        fit_panel(overlay, "overlay on original"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "line_detection_morphology.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("line detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
