# -*- coding: utf-8 -*-
"""
@file    color_segmentation.py
@brief   颜色分割演示：在 HSV 空间按色相区间提取目标颜色区域。
@details 流程：转换 HSV -> 设定颜色阈值 -> 生成掩码提取目标 -> 形态学优化掩码。
@note    依赖：opencv-python、numpy
         HSV 的色相是环形的，红色跨 0° 附近，必须用两段区间（0~10 与 170~179）合并；
         直接在 BGR 上设阈值很难把「苹果的红」与「阴影的暗红」分开。
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


def main():
    """@brief 主流程：HSV 色相区间分割 + 形态学优化。"""
    parser = argparse.ArgumentParser(
        description="颜色分割演示：HSV 色相阈值 + inRange + 形态学")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/apples.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/color_segmentation.png")
    parser.add_argument("-n", "--minSaturation", type=int, default=90,
                        help="最小饱和度，滤掉发白区域")
    parser.add_argument("-v", "--minValue", type=int, default=60,
                        help="最小明度，滤掉过暗区域")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("apples.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：转换到 HSV。
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hue = cv2.split(hsv)[0]

    # 步骤 2：设定颜色阈值。红色跨色相 0 点，用两段区间合并。
    low = cv2.inRange(hsv, np.array([0, args.minSaturation, args.minValue], np.uint8),
                      np.array([10, 255, 255], np.uint8))
    high = cv2.inRange(hsv, np.array([170, args.minSaturation, args.minValue], np.uint8),
                       np.array([179, 255, 255], np.uint8))

    # 步骤 3：生成掩码并提取目标。
    mask = cv2.bitwise_or(low, high)
    extracted = cv2.bitwise_and(src, src, mask=mask)

    # 步骤 4：形态学优化掩码（开运算去噪、闭运算填洞）。
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, element)
    refined_extracted = cv2.bitwise_and(src, src, mask=refined)

    raw_ratio = 100.0 * cv2.countNonZero(mask) / mask.size
    refined_ratio = 100.0 * cv2.countNonZero(refined) / refined.size
    print("输入: {}, S>={}, V>={}".format(input_path, args.minSaturation, args.minValue))
    print("  原始掩码占比: {:.2f}%".format(raw_ratio))
    print("  形态学优化后: {:.2f}%".format(refined_ratio))

    panels = [
        fit_panel(src, "original"),
        fit_panel(hue, "hue channel"),
        fit_panel(mask, "raw mask"),
        fit_panel(refined, "refined mask"),
        fit_panel(extracted, "extracted (raw)"),
        fit_panel(refined_extracted, "extracted (refined)"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "color_segmentation.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("color segmentation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
