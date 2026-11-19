# -*- coding: utf-8 -*-
"""
@file    histogram_processing.py
@brief   直方图处理演示：直方图统计、全局均衡化与 CLAHE 自适应增强的对比。
@details 流程：计算直方图 -> 全局均衡化 -> 自适应 CLAHE -> 对比效果。
@note    依赖：opencv-python、numpy
         全局均衡化按整幅图的累积分布重映射灰度，会同时放大噪声；
         CLAHE 只在局部小块内做均衡并裁剪直方图峰值，因此更适合雾霾、逆光这类光照不均的图像。
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


def draw_histogram(gray, width, height):
    """
    @brief  把 256 级直方图画成柱状图，便于直观对比分布变化。
    @param  gray   灰度图。
    @param  width  输出宽度。
    @param  height 输出高度。
    @return 柱状图图像（BGR）。
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(1.0, hist.max()) * (height - 10)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    bin_width = max(1, width // 256)
    for i, value in enumerate(hist):
        cv2.rectangle(canvas, (i * bin_width, height - int(value)),
                      (i * bin_width + bin_width, height), (80, 80, 80), cv2.FILLED)
    return canvas


def main():
    """@brief 主流程：直方图、均衡化与 CLAHE 对比。"""
    parser = argparse.ArgumentParser(
        description="直方图处理演示：calcHist / equalizeHist / CLAHE")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/haze.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/histogram_processing.png")
    parser.add_argument("-l", "--clipLimit", type=float, default=2.0,
                        help="CLAHE 的对比度限制")
    parser.add_argument("-t", "--tileSize", type=int, default=8,
                        help="CLAHE 的网格数")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("haze.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 1：计算直方图并记录原始分布统计。
    hist_original = draw_histogram(gray, 512, PANEL_HEIGHT)

    # 步骤 2：全局均衡化。
    equalized = cv2.equalizeHist(gray)
    hist_equalized = draw_histogram(equalized, 512, PANEL_HEIGHT)

    # 步骤 3：自适应 CLAHE。按小块分别均衡，避免全局拉伸放大噪声。
    tile = max(1, args.tileSize)
    clahe = cv2.createCLAHE(clipLimit=args.clipLimit, tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray)
    hist_clahe = draw_histogram(enhanced, 512, PANEL_HEIGHT)

    # 步骤 4：对比效果，标准差越大说明对比度被拉得越开。
    print("输入: {}".format(input_path))
    print("{:<12}{:<10}{}".format("method", "mean", "std"))
    for name, image in [("original", gray), ("equalized", equalized), ("clahe", enhanced)]:
        print("{:<12}{:<10.1f}{:.1f}".format(name, float(image.mean()), float(image.std())))

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(hist_original, "histogram"),
        fit_panel(equalized, "equalizeHist"),
        fit_panel(hist_equalized, "histogram (eq)"),
        fit_panel(enhanced, "CLAHE"),
        fit_panel(hist_clahe, "histogram (clahe)"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "histogram_processing.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("histogram processing", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
