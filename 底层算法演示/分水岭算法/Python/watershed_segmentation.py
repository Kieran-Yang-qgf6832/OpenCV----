# -*- coding: utf-8 -*-
"""
@file    watershed_segmentation.py
@brief   分水岭分割演示：用距离变换 + 标记控制的分水岭分离相互接触的物体。
@details 流程：二值化 -> 距离变换 -> 标记前景背景 -> 应用分水岭 -> 着色输出。
@note    依赖：opencv-python、numpy
         直接对梯度图跑分水岭会因噪声产生大量过分割，所以必须先用距离变换找出
         每个物体的「种子」（确定的内部区域），再把未知区域交给分水岭漫水填充。
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


def colorize_markers(markers, label_count, seed=20261119):
    """
    @brief  按分水岭标记给每个物体上色，边界画成红色。
    @param  markers     分水岭输出的标记图（-1 表示边界，1 表示背景）。
    @param  label_count 连通域数量。
    @param  seed        随机种子，保证每次结果一致。
    @return 着色结果。
    """
    rng = np.random.default_rng(seed)
    palette = np.full((label_count + 2, 3), 255, dtype=np.uint8)
    palette[2:label_count + 1] = rng.integers(40, 255, (label_count - 1, 3), dtype=np.uint8)

    colorized = np.full((markers.shape[0], markers.shape[1], 3), 255, dtype=np.uint8)
    safe = markers.copy()
    safe[safe < 0] = 0
    colorized[:] = palette[safe]
    colorized[markers == -1] = (0, 0, 255)  # 分水岭边界
    return colorized


def main():
    """@brief 主流程：二值化、距离变换、标记、分水岭与着色。"""
    parser = argparse.ArgumentParser(
        description="分水岭算法演示：距离变换 + 标记控制的分水岭分割")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/coins.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/watershed_segmentation.png")
    parser.add_argument("-f", "--fgRatio", type=float, default=0.4,
                        help="前景种子阈值，占最大距离的比例")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("coins.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：二值化。硬币比背景亮，Otsu 能自动找到阈值。
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 步骤 2：距离变换。每个前景像素到最近背景的距离，峰值即各物体的中心。
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    distance_view = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(distance, args.fgRatio * distance.max(), 255, cv2.THRESH_BINARY)
    sure_fg = cv2.morphologyEx(sure_fg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 步骤 3：标记前景与背景。膨胀后的二值图是确定背景，两者之差是未知区域。
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    label_count, labels = cv2.connectedComponents(sure_fg)
    markers = labels.astype(np.int32) + 1  # 背景标记为 1，物体从 2 开始
    markers[unknown == 255] = 0  # 未知区域置 0，交给分水岭决定归属

    # 步骤 4：应用分水岭，边界会被标记成 -1。
    cv2.watershed(src, markers)

    colorized = colorize_markers(markers, label_count)
    overlay = src.copy()
    overlay[markers == -1] = (0, 0, 255)

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(binary, "otsu binary"),
        fit_panel(distance_view, "distance transform"),
        fit_panel(sure_fg, "sure foreground (seeds)"),
        fit_panel(colorized, "watershed labels"),
        fit_panel(overlay, "boundaries on original"),
    ]
    result = montage(panels, 3)

    print("输入: {}, Otsu 阈值 = {:.0f}".format(input_path, otsu))
    print("检测到物体数: {}".format(label_count - 1))

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "watershed_segmentation.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("watershed segmentation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
