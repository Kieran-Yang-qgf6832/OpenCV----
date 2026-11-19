# -*- coding: utf-8 -*-
"""
@file    hit_or_miss.py
@brief   击中击不中演示：用形态学核精确匹配特定尺寸的孤立形状。
@details 流程：定义击中/击不中核 -> 应用变换 -> 定位匹配 -> 统计并标记。
@note    依赖：opencv-python、numpy
         核的取值约定：1 表示该位置必须是前景，-1 表示必须是背景，0 表示不关心。
         素材是 5x5 块重复排布的图案（35x35 方块、17x17 方块、59x9 横杠各 25 个），
         因此可以用「尺寸 + 一圈背景」的核唯一锁定某一类形状。
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


def make_isolation_kernel(size):
    """
    @brief  构造「精确尺寸 + 一圈背景」的击中击不中核。
    @param  size 目标形状的边长（像素）。
    @return int8 核：内部 1（必须前景），外环 -1（必须背景）。
    """
    kernel = np.full((size + 2, size + 2), -1, dtype=np.int8)
    kernel[1:size + 1, 1:size + 1] = 1
    return kernel


def visualize_kernel(kernel, scale=12):
    """
    @brief  把核渲染成可看的灰度图（前景白、背景黑、不关心灰）。
    @param  kernel 击中击不中核。
    @param  scale  放大倍数。
    @return 可视化图像。
    """
    view = np.full(kernel.shape, 128, dtype=np.uint8)
    view[kernel > 0] = 255
    view[kernel < 0] = 0
    return cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def mark_hits(src, hits, color):
    """
    @brief  在彩色图上标出所有命中位置。
    @param  src   背景图。
    @param  hits  击中击不中结果（255 为命中）。
    @param  color 标记颜色。
    @return 标注后的图像。
    """
    view = src.copy()
    # 命中结果通常只有孤立的单个像素，先膨胀再画圈才看得见。
    dilated = cv2.dilate(hits, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.circle(view, (x + w // 2, y + h // 2), 26, color, 2, cv2.LINE_AA)
    return view


def main():
    """@brief 主流程：两类形状的匹配与标注。"""
    parser = argparse.ArgumentParser(
        description="击中击不中演示：用形态学核匹配特定尺寸的孤立形状")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/hit_or_miss.png")
    parser.add_argument("-s", "--small", type=int, default=17, help="小方块的边长(像素)")
    parser.add_argument("-l", "--large", type=int, default=35, help="大方块的边长(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：定义击中/击不中核。素材是白底黑形状，先取反让形状成为前景。
    _, foreground = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)
    small_size = max(3, args.small)
    large_size = max(3, args.large)
    small_kernel = make_isolation_kernel(small_size)
    large_kernel = make_isolation_kernel(large_size)

    # 步骤 2：应用变换。
    small_hits = cv2.morphologyEx(foreground, cv2.MORPH_HITMISS, small_kernel)
    large_hits = cv2.morphologyEx(foreground, cv2.MORPH_HITMISS, large_kernel)

    # 步骤 3：定位匹配。
    small_count = int(np.count_nonzero(small_hits))
    large_count = int(np.count_nonzero(large_hits))

    # 步骤 4：在彩色图上标注命中位置。
    color_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    small_view = mark_hits(color_src, small_hits, (0, 180, 0))
    large_view = mark_hits(color_src, large_hits, (0, 0, 255))

    print("输入: {}".format(input_path))
    print("  {}x{} 方块命中: {}".format(small_size, small_size, small_count))
    print("  {}x{} 方块命中: {}".format(large_size, large_size, large_count))

    panels = [
        fit_panel(foreground, "foreground (inverted)"),
        fit_panel(visualize_kernel(small_kernel), "kernel {}+ring".format(small_size)),
        fit_panel(small_view, "hits {}".format(small_count)),
        fit_panel(large_view, "hits {}".format(large_count)),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "hit_or_miss.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("hit or miss", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
