# -*- coding: utf-8 -*-
"""
@file    distance_transform.py
@brief   距离变换演示：计算每个前景像素到最近背景像素的距离，并对比三种距离度量。
@details 流程：二值化 -> 计算距离变换 -> 归一化显示 -> 由距离峰值提取前景种子（配合分水岭）。
@note    依赖：opencv-python、numpy
         DIST_L2 给出欧氏距离，是对「物体粗细」最直观的度量；
         DIST_L1 是城市街区距离，呈菱形扩散；DIST_C 是棋盘距离，呈方形扩散。
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


def distance_view(distance):
    """
    @brief  把浮点距离图归一化后上色，便于观察距离分布。
    @param  distance 距离图（float32）。
    @return BGR 伪彩色图。
    """
    normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def main():
    """@brief 主流程：三种距离度量对比，并由此提取分水岭种子。"""
    parser = argparse.ArgumentParser(
        description="距离变换演示：DIST_L1 / DIST_L2 / DIST_C 对比与种子提取")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/distance_transform.png")
    parser.add_argument("-s", "--seedRatio", type=float, default=0.5,
                        help="前景种子的阈值，占最大距离的比例")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景（距离变换只对非零像素计算）。
    _, foreground = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

    # 步骤 2：计算距离变换。第三个参数是掩膜尺寸，3 或 5 表示使用 3x3 / 5x5 邻域。
    distance_l1 = cv2.distanceTransform(foreground, cv2.DIST_L1, 3)
    distance_l2 = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    distance_c = cv2.distanceTransform(foreground, cv2.DIST_C, 3)

    # 步骤 3：归一化显示。
    view_l1 = distance_view(distance_l1)
    view_l2 = distance_view(distance_l2)
    view_c = distance_view(distance_c)

    # 步骤 4：结合分水岭。距离峰值即各物体的中心，阈值化后可作为前景种子。
    max_l2 = float(distance_l2.max())
    _, seeds = cv2.threshold(distance_l2, args.seedRatio * max_l2, 255, cv2.THRESH_BINARY)
    seed_count = cv2.connectedComponents(seeds.astype(np.uint8))[0] - 1

    print("输入: {}".format(input_path))
    print("  DIST_L1 最大距离: {:.2f}".format(float(distance_l1.max())))
    print("  DIST_L2 最大距离: {:.2f}".format(max_l2))
    print("  DIST_C  最大距离: {:.2f}".format(float(distance_c.max())))
    print("  阈值 {}*max 提取的种子数: {}".format(args.seedRatio, seed_count))

    panels = [
        fit_panel(foreground, "foreground (inverted)"),
        fit_panel(view_l1, "DIST_L1 (manhattan)"),
        fit_panel(view_l2, "DIST_L2 (euclidean)"),
        fit_panel(view_c, "DIST_C (chessboard)"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "distance_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("distance transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
