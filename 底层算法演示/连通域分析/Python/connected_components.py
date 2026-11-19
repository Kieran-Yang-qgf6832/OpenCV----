# -*- coding: utf-8 -*-
"""
@file    connected_components.py
@brief   连通域分析演示：标记二值图中的连通区域并统计面积、包围盒、质心。
@details 流程：二值化 -> 连通域标记 -> 计算统计信息 -> 按条件过滤并可视化。
@note    依赖：opencv-python、numpy
         connectedComponents 只返回标记图与数量；需要面积/包围盒/质心时用
         connectedComponentsWithStats，它一次性给出全部统计量，避免再遍历像素。
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


def colorize_labels(labels, count, seed=20261119):
    """
    @brief  按标记值给连通域上色（同一标签同色）。
    @param  labels 标记图（int32）。
    @param  count  标签数（含背景 0）。
    @param  seed   随机种子，保证结果可复现。
    @return BGR 着色图。
    """
    rng = np.random.default_rng(seed)
    palette = np.zeros((count, 3), dtype=np.uint8)
    palette[1:] = rng.integers(50, 255, (count - 1, 3), dtype=np.uint8)
    return palette[labels]


def main():
    """@brief 主流程：标记连通域、统计并按面积过滤。"""
    parser = argparse.ArgumentParser(
        description="连通域分析演示：标记、统计与面积过滤")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/connected_components.png")
    parser.add_argument("-a", "--minArea", type=int, default=200,
                        help="最小面积过滤阈值，小于该值视为噪声")
    parser.add_argument("-c", "--connectivity", type=int, default=8, choices=[4, 8],
                        help="邻接方式，4 或 8")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景。
    _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

    # 步骤 2~3：标记并一次性取出统计信息。
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=args.connectivity, ltype=cv2.CV_32S)

    # 步骤 4：按面积过滤。
    overlay = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    kept = 0
    print("输入: {}, 邻接方式: {} 邻接".format(input_path, args.connectivity))
    print("  原始连通域(含背景): {}".format(count))
    print("{:<6}{:<10}{:<20}{}".format("label", "area", "bbox(x,y,w,h)", "centroid"))
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < args.minArea:
            continue
        kept += 1
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(i), (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        if kept <= 10:
            print("{:<6}{:<10}{:<20}{}".format(
                i, area, "{},{},{},{}".format(x, y, w, h),
                "({:.1f}, {:.1f})".format(centroids[i][0], centroids[i][1])))
    print("  面积 >= {} 的连通域: {} 个".format(args.minArea, kept))

    colored = colorize_labels(labels, count)
    panels = [
        fit_panel(binary, "binary input"),
        fit_panel(colored, "labels: {}".format(count - 1)),
        fit_panel(overlay, "area >= {}: {}".format(args.minArea, kept)),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "connected_components.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("connected components", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
