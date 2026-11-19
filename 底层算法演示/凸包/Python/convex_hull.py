# -*- coding: utf-8 -*-
"""
@file    convex_hull.py
@brief   凸包演示：计算轮廓的凸包与凸缺陷，并用实心度衡量形状的凹陷程度。
@details 流程：提取轮廓点集 -> 计算凸包 -> 计算凸缺陷 -> 可视化。
@note    依赖：opencv-python、numpy
         convexityDefects 返回的深度是定点数，需除以 256 才是像素值；
         并且它要求凸包用 returnPoints=False 得到索引，不能直接用点集。
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
    """@brief 主流程：凸包、凸缺陷与实心度统计。"""
    parser = argparse.ArgumentParser(
        description="凸包演示：convexHull + convexityDefects")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/convex_hull.png")
    parser.add_argument("-a", "--minArea", type=float, default=500,
                        help="最小面积过滤阈值")
    parser.add_argument("-d", "--minDepth", type=float, default=1.0,
                        help="凸缺陷的最小深度(像素)，滤掉抖动")
    parser.add_argument("-w", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：提取轮廓点集。素材白底黑形状，取反后形状成为前景。
    _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    binary_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    hull_view = binary_view.copy()
    defect_view = binary_view.copy()

    kept = 0
    total_defects = 0
    print("输入: {}, 轮廓总数: {}".format(input_path, len(contours)))
    print("{:<6}{:<10}{:<12}{:<10}{}".format("id", "area", "hullArea", "solidity", "defects"))

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < args.minArea:
            continue
        kept += 1

        # 步骤 2：计算凸包。两次调用：一次拿点集画图，一次拿索引算缺陷。
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # 步骤 3：计算凸缺陷。
        defects = 0
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defect_list = cv2.convexityDefects(contour, hull_indices)
            except cv2.error:
                defect_list = None
            if defect_list is not None:
                for start_i, end_i, far_i, depth_raw in defect_list[:, 0, :]:
                    depth = depth_raw / 256.0  # 定点数转像素
                    if depth < args.minDepth:
                        continue
                    defects += 1
                    start = tuple(contour[start_i][0])
                    end = tuple(contour[end_i][0])
                    far = tuple(contour[far_i][0])
                    cv2.line(defect_view, start, far, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(defect_view, end, far, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(defect_view, far, 5, (255, 0, 0), 1, cv2.LINE_AA)
        total_defects += defects

        cv2.drawContours(hull_view, [contour], -1, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.drawContours(hull_view, [hull], -1, (0, 180, 0), 2, cv2.LINE_AA)

        if kept <= 16:
            print("{:<6}{:<10}{:<12}{:<10.3f}{}".format(
                index, int(area), int(hull_area), solidity, defects))

    print("面积 >= {} 的轮廓: {} 个, 深度 >= {:.1f} 的凸缺陷共 {} 个".format(
        args.minArea, kept, args.minDepth, total_defects))

    panels = [
        fit_panel(binary, "binary contours"),
        fit_panel(hull_view, "convex hull (green)"),
        fit_panel(defect_view, "defects (red)"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "convex_hull.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("convex hull", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
