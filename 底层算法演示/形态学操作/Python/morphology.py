# -*- coding: utf-8 -*-
"""
@file    morphology.py
@brief   形态学操作演示：腐蚀、膨胀、开闭运算与形态学梯度。
@details 流程：定义结构元素 -> 腐蚀去噪 -> 膨胀填充 -> 开闭运算组合 -> 输出对比。
@note    依赖：opencv-python、numpy
         腐蚀 = 取邻域最小值（前景收缩），膨胀 = 取邻域最大值（前景扩张）；
         开运算 = 先腐蚀后膨胀（去小噪点），闭运算 = 先膨胀后腐蚀（填小孔洞）。
"""

import argparse
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30

SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


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
    """@brief 主流程：六种形态学操作的对比。"""
    parser = argparse.ArgumentParser(
        description="形态学操作演示：erode / dilate / open / close / gradient / tophat")
    parser.add_argument("-i", "--input", default=None,
                        help="二值图，缺省读取 assets/binary.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/morphology.png")
    parser.add_argument("-k", "--kernel", type=int, default=5,
                        help="结构元素尺寸（奇数）")
    parser.add_argument("-s", "--shape", default="ellipse", choices=list(SHAPES),
                        help="结构元素形状")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("binary.png")
    src = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 素材是白底黑内容，取反让文字/形状成为前景。
    _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

    # 步骤 1：定义结构元素。
    kernel = args.kernel
    if kernel < 3 or kernel % 2 == 0:
        print("kernel 必须是不小于 3 的奇数，已改用 5")
        kernel = 5
    element = cv2.getStructuringElement(SHAPES[args.shape], (kernel, kernel))

    # 步骤 2~4：腐蚀、膨胀、开闭运算与梯度。
    eroded = cv2.erode(binary, element)
    dilated = cv2.dilate(binary, element)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, element)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, element)

    print("输入: {}, kernel={} ({})".format(input_path, kernel, args.shape))
    print("前景像素数:")
    for name, image in [("input", binary), ("erode", eroded), ("dilate", dilated),
                        ("open", opened), ("close", closed), ("gradient", gradient),
                        ("tophat", tophat)]:
        print("  {:<12}{}".format(name, int(np.count_nonzero(image))))

    panels = [
        fit_panel(binary, "binary input"),
        fit_panel(eroded, "erode"),
        fit_panel(dilated, "dilate"),
        fit_panel(opened, "open (remove specks)"),
        fit_panel(closed, "close (fill holes)"),
        fit_panel(gradient, "morph gradient"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "morphology.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("morphology", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
