# -*- coding: utf-8 -*-
"""
@file    image_thresholding.py
@brief   图像阈值化演示：全局阈值、Otsu 自动阈值与自适应阈值对比。
@details 流程：灰度转换 -> 固定全局阈值 -> Otsu 自动阈值 -> 自适应阈值(均值/高斯)
         -> 多阈值结果拼接对比，并统计各方法的前景像素占比。
@note    依赖：opencv-python、numpy
         素材纸面上有明显光照不均：全局阈值（含 Otsu）会在暗部整片丢失，
         自适应阈值按局部邻域取阈值，因此能保留暗部文字——这是本演示的重点。
"""

import argparse
import os

import cv2
import numpy as np

# 拼接图中标题条的高度（像素）。
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


def make_panel(image, title):
    """
    @brief  给单通道二值图加标题条并转成 BGR，便于拼接。
    @param  image 单通道图像。
    @param  title 标题文字（仅支持 ASCII）。
    @return 带标题条的面板。
    """
    # 展示时缩到一半，控制拼接图体积。
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    panel = np.full((small.shape[0] + TITLE_HEIGHT, small.shape[1], 3), 255, dtype=np.uint8)
    panel[TITLE_HEIGHT:] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
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


def print_foreground_ratio(name, mask):
    """
    @brief  打印二值图的前景像素占比。
    @param  name 方法名称。
    @param  mask 二值图。
    """
    ratio = 100.0 * cv2.countNonZero(mask) / mask.size
    print("  {:<28}{:.2f}%".format(name, ratio))


def main():
    """@brief 主流程：四种阈值策略对比并输出拼接图。"""
    parser = argparse.ArgumentParser(
        description="图像阈值化演示：全局 / Otsu / 自适应阈值")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/paper.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_thresholding.png")
    parser.add_argument("-b", "--block", type=int, default=31,
                        help="自适应阈值的邻域尺寸，必须是大于 1 的奇数")
    parser.add_argument("-c", "--constant", type=float, default=5,
                        help="自适应阈值的常数偏移 C")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("paper.png")

    # 步骤 1：灰度转换。
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    block_size = args.block
    if block_size < 3 or block_size % 2 == 0:
        print("block 必须是大于 1 的奇数，已改用 31")
        block_size = 31
    constant = args.constant

    # 步骤 2：Otsu 全局阈值。阈值由灰度直方图自动确定，适用于双峰分布。
    otsu_threshold, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 步骤 3：固定全局阈值，作为对照。
    _, fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 步骤 4：自适应阈值。以每个像素邻域的加权均值减 C 作为局部阈值。
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    adaptive_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    adaptive_fine = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, constant)

    print("输入: {}, Otsu 自动阈值 = {:.0f}".format(input_path, otsu_threshold))
    print("前景(白)像素占比:")
    print_foreground_ratio("fixed 127", fixed)
    print_foreground_ratio("otsu", otsu)
    print_foreground_ratio("adaptive mean (block={})".format(block_size), adaptive_mean)
    print_foreground_ratio("adaptive gauss (block={})".format(block_size), adaptive_gauss)
    print_foreground_ratio("adaptive gauss (block=11)", adaptive_fine)

    panels = [
        make_panel(gray, "gray input"),
        make_panel(fixed, "fixed 127"),
        make_panel(otsu, "otsu {:.0f}".format(otsu_threshold)),
        make_panel(adaptive_mean, "adaptive mean"),
        make_panel(adaptive_gauss, "adaptive gaussian"),
        make_panel(adaptive_fine, "adaptive gaussian b=11"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_thresholding.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image thresholding", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
