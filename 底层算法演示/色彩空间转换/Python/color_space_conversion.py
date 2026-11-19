# -*- coding: utf-8 -*-
"""
@file    color_space_conversion.py
@brief   色彩空间转换演示：在 BGR / HSV / Lab / YCrCb 之间转换并分析各通道特性。
@details 流程：读取彩色图 -> 转换目标空间 -> 分离通道并可视化 -> 统计各通道特性
         -> 用 HSV 色相区间做颜色分割，说明换空间的实用价值。
@note    依赖：opencv-python、numpy
         HSV 的 H 取值 0~179（8 位图角度折半），Lab 的 a/b 以 128 为零点，
         直接当灰度显示会很暗或很平，展示时需归一化。
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


def channel_panel(channel, title):
    """
    @brief  把单通道图归一化到 0~255 并加标题条，转成三通道便于拼接。
    @param  channel 单通道图像。
    @param  title   标题文字（仅支持 ASCII）。
    @return 带标题条的三通道面板。
    @note   Lab 的 a/b 通道以 128 为零点、动态范围窄，不归一化几乎看不见差异。
    """
    # 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    small = cv2.resize(channel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
    panel = np.full((small.shape[0] + TITLE_HEIGHT, small.shape[1], 3), 255, dtype=np.uint8)
    panel[TITLE_HEIGHT:] = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def color_panel(image, title):
    """
    @brief  给彩色图加标题条（不做归一化）。
    @param  image 三通道图像。
    @param  title 标题文字。
    @return 带标题条的面板。
    """
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    panel = np.full((small.shape[0] + TITLE_HEIGHT, small.shape[1], 3), 255, dtype=np.uint8)
    panel[TITLE_HEIGHT:] = small
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


def print_channel_stats(space_name, channels, names):
    """
    @brief  打印某个空间的各通道统计量。
    @param  space_name 空间名称。
    @param  channels   通道列表。
    @param  names      通道名称列表。
    """
    print("{}:".format(space_name))
    for name, channel in zip(names, channels):
        print("  {:<4} mean={:>6.1f} std={:>6.1f}".format(
            name, float(channel.mean()), float(channel.std())))


def main():
    """@brief 主流程：转换三种色彩空间、分离通道、统计特性并做 HSV 颜色分割。"""
    parser = argparse.ArgumentParser(
        description="色彩空间转换演示：BGR / HSV / Lab / YCrCb 通道分析与 HSV 分割")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/fruits.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/color_space_conversion.png")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("fruits.png")

    # 步骤 1：读取图像。
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 2：转换到目标色彩空间。
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    # 步骤 3：分离通道。
    hsv_channels = cv2.split(hsv)
    lab_channels = cv2.split(lab)
    ycrcb_channels = cv2.split(ycrcb)

    # 步骤 4：分析特性（均值反映整体色调，标准差反映通道内的分布范围）。
    print_channel_stats("HSV (H:0~179, S/V:0~255)", hsv_channels, ["H", "S", "V"])
    print_channel_stats("Lab (L:0~255, a/b:以128为零点)", lab_channels, ["L", "a", "b"])
    print_channel_stats("YCrCb", ycrcb_channels, ["Y", "Cr", "Cb"])

    # 步骤 5：用 HSV 的色相区间做颜色分割——这是换色彩空间最实用的场景。
    #        柠檬的黄色在 OpenCV 的 H 轴上大约落在 20~35。
    mask = cv2.inRange(np.asarray(hsv, dtype=np.uint8),
                       np.array([20, 80, 80], dtype=np.uint8),
                       np.array([35, 255, 255], dtype=np.uint8))
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    ratio = 100.0 * cv2.countNonZero(mask) / mask.size
    print("黄色像素占比: {:.2f}%".format(ratio))

    # 拼接结果。
    panels = [
        color_panel(bgr, "BGR original"),
        channel_panel(hsv_channels[0], "HSV H (hue)"),
        channel_panel(hsv_channels[1], "HSV S (saturation)"),
        channel_panel(hsv_channels[2], "HSV V (value)"),
        channel_panel(lab_channels[0], "Lab L (lightness)"),
        channel_panel(lab_channels[1], "Lab a (green-red)"),
        channel_panel(lab_channels[2], "Lab b (blue-yellow)"),
        channel_panel(ycrcb_channels[0], "YCrCb Y (luma)"),
        channel_panel(mask, "H in [20,35] mask"),
        color_panel(masked, "yellow only"),
    ]
    result = montage(panels, 5)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "color_space_conversion.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("color space conversion", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
