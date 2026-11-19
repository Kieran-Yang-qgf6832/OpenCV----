# -*- coding: utf-8 -*-
"""
@file    image_transform.py
@brief   图像变换演示：用查找表实现对数变换、伽马变换与对比度拉伸。
@details 流程：读取灰度图 -> 构建 256 项查找表 -> 对数变换 / 伽马变换 / 对比度拉伸
         -> 拼接对比图并统计各变换后的灰度分布。
@note    依赖：opencv-python、numpy
         灰度变换是逐像素单调映射，用 LUT 只需算 256 次，比逐像素算 pow/log 快两个量级。
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


def build_log_lut():
    """
    @brief  构造对数变换查找表：s = c * ln(1 + r)，c 使 255 映射到 255。
    @return 形状 (256, 1) 的 uint8 查找表。
    @note   对数变换压缩高灰度、拉伸低灰度，适合整体偏暗、暗部细节丢失的图像。
    """
    i = np.arange(256, dtype=np.float64)
    c = 255.0 / np.log(1.0 + 255.0)
    return np.clip(c * np.log(1.0 + i), 0, 255).astype(np.uint8).reshape(256, 1)


def build_gamma_lut(gamma):
    """
    @brief  构造伽马变换查找表：s = 255 * (r / 255) ^ gamma。
    @param  gamma 伽马值；< 1 提亮暗部，> 1 压暗。
    @return 形状 (256, 1) 的 uint8 查找表。
    """
    i = np.arange(256, dtype=np.float64)
    return np.clip(255.0 * np.power(i / 255.0, gamma), 0, 255).astype(np.uint8).reshape(256, 1)


def build_stretch_lut(lo, hi):
    """
    @brief  构造对比度拉伸查找表：把 [lo, hi] 线性映射到 [0, 255]。
    @param  lo 输入图的实际最小灰度。
    @param  hi 输入图的实际最大灰度。
    @return 形状 (256, 1) 的 uint8 查找表。
    """
    i = np.arange(256, dtype=np.float64)
    span = (hi - lo) if hi > lo else 1.0
    return np.clip((i - lo) * 255.0 / span, 0, 255).astype(np.uint8).reshape(256, 1)


def make_panel(gray, title):
    """
    @brief  给单通道图加上标题条并转成 BGR，便于拼接展示。
    @param  gray  单通道图像。
    @param  title 标题文字（仅支持 ASCII）。
    @return 带标题条的三通道面板。
    """
    # 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    panel = np.full((small.shape[0] + TITLE_HEIGHT, small.shape[1], 3), 255, dtype=np.uint8)
    panel[TITLE_HEIGHT:, :] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def montage(panels, cols):
    """
    @brief  按列数把若干等尺寸面板拼成一张大图。
    @param  panels 面板列表。
    @param  cols   每行面板数。
    @return 拼接结果。
    """
    rows = (len(panels) + cols - 1) // cols
    h, w = panels[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, cols)
        canvas[r * h:r * h + panel.shape[0], c * w:c * w + panel.shape[1]] = panel[:h, :w]
    return canvas


def print_stats(name, img):
    """@brief 打印图像灰度统计，用于对比不同变换的效果。"""
    print("  {:<18} min={:>4d} max={:>4d} mean={:.1f}".format(
        name, int(img.min()), int(img.max()), float(img.mean())))


def main():
    """@brief 主流程：读取灰度图、构建查找表、逐项变换并拼接输出。"""
    parser = argparse.ArgumentParser(
        description="图像变换演示：LUT 实现对数 / 伽马 / 对比度拉伸")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/portrait.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_transform.png")
    parser.add_argument("-g", "--gamma", type=float, default=0.4, help="伽马变换的主参数")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("portrait.png")

    # 步骤 1：读取并转为灰度图。
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print("输入: {} ({}x{})".format(input_path, gray.shape[1], gray.shape[0]))
    print_stats("original", gray)

    # 步骤 2：构建查找表。三种变换都是 256 项单调映射，先算表再一次 LUT。
    # 拉伸区间取 1%~99% 百分位，避免极少数过曝/欠曝像素主导动态范围。
    lo, hi = (float(v) for v in np.percentile(gray, (1, 99)))
    print("拉伸区间: [{:.0f}, {:.0f}]".format(lo, hi))
    main_gamma = args.gamma
    high_gamma = 1.0 / main_gamma

    # 步骤 3：逐项变换。
    log_result = cv2.LUT(gray, build_log_lut())
    gamma_low = cv2.LUT(gray, build_gamma_lut(main_gamma))
    gamma_one = cv2.LUT(gray, build_gamma_lut(1.0))
    gamma_high = cv2.LUT(gray, build_gamma_lut(high_gamma))
    stretched = cv2.LUT(gray, build_stretch_lut(lo, hi))

    print("变换后统计:")
    print_stats("log", log_result)
    print_stats("gamma {:.2f}".format(main_gamma), gamma_low)
    print_stats("gamma 1.0", gamma_one)
    print_stats("gamma {:.2f}".format(high_gamma), gamma_high)
    print_stats("stretch", stretched)

    # 步骤 4：拼接对比图并保存。
    panels = [
        make_panel(gray, "original"),
        make_panel(log_result, "log"),
        make_panel(gamma_low, "gamma {:.2f}".format(main_gamma)),
        make_panel(gamma_one, "gamma 1.0 (identity)"),
        make_panel(gamma_high, "gamma {:.2f}".format(high_gamma)),
        make_panel(stretched, "stretch 1%-99%"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
