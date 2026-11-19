# -*- coding: utf-8 -*-
"""
@file    linear_transform.py
@brief   线性变换演示：用增益 alpha 与偏置 beta 调整对比度和亮度。
@details 流程：读取灰度图 -> 设定增益和偏置 -> 逐像素变换 -> 裁剪有效范围
         -> 拼接对比图，可选滑动条实时调节。
@note    依赖：opencv-python、numpy
         cv2.convertScaleAbs 的语义是 |alpha * x + beta| 再饱和到 0~255，
         中间的绝对值会让负增益出现反直觉结果（alpha=-1, beta=0 时图像不变），
         本演示专门把这一点和真正的截断版本放在一起对比。
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


def linear_abs(gray, alpha, beta):
    """
    @brief  标准线性变换：|alpha * x + beta| 后饱和到 0~255。
    @param  gray  输入灰度图。
    @param  alpha 增益（对比度）。
    @param  beta  偏置（亮度）。
    @return 变换结果。
    """
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)


def linear_clamped(gray, alpha, beta):
    """
    @brief  严格线性变换：alpha * x + beta 直接截断到 0~255（不做绝对值）。
    @param  gray  输入灰度图。
    @param  alpha 增益。
    @param  beta  偏置。
    @return 变换结果。
    @note   这才是数学意义上的线性映射；convertScaleAbs 因为取绝对值，负增益会失真。
    """
    values = np.clip(alpha * np.arange(256, dtype=np.float64) + beta, 0, 255)
    lut = values.astype(np.uint8).reshape(256, 1)
    return cv2.LUT(gray, lut)


def make_panel(image, title):
    """
    @brief  给单通道图加标题条并转成 BGR。
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


def print_stats(name, img):
    """@brief 打印均值与标准差，量化对比度变化。"""
    print("  {:<26} mean={:>6.1f} std={:>6.1f}".format(
        name, float(img.mean()), float(img.std())))


def interactive_tune(gray):
    """
    @brief  弹窗 + 滑动条实时调节增益与偏置。
    @param  gray 输入灰度图。
    @note   仅在 --display 时进入；ESC 或 q 退出。
    """
    window = "linear transform (ESC/q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("alpha x100", window, 150, 400, lambda _v: None)
    cv2.createTrackbar("beta +100", window, 100, 200, lambda _v: None)
    while True:
        alpha = cv2.getTrackbarPos("alpha x100", window) / 100.0
        beta = cv2.getTrackbarPos("beta +100", window) - 100.0
        cv2.imshow(window, linear_abs(gray, alpha, beta))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def main():
    """@brief 主流程：多组 alpha/beta 组合对比，并演示绝对值语义的坑。"""
    parser = argparse.ArgumentParser(
        description="线性变换演示：增益/偏置调整对比度与亮度")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/low_contrast.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="参考图像路径，缺省读取 assets/portrait.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/linear_transform.png")
    parser.add_argument("-a", "--alpha", type=float, default=1.6, help="主参数：增益（对比度）")
    parser.add_argument("-b", "--beta", type=float, default=0.0, help="主参数：偏置（亮度）")
    parser.add_argument("-d", "--display", action="store_true",
                        help="弹窗并用滑动条实时调节")
    args = parser.parse_args()

    input_path = args.input or locate_asset("low_contrast.png")

    # 步骤 1：读取灰度图。
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    alpha, beta = args.alpha, args.beta

    # 步骤 3：逐像素变换（convertScaleAbs 内部就是 y = |alpha * x + beta| 的饱和版本）。
    main_result = linear_abs(gray, alpha, beta)

    # 步骤 4：裁剪有效范围。对比 convertScaleAbs 与严格截断在负增益下的差异。
    abs_neg = linear_abs(gray, -1.0, 0.0)       # 取绝对值：与原图相同
    clamp_neg = linear_clamped(gray, -1.0, 0.0)  # 严格截断：全部为 0

    print("输入: {}, alpha = {}, beta = {}".format(input_path, alpha, beta))
    print_stats("input", gray)
    print_stats("alpha 0.5", linear_abs(gray, 0.5, 0.0))
    print_stats("alpha {:.2f}".format(alpha), main_result)
    print_stats("alpha 2.0", linear_abs(gray, 2.0, 0.0))
    print_stats("beta -60", linear_abs(gray, 1.0, -60.0))
    print_stats("beta +60", linear_abs(gray, 1.0, 60.0))
    print_stats("convertScaleAbs a=-1", abs_neg)
    print_stats("clamped a=-1", clamp_neg)

    # 参考图：同一场景的正常曝光版本，用来目测对比度是否已拉回正常水平。
    reference_path = args.reference or locate_asset("portrait.png")
    ref_color = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    reference = (cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
                 if ref_color is not None else gray.copy())

    panels = [
        make_panel(gray, "low contrast input"),
        make_panel(reference, "reference (normal)"),
        make_panel(main_result, "alpha {:.2f} beta {:d}".format(alpha, int(beta))),
        make_panel(linear_abs(gray, 0.5, 0.0), "alpha 0.5"),
        make_panel(linear_abs(gray, 2.0, 0.0), "alpha 2.0"),
        make_panel(linear_abs(gray, 1.0, -60.0), "beta -60"),
        make_panel(linear_abs(gray, 1.0, 60.0), "beta +60"),
        make_panel(abs_neg, "convertScaleAbs a=-1 (abs)"),
        make_panel(clamp_neg, "clamped a=-1 (black)"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "linear_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    # 步骤 5：实时调节（可选）。
    if args.display:
        interactive_tune(gray)


if __name__ == "__main__":
    main()
