# -*- coding: utf-8 -*-
"""
@file    image_filtering.py
@brief   图像滤波演示：高斯、中值、双边滤波与锐化的效果对比。
@details 流程：选择滤波器类型 -> 设定核大小 -> 卷积运算 -> 对比效果。
@note    依赖：opencv-python、numpy
         高斯/均值按空间距离加权，边缘一起被模糊；中值对椒盐噪声最有效；
         双边滤波同时考虑空间距离与灰度差，因此能「保边去噪」。
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


def sharpness(gray):
    """
    @brief  用拉普拉斯方差衡量图像的高频（细节/噪声）强度。
    @param  gray 灰度图。
    @return 拉普拉斯响应的方差，越大越锐利。
    """
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def main():
    """@brief 主流程：四种滤波 + 锐化的对比。"""
    parser = argparse.ArgumentParser(
        description="图像滤波演示：高斯 / 中值 / 双边 / 锐化")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/texture.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_filtering.png")
    parser.add_argument("-k", "--kernel", type=int, default=7,
                        help="高斯/中值滤波的核大小（奇数）")
    parser.add_argument("-s", "--sigma", type=float, default=0,
                        help="高斯标准差，0 表示由核大小推导")
    parser.add_argument("-e", "--sigmaColor", type=float, default=60,
                        help="双边滤波的颜色标准差")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("texture.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    kernel = args.kernel
    if kernel < 3 or kernel % 2 == 0:
        print("kernel 必须是不小于 3 的奇数，已改用 7")
        kernel = 7

    # 步骤 1~2：按核大小与标准差构造各滤波器。
    # 为了体现降噪效果，先人为加入高斯噪声作为输入。
    rng = np.random.default_rng(20261119)
    noise = rng.normal(0, 18, gray.shape)
    noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 步骤 3：卷积运算。
    gaussian = cv2.GaussianBlur(noisy, (kernel, kernel), args.sigma)
    median = cv2.medianBlur(noisy, kernel)
    bilateral = cv2.bilateralFilter(noisy, kernel, args.sigmaColor, kernel / 2.0)
    # 非锐化掩模：原图减去模糊图得到高频，再按比例加回。
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0.0)

    # 步骤 4：对比效果。锐度指标用拉普拉斯方差。
    print("输入: {}, kernel={}".format(input_path, kernel))
    print("{:<12}{}".format("image", "sharpness"))
    for name, image in [("original", gray), ("noisy", noisy), ("gaussian", gaussian),
                        ("median", median), ("bilateral", bilateral),
                        ("sharpened", sharpened)]:
        print("{:<12}{:.1f}".format(name, sharpness(image)))

    panels = [
        fit_panel(gray, "original"),
        fit_panel(noisy, "noisy (sigma 18)"),
        fit_panel(gaussian, "gaussian"),
        fit_panel(median, "median"),
        fit_panel(bilateral, "bilateral"),
        fit_panel(sharpened, "unsharp mask"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_filtering.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image filtering", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
