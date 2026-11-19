# -*- coding: utf-8 -*-
"""
@file    image_pyramid.py
@brief   图像金字塔演示：构建高斯金字塔、拉普拉斯金字塔并用拉普拉斯金字塔重建原图。
@details 流程：高斯模糊降采样 -> 上采样重建 -> 拉普拉斯计算 -> 由拉普拉斯金字塔重建。
@note    依赖：opencv-python、numpy
         pyrDown = 高斯模糊 + 隔行隔列采样，pyrUp 只是插值放大（不可逆），
         所以拉普拉斯层保存的正是「上采样丢失的高频」；把它们逐层加回去即可无损重建。
"""

import argparse
import math
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


def laplacian_view(laplacian):
    """
    @brief  把带正负值的拉普拉斯层可视化（零点映射到中灰）。
    @param  laplacian 拉普拉斯层（float32）。
    @return uint8 灰度可视化图。
    """
    normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def psnr(image, reference):
    """@brief 计算两幅 8 位灰度图的 PSNR(dB)。"""
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    """@brief 主流程：构建两层金字塔并做无损重建。"""
    parser = argparse.ArgumentParser(
        description="图像金字塔演示：高斯金字塔 + 拉普拉斯金字塔重建")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/skyline.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_pyramid.png")
    parser.add_argument("-l", "--levels", type=int, default=3,
                        help="金字塔层数（不含第 0 层）")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("skyline.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    levels = max(1, min(4, args.levels))

    # 步骤 1：高斯模糊降采样。每层尺寸减半，细节逐层减少。
    gaussian = [gray]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    # 步骤 2~3：上采样重建并求拉普拉斯层（上采样丢失的高频）。
    laplacian = []
    for i in range(levels):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        laplacian.append(gaussian[i].astype(np.float32) - up.astype(np.float32))

    # 步骤 4：由拉普拉斯金字塔逐层重建。
    reconstructed = gaussian[-1].astype(np.float32)
    for i in range(levels - 1, -1, -1):
        up = cv2.pyrUp(reconstructed.astype(np.uint8),
                       dstsize=(gaussian[i].shape[1], gaussian[i].shape[0])).astype(np.float32)
        reconstructed = up + laplacian[i]
    rebuilt = np.clip(reconstructed, 0, 255).astype(np.uint8)

    print("输入: {}, 层数: {}".format(input_path, levels + 1))
    for index, level in enumerate(gaussian):
        print("  level {}: {}x{}".format(index, level.shape[1], level.shape[0]))
    print("  重建 PSNR: {:.2f} dB (拉普拉斯金字塔可无损重建)".format(psnr(rebuilt, gray)))

    error = cv2.convertScaleAbs(cv2.absdiff(rebuilt, gray), alpha=8.0)  # 放大 8 倍才能看清残差

    panels = [
        fit_panel(gray, "level 0 (original)"),
        fit_panel(gaussian[1], "level 1 (pyrDown)"),
        fit_panel(gaussian[2], "level 2 (pyrDown)"),
        fit_panel(laplacian_view(laplacian[0]), "laplacian level 0"),
        fit_panel(rebuilt, "reconstructed"),
        fit_panel(error, "abs error x8"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_pyramid.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image pyramid", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
