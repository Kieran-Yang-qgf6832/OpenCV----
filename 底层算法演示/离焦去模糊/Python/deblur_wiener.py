# -*- coding: utf-8 -*-
"""
@file    deblur_wiener.py
@brief   离焦去模糊演示：用维纳滤波在频域复原高斯离焦模糊。
@details 流程：读取离焦模糊 -> 估计点扩散函数 -> 维纳滤波复原 -> 用 PSNR 评估质量。
@note    依赖：opencv-python、numpy
         维纳滤波 F = G·H* / (|H|² + K)：K 是噪声与信号的功率比，取太小会放大噪声，
         取太大则退化为低通、几乎没有锐化效果，所以 K 必须和 PSF 一起调。
"""

import argparse
import math
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30
PSF_SIZE = 63


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


def gaussian_psf_spectrum(shape, sigma):
    """
    @brief  构造高斯点扩散函数的频域形式。
    @param  shape 图像尺寸 (height, width)。
    @param  sigma 高斯半径，即离焦程度。
    @return 频域传递函数（复数）。
    @note   先把空间域的核放在画布中心，再 ifftshift 把中心搬到原点，
         否则负频部分会落在错误的位置，复原结果完全错误。
    """
    kernel1d = cv2.getGaussianKernel(PSF_SIZE, sigma)
    psf = np.outer(kernel1d, kernel1d).astype(np.float32)
    padded = np.zeros(shape, np.float32)
    half = PSF_SIZE // 2
    cy, cx = shape[0] // 2, shape[1] // 2
    padded[cy - half:cy + half + 1, cx - half:cx + half + 1] = psf
    return np.fft.fft2(np.fft.ifftshift(padded))


def wiener_deblur(gray, sigma, k):
    """
    @brief  维纳滤波复原。
    @param  gray  模糊灰度图。
    @param  sigma 高斯 PSF 的标准差。
    @param  k     噪声与信号功率比。
    @return 复原结果（uint8）。
    """
    img = gray.astype(np.float32)
    spectrum = np.fft.fft2(img)
    h = gaussian_psf_spectrum(img.shape, sigma)
    restored = spectrum * np.conj(h) / (np.abs(h) ** 2 + k)
    return np.clip(np.real(np.fft.ifft2(restored)), 0, 255).astype(np.uint8)


def psnr(image, reference):
    """
    @brief  计算两幅 8 位灰度图的 PSNR(dB)。
    @param  image     待评估图像。
    @param  reference 参考图像。
    @return PSNR(dB)。
    """
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    """@brief 主流程：扫描 PSF 与噪声比，取最优复原结果。"""
    parser = argparse.ArgumentParser(
        description="离焦去模糊演示：维纳滤波复原高斯离焦模糊")
    parser.add_argument("-i", "--input", default=None,
                        help="模糊图像，缺省读取 assets/blurred.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="清晰参考图，缺省读取 assets/sharp.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/deblur_wiener.png")
    parser.add_argument("-s", "--sigma", type=float, default=0,
                        help="PSF 标准差，0 表示自动搜索")
    parser.add_argument("-k", "--K", type=float, default=0,
                        help="噪声功率比，0 表示自动搜索")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("blurred.png")
    ref_path = args.reference or locate_asset("sharp.png")

    blurred = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if blurred is None:
        raise FileNotFoundError("无法读取模糊图像: {}".format(os.path.abspath(input_path)))
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    has_ref = reference is not None and reference.shape == blurred.shape

    # 步骤 1~2：确定 PSF。有参考图时直接搜索「让 PSNR 最高」的 sigma 与 K。
    best_sigma, best_k, best_psnr = args.sigma, args.K, -1.0
    best_restored = None

    if best_sigma > 0 and best_k > 0:
        best_restored = wiener_deblur(blurred, best_sigma, best_k)
        best_psnr = psnr(best_restored, reference) if has_ref else 0.0
    else:
        print("参数扫描 (sigma, K) -> PSNR(dB):")
        for sigma in (1.5, 2.0, 2.5, 3.0, 4.0):
            line = "  sigma={:4.1f}".format(sigma)
            for k in (1e-4, 1e-2, 1e-1):
                restored = wiener_deblur(blurred, sigma, k)
                value = psnr(restored, reference) if has_ref else 0.0
                line += "  K={:g}:{:6.2f}".format(k, value)
                if value > best_psnr:
                    best_psnr, best_sigma, best_k = value, sigma, k
                    best_restored = restored
            print(line)

    # 步骤 3：用最优参数复原，并给出一组过激进的参数作为对照。
    if best_restored is None:
        best_restored = wiener_deblur(blurred, max(0.5, best_sigma), max(1e-6, best_k))
    over_sharp = wiener_deblur(blurred, best_sigma, 1e-5)

    # 步骤 4：评估质量。
    print()
    print("输入: {}".format(input_path))
    print("  最优参数: sigma={:.2f}, K={:g}".format(best_sigma, best_k))
    if has_ref:
        print("  模糊图 PSNR:   {:.2f} dB".format(psnr(blurred, reference)))
        print("  复原图 PSNR:   {:.2f} dB".format(psnr(best_restored, reference)))
        print("  过激 K=1e-5:   {:.2f} dB (噪声被放大)".format(psnr(over_sharp, reference)))
    else:
        print("  参考图缺失，跳过 PSNR 评估")

    panels = [
        fit_panel(blurred, "blurred input"),
        fit_panel(best_restored, "wiener restored"),
        fit_panel(over_sharp, "wiener K too small"),
        fit_panel(reference if has_ref else blurred, "sharp reference"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "deblur_wiener.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("deblur wiener", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
