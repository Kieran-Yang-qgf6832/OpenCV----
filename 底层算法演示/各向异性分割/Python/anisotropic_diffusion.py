# -*- coding: utf-8 -*-
"""
@file    anisotropic_diffusion.py
@brief   各向异性扩散演示：Perona-Malik 扩散在平滑噪声的同时保留边缘。
@details 流程：设置扩散参数 -> 迭代扩散 -> 与高斯滤波对比 -> 用 PSNR 与边缘强度量化评估。
@note    依赖：opencv-python（含 contrib 的 ximgproc）、numpy
         各向异性扩散的扩散系数随梯度增大而减小，因此平坦区被强平滑、边缘处几乎不扩散；
         高斯滤波对全图一视同仁，必然把边缘一起糊掉。
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


def psnr(image, reference):
    """
    @brief  计算两幅图之间的峰值信噪比。
    @param  image     待评图像。
    @param  reference 参考图。
    @return PSNR( dB )。
    """
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def edge_gradient_p99(gray):
    """
    @brief  用梯度幅值的 99 分位数衡量边缘保留程度。
    @param  gray 灰度图。
    @return 梯度幅值的 99 分位数，越大说明物体边界越锐利。
    @note   不能用梯度均值：均值被残余噪声主导，滤波越弱反而越大。
         真边缘位于高分位，取 99 分位才能把「边界陡不陡」单独拎出来。
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.percentile(cv2.magnitude(gx, gy), 99))


def match_gaussian_sigma(src, target_noise):
    """
    @brief  自动匹配高斯滤波的 sigma，使其平坦区噪声与扩散结果相当。
    @param  src          含噪彩色图。
    @param  target_noise 目标平坦区噪声标准差。
    @return 最接近目标的高斯 sigma。
    @note   只有把两者的平滑力度拉到同一水平，比较边缘锐利度才有意义。
    """
    def noise_of(sigma):
        blurred = cv2.GaussianBlur(src, (0, 0), sigma)
        return flat_region_noise(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    best_sigma, best_diff = 1.0, float("inf")
    for sigma in np.arange(0.2, 3.0 + 1e-9, 0.1):
        diff = abs(noise_of(sigma) - target_noise)
        if diff < best_diff:
            best_diff, best_sigma = diff, sigma
    for sigma in np.arange(max(0.05, best_sigma - 0.1), best_sigma + 0.1 + 1e-9, 0.01):
        diff = abs(noise_of(sigma) - target_noise)
        if diff < best_diff:
            best_diff, best_sigma = diff, sigma
    return float(best_sigma)


def flat_region_noise(gray):
    """
    @brief  取左上角一块平坦背景，用其标准差衡量残余噪声。
    @param  gray 灰度图。
    @return 标准差，越小说明噪声被压得越干净。
    """
    size = min(200, gray.shape[0], gray.shape[1])
    return float(gray[:size, :size].std())


def main():
    """@brief 主流程：各向异性扩散与高斯滤波的定量对比。"""
    parser = argparse.ArgumentParser(
        description="各向异性扩散演示：边缘保持平滑 vs 高斯滤波")
    parser.add_argument("-i", "--input", default=None,
                        help="含噪图像，缺省读取 assets/noisy.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="干净参考图，缺省读取 assets/smooth.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/anisotropic_diffusion.png")
    parser.add_argument("-a", "--alpha", type=float, default=0.05,
                        help="扩散速率，越大平滑越快，超过 0.25 会不稳定")
    parser.add_argument("-k", "--K", type=float, default=0.10,
                        help="边缘敏感度(归一化尺度)，梯度超过 K 后几乎不再扩散")
    parser.add_argument("-n", "--niters", type=int, default=10, help="迭代次数")
    parser.add_argument("-s", "--sigma", type=float, default=0,
                        help="对照用高斯滤波的标准差，0 表示自动匹配噪声水平")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("noisy.png")
    ref_path = args.reference or locate_asset("smooth.png")

    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 1：设置扩散参数（alpha 决定速率，K 决定把多大的梯度当成边缘）。
    alpha, k = args.alpha, args.K
    iterations = max(1, args.niters)

    # 步骤 2：迭代扩散。该函数只接受 8UC3 彩色输入，因此直接处理彩色图。
    filtered = cv2.ximgproc.anisotropicDiffusion(src, alpha, k, iterations)

    # 步骤 3：对比高斯滤波。先把 sigma 调到与扩散结果相同的平坦区噪声，
    #        否则「平滑力度不同」会让边缘锐利度的比较失去意义。
    #        指标一律在灰度图上计算，与参考图保持同一口径。
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    sigma = args.sigma
    if sigma <= 0:
        sigma = match_gaussian_sigma(src, flat_region_noise(filtered_gray))
    gaussian = cv2.GaussianBlur(src, (0, 0), sigma)
    gaussian_gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

    # 步骤 4：量化评估。
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    has_ref = reference is not None and reference.shape == gray.shape

    print("输入: {}, alpha = {}, K = {}, niters = {}, 匹配 sigma = {:.2f}".format(
        input_path, alpha, k, iterations, sigma))
    print("{:<22}{:<16}{:<16}{}".format("method", "flat noise", "grad p99", "PSNR(dB)"))

    def report(name, image):
        line = "{:<22}{:<16.2f}{:<16.2f}".format(
            name, flat_region_noise(image), edge_gradient_p99(image))
        line += "{:.2f}".format(psnr(image, reference)) if has_ref else "-"
        print(line)

    report("noisy input", gray)
    report("anisotropic", filtered_gray)
    report("gaussian", gaussian_gray)
    if has_ref:
        report("clean reference", reference)
    else:
        print("参考图缺失或尺寸不一致，跳过 PSNR")

    panels = [
        fit_panel(gray, "noisy input"),
        fit_panel(reference if has_ref else gray, "clean reference"),
        fit_panel(filtered, "anisotropic diffusion"),
        fit_panel(gaussian, "gaussian sigma {:.2f}".format(sigma)),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "anisotropic_diffusion.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("anisotropic diffusion", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
