# -*- coding: utf-8 -*-
"""
@file    motion_deblur.py
@brief   运动去模糊演示：用直线运动模糊核 + 维纳滤波复原运动模糊图像。
@details 流程：构造运动模糊核 -> 扫描角度与长度 -> 维纳滤波复原 -> 用 PSNR 选最优参数。
@note    依赖：opencv-python、numpy
         与离焦模糊（各向同性高斯核）不同，运动模糊的 PSF 是一条线段，
         方向和长度都未知，所以必须二维扫描；核的能量归一化为 1，否则复原图亮度会漂移。
"""

import argparse
import math
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30
KERNEL_SIZE = 63


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


def motion_kernel(length, angle_deg):
    """
    @brief  构造直线运动模糊核。
    @param  length    运动长度（像素）。
    @param  angle_deg 运动方向（度）。
    @return KERNEL_SIZE×KERNEL_SIZE 的浮点核，能量归一化为 1。
    """
    center = (KERNEL_SIZE // 2, KERNEL_SIZE // 2)
    rad = math.radians(angle_deg)
    offset = (math.cos(rad) * length / 2.0, math.sin(rad) * length / 2.0)
    p1 = (int(round(center[0] - offset[0])), int(round(center[1] - offset[1])))
    p2 = (int(round(center[0] + offset[0])), int(round(center[1] + offset[1])))
    kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE), np.float32)
    cv2.line(kernel, p1, p2, 1.0, 1, cv2.LINE_AA)
    total = float(kernel.sum())
    return kernel / total if total > 0 else kernel  # 归一化，保证亮度不漂移


def wiener(gray, kernel, k):
    """
    @brief  维纳滤波复原。
    @param  gray   模糊灰度图。
    @param  kernel 空间域 PSF（尺寸可小于图像）。
    @param  k      噪声功率比。
    @return 复原结果（uint8）。
    """
    img = gray.astype(np.float32)
    spectrum = np.fft.fft2(img)

    padded = np.zeros_like(img)
    half = kernel.shape[0] // 2
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    padded[cy - half:cy + half + 1, cx - half:cx + half + 1] = kernel
    h = np.fft.fft2(np.fft.ifftshift(padded))

    restored = spectrum * np.conj(h) / (np.abs(h) ** 2 + k)
    return np.clip(np.real(np.fft.ifft2(restored)), 0, 255).astype(np.uint8)


def psnr(image, reference):
    """@brief 计算两幅 8 位灰度图的 PSNR(dB)。"""
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    """@brief 主流程：扫描运动核的角度与长度并取最优复原。"""
    parser = argparse.ArgumentParser(
        description="运动去模糊演示：运动核扫描 + 维纳滤波")
    parser.add_argument("-i", "--input", default=None,
                        help="模糊图像，缺省读取 assets/motion_blurred.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="清晰参考图，缺省读取 assets/sharp.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/motion_deblur.png")
    parser.add_argument("-k", "--K", type=float, default=0.01, help="噪声功率比")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("motion_blurred.png")
    ref_path = args.reference or locate_asset("sharp.png")

    blurred = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if blurred is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    has_ref = reference is not None and reference.shape == blurred.shape
    k = max(1e-8, args.K)

    # 步骤 1~2：两阶段扫描运动核。先粗扫定位大致方向，再在附近细化，
    #          否则角度/长度的量化误差会让复原质量比不处理还差。
    best_psnr, best_angle, best_length, best_restored = -1.0, 0.0, 10.0, None

    def evaluate(angle, length):
        nonlocal best_psnr, best_angle, best_length, best_restored
        restored = wiener(blurred, motion_kernel(length, angle), k)
        value = psnr(restored, reference) if has_ref else 0.0
        if value > best_psnr:
            best_psnr, best_angle, best_length, best_restored = value, angle, length, restored

    print("粗扫 (角度步长 15, 长度 10/20/30):")
    for angle in range(0, 180, 15):
        for length in (10.0, 20.0, 30.0):
            evaluate(angle, length)
    print("  粗扫最优: 角度 {} 度, 长度 {} 像素, PSNR {:.2f} dB".format(
        best_angle, int(best_length), best_psnr))

    coarse_angle, coarse_length = best_angle, best_length
    for angle in np.arange(coarse_angle - 10, coarse_angle + 10 + 1e-9, 5):
        for length in np.arange(max(2.0, coarse_length - 6), coarse_length + 6 + 1e-9, 2):
            evaluate(float(angle), float(length))
    print("  细化后最优: 角度 {} 度, 长度 {} 像素, PSNR {:.2f} dB".format(
        best_angle, int(best_length), best_psnr))

    # 步骤 3~4：用最优核复原并评估。
    if best_restored is None:
        best_restored = wiener(blurred, motion_kernel(16.0, 0.0), k)
    print()
    print("输入: {}".format(input_path))
    print("  最优运动核: 角度 {} 度, 长度 {} 像素, K={:g}".format(best_angle, best_length, k))
    if has_ref:
        print("  模糊图 PSNR: {:.2f} dB".format(psnr(blurred, reference)))
        print("  复原图 PSNR: {:.2f} dB".format(psnr(best_restored, reference)))
    else:
        print("  参考图缺失，跳过 PSNR 评估")

    kernel_display = cv2.normalize(motion_kernel(best_length, best_angle), None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
    kernel_display = cv2.resize(kernel_display, (240, 240), interpolation=cv2.INTER_NEAREST)

    panels = [
        fit_panel(blurred, "motion blurred"),
        fit_panel(best_restored, "wiener restored"),
        fit_panel(reference if has_ref else blurred, "sharp reference"),
        fit_panel(kernel_display, "estimated kernel"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "motion_deblur.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("motion deblur", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
