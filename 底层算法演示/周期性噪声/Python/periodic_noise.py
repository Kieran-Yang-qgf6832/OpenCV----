# -*- coding: utf-8 -*-
"""
@file    periodic_noise.py
@brief   周期性噪声演示：在频域定位噪声峰值并用陷波滤波器去除。
@details 流程：傅里叶变换 -> 定位噪声峰值 -> 构造陷波滤波器 -> 逆变换复原。
@note    依赖：opencv-python、numpy
         周期性噪声在频域表现为成对出现的孤立亮点（关于零频对称），
         用陷波掩膜把它们和对称点一起挖掉即可，比空域滤波干净得多。
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
    """@brief 计算两幅 8 位灰度图的 PSNR(dB)。"""
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    """@brief 主流程：频谱峰值检测 + 陷波滤波。"""
    parser = argparse.ArgumentParser(
        description="周期性噪声演示：DFT 峰值检测 + 陷波滤波")
    parser.add_argument("-i", "--input", default=None,
                        help="含噪图像，缺省读取 assets/periodic_noise.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="干净参考图，缺省读取 assets/photo.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/periodic_noise.png")
    parser.add_argument("-p", "--peaks", type=int, default=4,
                        help="要挖掉的峰值对数量")
    parser.add_argument("-n", "--notchRadius", type=int, default=7,
                        help="陷波半径(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("periodic_noise.png")
    ref_path = args.reference or locate_asset("photo.png")

    src_color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src_color is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
    img = gray.astype(np.float32)

    # 步骤 1：傅里叶变换并求对数幅度谱。
    spectrum = np.fft.fft2(img)
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = np.fft.fftshift(magnitude)
    spectrum_view = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 步骤 2：定位噪声峰值。屏蔽中心低频区，再逐个做非极大值抑制。
    radius = max(2, args.notchRadius)
    peak_count = max(1, args.peaks)
    work = magnitude.copy()
    center = (work.shape[1] // 2, work.shape[0] // 2)
    cv2.circle(work, center, 40, 0, cv2.FILLED)  # 屏蔽低频主瓣，避免把图像自身结构当成噪声峰

    peaks = []
    for _ in range(peak_count):
        index = int(np.argmax(work))
        y, x = divmod(index, work.shape[1])
        peaks.append((x, y))
        # 峰值必然成对出现，抑制时要连同对称点一起挖掉，否则会重复计入同一对。
        cv2.circle(work, (x, y), radius * 3, 0, cv2.FILLED)
        cv2.circle(work, (2 * center[0] - x, 2 * center[1] - y), radius * 3, 0, cv2.FILLED)

    # 步骤 3：构造陷波滤波器。峰值成对出现，对称点要一起挖掉。
    notch = np.ones(img.shape, np.float32)
    print("输入: {}, 检出峰值 {} 对:".format(input_path, len(peaks)))
    for x, y in peaks:
        cv2.circle(notch, (x, y), radius, 0, cv2.FILLED)
        symmetric = (2 * center[0] - x, 2 * center[1] - y)
        cv2.circle(notch, symmetric, radius, 0, cv2.FILLED)
        print("  ({}, {}) 及其对称点 ({}, {})".format(x, y, symmetric[0], symmetric[1]))

    # 步骤 4：陷波后逆变换。掩膜在中心化坐标系下构造，需反中心化后才对齐原始频谱。
    notch = np.fft.ifftshift(notch)
    restored = np.fft.ifft2(spectrum * notch)
    result = np.clip(np.real(restored), 0, 255).astype(np.uint8)

    # 质量评估。
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if reference is not None and reference.shape == gray.shape:
        print("  PSNR: 含噪 {:.2f} dB -> 陷波后 {:.2f} dB".format(
            psnr(gray, reference), psnr(result, reference)))
    else:
        print("  参考图缺失，跳过 PSNR 评估")

    panels = [
        fit_panel(reference if reference is not None else gray, "clean reference"),
        fit_panel(gray, "periodic noise"),
        fit_panel(spectrum_view, "magnitude spectrum"),
        fit_panel(result, "notch filtered"),
    ]
    output = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "periodic_noise.png")

    if not cv2.imwrite(output_path, output):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("periodic noise", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
