# -*- coding: utf-8 -*-
"""
@file    fourier_transform.py
@brief   傅里叶变换演示：离散傅里叶变换、频谱中心化与频域低通/高通滤波。
@details 流程：灰度转换 -> 离散傅里叶变换 -> 频谱中心化与对数幅度谱 -> 频域滤波 -> 逆变换。
@note    依赖：opencv-python、numpy
         幅度谱动态范围极大，必须取 log(1+|F|) 才能看见细节；
         dft 在每个维度上按 2、3、5 的幂最快，所以先用 getOptimalDFTSize 补边。
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


def fft_shift(matrix):
    """
    @brief  把零频分量搬到画布中心（四个象限对角互换）。
    @param  matrix 单通道浮点矩阵。
    @return 中心化后的矩阵。
    @note   dft 的零频在左上角，中心化只是为了人眼看频谱方便，不影响滤波结果。
    """
    cy, cx = matrix.shape[0] // 2, matrix.shape[1] // 2
    shifted = matrix.copy()
    shifted[:cy, :cx] = matrix[cy:, cx:]
    shifted[cy:, cx:] = matrix[:cy, :cx]
    shifted[:cy, cx:] = matrix[cy:, :cx]
    shifted[cy:, :cx] = matrix[:cy, cx:]
    return shifted


def filter_spectrum(spectrum, mask, size):
    """
    @brief  用给定掩膜过滤复数频谱并做逆变换。
    @param  spectrum 双通道复数频谱（零频在左上角）。
    @param  mask     单通道浮点掩膜，尺寸与频谱一致。
    @param  size     原始图像尺寸 (width, height)，用于裁掉补边。
    @return 滤波后的灰度图（uint8）。
    """
    planes = cv2.split(spectrum)
    filtered = cv2.merge([planes[0] * mask, planes[1] * mask])
    inverse = cv2.idft(filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return np.clip(inverse[:size[1], :size[0]], 0, 255).astype(np.uint8)


def main():
    """@brief 主流程：计算频谱、构造低通/高通掩膜并做逆变换。"""
    parser = argparse.ArgumentParser(
        description="傅里叶变换演示：DFT / 频谱中心化 / 低通与高通滤波")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/texture.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/fourier_transform.png")
    parser.add_argument("-r", "--radius", type=int, default=60,
                        help="低通掩膜的截止半径(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("texture.png")

    # 步骤 1：灰度转换。
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 2：离散傅里叶变换。先补边到最优尺寸，再用双通道浮点矩阵存实部与虚部。
    rows = cv2.getOptimalDFTSize(gray.shape[0])
    cols = cv2.getOptimalDFTSize(gray.shape[1])
    padded = cv2.copyMakeBorder(gray, 0, rows - gray.shape[0], 0, cols - gray.shape[1],
                                cv2.BORDER_CONSTANT, value=0)
    spectrum = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 步骤 3：频谱中心化。幅度取对数后才能看到高频细节。
    planes = cv2.split(spectrum)
    magnitude = cv2.magnitude(planes[0], planes[1])
    spectrum_view = fft_shift(np.log1p(magnitude))
    spectrum_view = cv2.normalize(spectrum_view, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 步骤 4：频域滤波。掩膜在中心化坐标系下画圆，再反中心化以对齐零频在左上角的频谱。
    radius = max(1, args.radius)
    low_mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(low_mask, (cols // 2, rows // 2), radius, 1.0, cv2.FILLED)
    low_mask = fft_shift(low_mask)
    high_mask = 1.0 - low_mask

    low_pass = filter_spectrum(spectrum, low_mask, (gray.shape[1], gray.shape[0]))
    high_pass = filter_spectrum(spectrum, high_mask, (gray.shape[1], gray.shape[0]))

    low_energy = float(low_mask.sum()) / low_mask.size
    print("输入: {} ({}x{}) -> 补边后 {}x{}".format(
        input_path, gray.shape[1], gray.shape[0], cols, rows))
    print("低通截止半径: {} 像素, 掩膜能量占比: {:.2f}%".format(radius, low_energy * 100.0))

    panels = [
        fit_panel(gray, "gray input"),
        fit_panel(spectrum_view, "log magnitude spectrum"),
        fit_panel(low_pass, "low pass (blur)"),
        fit_panel(high_pass, "high pass (edges)"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "fourier_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("fourier transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
