# -*- coding: utf-8 -*-
"""
@file    image_inpainting.py
@brief   图像修复演示：用 TELEA 与 Navier-Stokes 两种算法修复划痕与损坏区域。
@details 流程：读取损坏图与掩码 -> 选择修复算法 -> 执行修复 -> 与干净参考图比较质量。
@note    依赖：opencv-python、numpy
         掩码必须是单通道 8 位、非零像素表示待修复区域；两种算法都靠周边像素扩散填补，
         TELEA 用快速行进法，速度快；NS 用流体动力学模型，纹理连续性更好但更慢。
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
    """@brief 计算两幅 8 位图像的 PSNR(dB)。"""
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    """@brief 主流程：两种修复算法对比。"""
    parser = argparse.ArgumentParser(
        description="图像修复演示：TELEA / Navier-Stokes")
    parser.add_argument("-i", "--input", default=None,
                        help="损坏图像，缺省读取 assets/damaged.png")
    parser.add_argument("-m", "--mask", default=None,
                        help="掩码图像，缺省读取 assets/mask.png")
    parser.add_argument("-r", "--reference", default=None,
                        help="干净参考图，缺省读取 assets/photo.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_inpainting.png")
    parser.add_argument("-a", "--radius", type=int, default=3,
                        help="修复时参考的邻域半径")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("damaged.png")
    mask_path = args.mask or locate_asset("mask.png")
    ref_path = args.reference or locate_asset("photo.png")

    # 步骤 1：读取损坏图与掩码。掩码必须与图像同尺寸、单通道。
    damaged = cv2.imread(input_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if damaged is None or mask is None:
        raise FileNotFoundError("无法读取图像或掩码")
    if mask.shape != damaged.shape[:2]:
        mask = cv2.resize(mask, (damaged.shape[1], damaged.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    radius = max(1, args.radius)

    # 步骤 2~3：两种算法各修复一次。
    telea = cv2.inpaint(damaged, binary_mask, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(damaged, binary_mask, radius, cv2.INPAINT_NS)

    # 步骤 4：评估质量。只在掩码区域内统计误差才有意义。
    reference = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    ratio = 100.0 * cv2.countNonZero(binary_mask) / binary_mask.size
    print("输入: {}, 掩码像素占比: {:.2f}%".format(input_path, ratio))
    if reference is not None and reference.shape == damaged.shape:
        diff_telea = cv2.absdiff(telea, reference)
        diff_ns = cv2.absdiff(ns, reference)
        masked_telea = cv2.mean(diff_telea, mask=binary_mask)[0]
        masked_ns = cv2.mean(diff_ns, mask=binary_mask)[0]
        print("  全图 PSNR: TELEA {:.2f} dB, NS {:.2f} dB".format(
            psnr(telea, reference), psnr(ns, reference)))
        print("  掩码区平均绝对误差: TELEA {:.2f}, NS {:.2f}".format(masked_telea, masked_ns))
    else:
        reference = damaged
        print("  参考图缺失，跳过质量评估")

    print("  注: 掩码面积很小（约 1%），所以全图 PSNR 差异不大，看掩码区误差更有意义")

    panels = [
        fit_panel(damaged, "damaged"),
        fit_panel(binary_mask, "mask"),
        fit_panel(telea, "TELEA"),
        fit_panel(ns, "Navier-Stokes"),
        fit_panel(reference, "clean reference"),
        fit_panel(cv2.absdiff(telea, reference), "telea error x4"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_inpainting.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image inpainting", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
