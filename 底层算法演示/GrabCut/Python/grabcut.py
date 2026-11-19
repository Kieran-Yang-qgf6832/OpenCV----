# -*- coding: utf-8 -*-
"""
@file    grabcut.py
@brief   GrabCut 演示：用矩形框初始化，迭代优化 GMM 模型并提取前景。
@details 流程：定义初始矩形 -> 迭代优化模型 -> 涂鸦修正（可选）-> 提取前景。
@note    依赖：opencv-python、numpy
         GrabCut 在 RGB 空间建 GMM，先验只有「矩形内=可能前景、外=确定背景」，
         所以矩形必须把目标完整框住且尽量贴紧，否则背景颜色会混进前景模型。
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


def main():
    """@brief 主流程：矩形初始化的 GrabCut 前景提取。"""
    parser = argparse.ArgumentParser(
        description="GrabCut 演示：矩形初始化 + GMM 迭代")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/subject.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/grabcut.png")
    parser.add_argument("-m", "--margin", type=float, default=0.10,
                        help="矩形相对边界的留白比例")
    parser.add_argument("-n", "--iters", type=int, default=5, help="迭代次数")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("subject.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))

    # 步骤 1：定义初始矩形。留一圈边距，保证背景像素也能进入背景 GMM。
    margin = max(0.0, min(0.4, args.margin))
    height, width = src.shape[:2]
    rect = (int(width * margin), int(height * margin),
            int(width * (1 - 2 * margin)), int(height * (1 - 2 * margin)))

    # 步骤 2：迭代优化模型。
    mask = np.zeros(src.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(src, mask, rect, bgd_model, fgd_model, max(1, args.iters), cv2.GC_INIT_WITH_RECT)

    # 步骤 3~4：按标签提取前景。PR_FGD 表示「可能前景」，与 FGD 一并保留。
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    foreground = cv2.bitwise_and(src, src, mask=binary)
    foreground[binary == 0] = 255
    background = cv2.bitwise_and(src, src, mask=cv2.bitwise_not(binary))

    # 标签可视化：确定前景绿、可能前景浅绿、确定背景黑、可能背景灰。
    label_view = np.zeros_like(src)
    label_view[mask == cv2.GC_FGD] = (0, 60, 0)
    label_view[mask == cv2.GC_PR_FGD] = (0, 160, 0)
    label_view[mask == cv2.GC_PR_BGD] = (60, 60, 60)

    ratio = 100.0 * cv2.countNonZero(binary) / binary.size
    print("输入: {}, 迭代 {} 次".format(input_path, max(1, args.iters)))
    print("  前景占比: {:.2f}%".format(ratio))

    rect_view = src.copy()
    cv2.rectangle(rect_view, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                  (0, 255, 255), 3, cv2.LINE_AA)

    panels = [
        fit_panel(rect_view, "initial rect"),
        fit_panel(label_view, "grabCut labels"),
        fit_panel(foreground, "foreground"),
        fit_panel(background, "background"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "grabcut.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("grabcut", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
