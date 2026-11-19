# -*- coding: utf-8 -*-
"""
@file    affine_transform.py
@brief   仿射变换演示：旋转、缩放、平移、剪切与插值方式对比。
@details 流程：定义对应点 -> 计算仿射矩阵 -> 应用变换 -> 插值方式对比 -> 输出拼接图。
@note    依赖：opencv-python、numpy
         仿射变换只有 6 个自由度，保持平行线平行；要产生消失点必须用 8 自由度的单应矩阵。
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
    scale = min(PANEL_WIDTH / image.shape[1], PANEL_HEIGHT / image.shape[0])
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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


def print_matrix(name, matrix):
    """@brief 打印 2x3 仿射矩阵。"""
    print("{}:".format(name))
    for row in matrix:
        print("  [" + "".join("{:11.4f}".format(v) for v in row) + " ]")


def main():
    """@brief 主流程：五种变换与两种插值方式的对比。"""
    parser = argparse.ArgumentParser(
        description="仿射变换演示：旋转/缩放/平移/剪切与插值对比")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/scene.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/affine_transform.png")
    parser.add_argument("-a", "--angle", type=float, default=25, help="旋转角度(度)")
    parser.add_argument("-s", "--scale", type=float, default=0.85,
                        help="旋转时同时施加的缩放")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("scene.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    height, width = src.shape[:2]
    center = (width / 2.0, height / 2.0)
    size = (width, height)
    border = (255, 255, 255)

    # 步骤 1：定义对应点。三点不共线即可唯一确定一个仿射矩阵。
    src_tri = np.float32([[100, 100], [924, 120], [120, 924]])
    dst_tri = np.float32([[180, 160], [880, 60], [120, 940]])

    # 步骤 2：计算仿射矩阵。
    rotation = cv2.getRotationMatrix2D(center, args.angle, args.scale)
    scaling = cv2.getRotationMatrix2D(center, 0.0, 1.5)
    translation = np.float32([[1, 0, 80], [0, 1, -60]])
    shear = cv2.getAffineTransform(src_tri, dst_tri)
    print_matrix("rotation", rotation)
    print_matrix("shear (from 3 point pairs)", shear)

    # 步骤 3：应用变换。
    rotate_kwargs = dict(dsize=size, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    rotated = cv2.warpAffine(src, rotation, **rotate_kwargs)
    scaled = cv2.warpAffine(src, scaling, **rotate_kwargs)
    translated = cv2.warpAffine(src, translation, **rotate_kwargs)
    sheared = cv2.warpAffine(src, shear, **rotate_kwargs)

    # 步骤 4：插值处理。缩小后放大才能看出最近邻的锯齿与双线性的平滑。
    small = cv2.resize(src, None, fx=0.12, fy=0.12, interpolation=cv2.INTER_NEAREST)
    nearest = cv2.resize(small, size, interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(small, size, interpolation=cv2.INTER_LINEAR)

    # 链式变换：先旋转再平移，等价于矩阵相乘（注意顺序）。
    combined = cv2.warpAffine(rotated, translation, **rotate_kwargs)

    panels = [
        fit_panel(src, "original"),
        fit_panel(rotated, "rotate {:.0f}deg".format(args.angle)),
        fit_panel(scaled, "scale 1.5x"),
        fit_panel(translated, "translate (+80, -60)"),
        fit_panel(sheared, "shear (3 point pairs)"),
        fit_panel(combined, "rotate then translate"),
        fit_panel(nearest, "INTER_NEAREST (12% then back)"),
        fit_panel(linear, "INTER_LINEAR (12% then back)"),
    ]
    result = montage(panels, 4)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "affine_transform.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("输入: {} ({}x{})".format(input_path, width, height))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("affine transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
