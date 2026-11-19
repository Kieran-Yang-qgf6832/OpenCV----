# -*- coding: utf-8 -*-
"""
@file    geometric_drawing.py
@brief   几何绘制演示：在画布上绘制基本几何形状、样式与文字标注。
@details 流程：创建画布 -> 绘制几何形状 -> 设置样式(线宽/线型/填充/亚像素) -> 标注文字 -> 保存/显示。
@note    依赖：opencv-python（pip install opencv-python）、numpy
         cv2.putText 只支持 ASCII（Hershey 矢量字体），中文需借助 freetype 模块。
"""

import argparse
import os

import cv2
import numpy as np

# 常用配色（BGR 顺序）。
RED = (60, 60, 220)
BLUE = (220, 120, 40)
GREEN = (60, 160, 60)
MAGENTA = (200, 60, 200)
ORANGE = (0, 150, 255)
GRAY = (120, 120, 120)


def locate_asset(file_name):
    """
    @brief  在若干候选位置中查找素材，便于从不同工作目录运行。
    @param  file_name 素材文件名，如 canvas.png。
    @return 第一个存在的路径；都不存在时返回首选候选。
    """
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
    """
    @brief  由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。
    @param  input_path 素材路径。
    @return 输出目录路径（尚未创建）。
    """
    directory = os.path.dirname(input_path)
    if os.path.basename(directory) == "assets":
        directory = os.path.dirname(directory)
    return os.path.join(directory, "output")


def load_or_create_canvas(path, width, height):
    """
    @brief  载入画布；素材缺失时创建纯白画布。
    @param  path   画布素材路径。
    @param  width  空白画布宽度。
    @param  height 空白画布高度。
    @return 三通道 BGR 画布。
    """
    if path:
        canvas = cv2.imread(path, cv2.IMREAD_COLOR)
        if canvas is not None:
            return canvas.copy()
        print("无法读取画布素材，改用空白画布: {}".format(path))
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_outlines(canvas):
    """
    @brief  绘制空心几何形状：不同线宽、矩形、圆、椭圆、折线与箭头。
    @param  canvas 目标画布。
    @note   线型用 cv2.LINE_AA 抗锯齿；thickness 为 -1 表示填充。
    """
    # 1) 同一位置的直线：线宽 1 / 3 / 7 对比。
    cv2.line(canvas, (60, 90), (460, 90), RED, 1, cv2.LINE_AA)
    cv2.line(canvas, (60, 150), (460, 150), RED, 3, cv2.LINE_AA)
    cv2.line(canvas, (60, 220), (460, 220), RED, 7, cv2.LINE_AA)
    cv2.putText(canvas, "thickness 1 / 3 / 7", (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1, cv2.LINE_AA)

    # 2) 空心矩形与圆。
    cv2.rectangle(canvas, (60, 300), (300, 450), BLUE, 2, cv2.LINE_AA)
    cv2.circle(canvas, (400, 375), 78, GREEN, 3, cv2.LINE_AA)

    # 3) 椭圆：angle 控制长轴倾角，区间 0~360 表示画完整椭圆。
    cv2.ellipse(canvas, (250, 600), (180, 80), 20, 0, 360, MAGENTA, 2, cv2.LINE_AA)

    # 4) 带箭头的直线。
    cv2.arrowedLine(canvas, (60, 740), (470, 800), ORANGE, 2, cv2.LINE_AA, 0, 0.05)


def draw_filled(canvas):
    """
    @brief  绘制填充图形、多边形与亚像素线条。
    @param  canvas 目标画布。
    """
    # 1) 填充矩形与填充圆：thickness = -1。
    cv2.rectangle(canvas, (560, 80), (740, 230), BLUE, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(canvas, (870, 155), 75, GREEN, cv2.FILLED, cv2.LINE_AA)

    # 2) fillPoly：用点集围成的任意多边形。
    triangle = np.array([[640, 300], [560, 450], [720, 450]], dtype=np.int32)
    cv2.fillPoly(canvas, [triangle], MAGENTA, cv2.LINE_AA)

    # 3) polylines：一次画多条折线，并标出拐点。
    zigzag = np.array([[780, 300], [850, 380], [920, 310], [960, 450]], dtype=np.int32)
    cv2.polylines(canvas, [zigzag], False, RED, 2, cv2.LINE_AA)
    for point in zigzag:
        cv2.circle(canvas, tuple(point), 5, ORANGE, cv2.FILLED, cv2.LINE_AA)

    # 4) 亚像素坐标：坐标左移 shift 位做定点数运算（Python 接口不接受浮点元组）。
    shift = 4  # 定点小数位，坐标值需乘以 2^shift
    cv2.line(canvas, (560 << shift, 534 << shift), (960 << shift, 534 << shift),
             GRAY, 1, cv2.LINE_AA, shift)
    cv2.putText(canvas, "shift=4 (sub-pixel)", (560, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)


def draw_text(canvas):
    """
    @brief  绘制文字：字体、字号、粗细、描边背景与亚像素定位。
    @param  canvas 目标画布。
    @note   中文无法用 cv2.putText 渲染，只能走 freetype 模块的 cv2.freetype.putText。
    """
    cv2.putText(canvas, "OpenCV Drawing", (60, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, BLUE, 3, cv2.LINE_AA)
    cv2.putText(canvas, "FONT_HERSHEY_DUPLEX", (60, 950),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, MAGENTA, 2, cv2.LINE_AA)
    cv2.putText(canvas, "FONT_ITALIC", (60, 995),
                cv2.FONT_ITALIC, 1.0, GREEN, 2, cv2.LINE_AA)

    # 反色标签：先铺实心矩形再写白字，是最常用的标注样式。
    label = "annotated"
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = 760, 880
    cv2.rectangle(canvas, (x, y), (x + text_w + 20, y + text_h + baseline + 16),
                  BLUE, cv2.FILLED, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 10, y + text_h + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    """@brief 主流程：准备画布、依次绘制三类内容、保存结果。"""
    parser = argparse.ArgumentParser(
        description="几何绘制演示：在画布上绘制形状、样式与文字")
    parser.add_argument("-i", "--input", default=None,
                        help="画布素材路径，缺省读取 assets/canvas.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/geometric_drawing.png")
    parser.add_argument("-w", "--width", type=int, default=1024, help="空白画布宽度")
    parser.add_argument("-H", "--height", type=int, default=1024, help="空白画布高度")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("canvas.png")

    # 步骤 1：创建画布
    canvas = load_or_create_canvas(input_path, args.width, args.height)

    # 步骤 2：绘制几何形状
    draw_outlines(canvas)
    draw_filled(canvas)

    # 步骤 3：设置样式并标注文字
    draw_text(canvas)

    # 步骤 4：保存 / 显示结果
    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "geometric_drawing.png")

    if not cv2.imwrite(output_path, canvas):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("画布尺寸: {}x{}, 结果已保存至 {}".format(
        canvas.shape[1], canvas.shape[0], os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("geometric drawing", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
