# -*- coding: utf-8 -*-
"""
@file    corner_subpixel.py
@brief   角点亚像素演示：把整数像素角点精化到亚像素精度，并检验收敛性。
@details 流程：粗检测角点 -> 设定搜索窗口 -> cornerSubPix 精化 -> 与抖动初值的结果对比精度。
@note    依赖：opencv-python、numpy
         cornerSubPix 基于梯度正交假设迭代求解：所有边缘方向的梯度都应垂直于
         「角点→邻域点」向量。它要求输入图像有抗锯齿或真实成像的灰度过渡，
         纯二值图（只有 0/255）会退化成一个固定的半像素偏置。
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


def refine(gray, corners, win_size, max_iter=40, epsilon=0.001):
    """
    @brief  对给定初值做亚像素精化。
    @param  gray      灰度图。
    @param  corners   初值 (N, 1, 2)。
    @param  win_size  搜索窗口半径。
    @param  max_iter  最大迭代次数。
    @param  epsilon   收敛阈值。
    @return 精化后的角点 (N, 1, 2)。
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    return cv2.cornerSubPix(gray, corners.copy(), (win_size, win_size), (-1, -1), criteria)


def orthogonality_residual(gx, gy, point, half):
    """
    @brief  计算梯度正交残差：窗口内 sum |∇I(q)·(q-p)|² / N。
    @param  gx     x 方向梯度（float32）。
    @param  gy     y 方向梯度（float32）。
    @param  point  角点位置 (x, y)。
    @param  half   窗口半径。
    @return 平均残差，越小说明角点位置越符合「梯度垂直于角点→邻域点向量」的假设。
    @note   这正是 cornerSubPix 最小化的目标函数，可直接用来量化精化效果。
    """
    x0, y0 = int(round(point[0])), int(round(point[1]))
    height, width = gx.shape
    total, count = 0.0, 0
    for y in range(max(0, y0 - half), min(height, y0 + half + 1)):
        for x in range(max(0, x0 - half), min(width, x0 + half + 1)):
            dx = x - point[0]
            dy = y - point[1]
            value = gx[y, x] * dx + gy[y, x] * dy
            total += value * value
            count += 1
    return total / count if count else 0.0


def mean_residual(gray, corners, half):
    """
    @brief  计算一组角点的平均正交残差。
    @param  gray    灰度图。
    @param  corners 角点位置 (N, 1, 2)。
    @param  half    窗口半径。
    @return 平均残差。
    """
    gx = cv2.Sobel(np.float32(gray), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(gray), cv2.CV_32F, 0, 1, ksize=3)
    points = corners.reshape(-1, 2)
    if len(points) == 0:
        return 0.0
    return float(np.mean([orthogonality_residual(gx, gy, p, half) for p in points]))


def print_error(name, a, b):
    """@brief 计算两组点之间的平均/最大距离。"""
    deltas = np.linalg.norm(a.reshape(-1, 2) - b.reshape(-1, 2), axis=1)
    print("  {:<28} mean={:8.4f} max={:8.4f}".format(name, deltas.mean(), deltas.max()))


def main():
    """@brief 主流程：粗检测、亚像素精化、抖动初值收敛性检验。"""
    parser = argparse.ArgumentParser(
        description="角点亚像素演示：cornerSubPix 精化与收敛性")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/checkerboard.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/corner_subpixel.png")
    parser.add_argument("-n", "--maxCorners", type=int, default=200,
                        help="粗检测角点上限")
    parser.add_argument("-q", "--quality", type=float, default=0.40,
                        help="角点质量阈值，太低会把棋盘外边界的弱角点也收进来")
    parser.add_argument("-m", "--minDistance", type=float, default=60,
                        help="角点最小间距，应大于棋盘格的半格宽")
    parser.add_argument("-w", "--winSize", type=int, default=5,
                        help="cornerSubPix 搜索窗口半径")
    parser.add_argument("-j", "--jitter", type=float, default=1.5,
                        help="收敛性检验时给初值加的最大抖动(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("checkerboard.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 1：粗检测角点（整数坐标）。
    coarse = cv2.goodFeaturesToTrack(
        gray, max(1, args.maxCorners), args.quality, args.minDistance)
    if coarse is None:
        raise RuntimeError("未检测到角点")

    # 步骤 2~3：设定搜索窗口并做亚像素精化。
    win_size = max(2, args.winSize)
    refined = refine(gray, coarse, win_size)

    # 步骤 4：对比精度。给初值加上随机抖动再精化，看结果是否收敛到同一位置。
    rng = np.random.default_rng(20261119)
    jittered = coarse + rng.uniform(-args.jitter, args.jitter, coarse.shape).astype(np.float32)
    jitter_refined = refine(gray, jittered, win_size)

    print("输入: {}, 角点数: {}, 搜索窗口: {}x{}".format(
        input_path, len(coarse), win_size, win_size))
    print_error("整数 -> 亚像素", coarse, refined)
    print_error("抖动初值 -> 亚像素", refined, jitter_refined)

    # 梯度正交残差：cornerSubPix 的目标函数值，越小说明定位越准。
    residual_coarse = mean_residual(gray, coarse, win_size)
    residual_refined = mean_residual(gray, refined, win_size)
    print("  梯度正交残差: 整数 {:.0f} -> 亚像素 {:.0f} ({:.1f}% 改善)".format(
        residual_coarse, residual_refined,
        100.0 * (1.0 - residual_refined / max(1e-9, residual_coarse))))

    # 可视化：整数角点（绿）与亚像素角点（红）。
    coarse_view = src.copy()
    for x, y in coarse.reshape(-1, 2):
        cv2.circle(coarse_view, (int(round(x)), int(round(y))), 5, (0, 255, 0), 1, cv2.LINE_AA)

    refined_view = src.copy()
    for (cx, cy), (rx, ry) in zip(coarse.reshape(-1, 2), refined.reshape(-1, 2)):
        cv2.circle(refined_view, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(refined_view, (int(round(rx)), int(round(ry))), (0, 0, 255),
                       cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

    # 放大一个角点，肉眼才能看出半像素级的差别。
    px, py = refined.reshape(-1, 2)[len(refined) // 2]
    half = 12
    x0 = max(0, int(px) - half)
    y0 = max(0, int(py) - half)
    zoom = refined_view[y0:y0 + 2 * half, x0:x0 + 2 * half]
    zoom = cv2.resize(zoom, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)

    panels = [
        fit_panel(coarse_view, "integer corners (green)"),
        fit_panel(refined_view, "sub-pixel (red cross)"),
        fit_panel(zoom, "zoom x12"),
    ]
    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "corner_subpixel.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("corner subpixel", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
