# -*- coding: utf-8 -*-
"""
@file    optical_flow.py
@brief   光流法演示：Lucas-Kanade 稀疏光流与 Farneback 稠密光流。
@details 流程：读取连续两帧 -> 采样特征点 -> 计算光流矢量 -> 可视化运动。
@note    依赖：opencv-python、numpy
         LK 假设「小位移 + 亮度恒定」，位移过大必须靠金字塔层（maxLevel）兜底；
         Farneback 给出逐像素位移场，用 HSV 编码（色相=方向，明度=幅值）最易读。
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


def flow_to_color(flow):
    """
    @brief  把稠密光流编码成 HSV 彩色图。
    @param  flow 双通道浮点光流场 (dx, dy)。
    @return BGR 可视化结果（色相表示方向，明度表示幅值）。
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 0.5).astype(np.uint8)  # OpenCV 的 H 区间是 0~179
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    """@brief 主流程：两帧之间的稀疏与稠密光流计算及可视化。"""
    parser = argparse.ArgumentParser(
        description="光流法演示：LK 稀疏光流 + Farneback 稠密光流")
    parser.add_argument("-a", "--frameA", default=None,
                        help="第一帧，缺省读取 assets/frame_01.png")
    parser.add_argument("-b", "--frameB", default=None,
                        help="第二帧，缺省读取 assets/frame_02.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/optical_flow.png")
    parser.add_argument("-p", "--points", type=int, default=120,
                        help="LK 采样的最大特征点数")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    path_a = args.frameA or locate_asset("frame_01.png")
    path_b = args.frameB or locate_asset("frame_02.png")

    # 步骤 1：读取连续两帧。
    frame_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    frame_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if frame_a is None or frame_b is None:
        missing = path_a if frame_a is None else path_b
        raise FileNotFoundError("无法读取连续帧: {}".format(os.path.abspath(missing)))
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # 步骤 2：采样特征点。角点比边缘更适合跟踪，因为后者存在孔径问题。
    points_a = cv2.goodFeaturesToTrack(gray_a, max(4, args.points), 0.01, 10.0)
    if points_a is None:
        raise RuntimeError("未找到可跟踪的特征点")

    # 步骤 3：计算光流矢量。
    points_b, status, _err = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, points_a, None, winSize=(21, 21), maxLevel=3)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 步骤 4：可视化运动。
    sparse_view = frame_a.copy()
    shifts = []
    for old, new, ok in zip(points_a.reshape(-1, 2), points_b.reshape(-1, 2), status.ravel()):
        if not ok:
            continue
        p0 = (int(round(old[0])), int(round(old[1])))
        p1 = (int(round(new[0])), int(round(new[1])))
        cv2.line(sparse_view, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(sparse_view, p1, 3, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
        shifts.append(np.linalg.norm(new - old))

    dense_color = flow_to_color(flow)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    dense_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mean_dx, mean_dy = float(flow[..., 0].mean()), float(flow[..., 1].mean())
    print("LK 跟踪点: {}/{}, 平均位移: {:.3f} 像素".format(
        len(shifts), len(points_a), float(np.mean(shifts)) if shifts else 0.0))
    print("Farneback 平均光流: dx={:.3f}, dy={:.3f}".format(mean_dx, mean_dy))

    panels = [
        fit_panel(frame_a, "frame A"),
        fit_panel(sparse_view, "LK sparse flow"),
        fit_panel(dense_color, "Farneback (HSV)"),
        fit_panel(dense_magnitude, "flow magnitude"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(path_a)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "optical_flow.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("optical flow", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
