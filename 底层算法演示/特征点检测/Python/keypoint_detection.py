# -*- coding: utf-8 -*-
"""
@file    keypoint_detection.py
@brief   特征点检测演示：ORB / SIFT / FAST 三种检测器的数量与耗时对比。
@details 流程：创建检测器 -> 检测关键点 -> 绘制关键点 -> 对比算法性能。
@note    依赖：opencv-python、numpy
         FAST 只做检测、不做尺度与方向，数量最多但对尺度/旋转不具不变性；
         ORB = FAST + 方向 + 描述子，SIFT 有多尺度空间，数量与耗时都更高。
"""

import argparse
import os
import time

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


def run_detector(detector, gray):
    """
    @brief  运行一个检测器并计时。
    @param  detector 检测器对象。
    @param  gray     灰度图。
    @return (keypoints, 耗时毫秒)
    """
    start = time.perf_counter()
    keypoints = detector.detect(gray, None)
    elapsed = (time.perf_counter() - start) * 1000.0
    return keypoints, elapsed


def main():
    """@brief 主流程：三种检测器对比与可视化。"""
    parser = argparse.ArgumentParser(
        description="特征点检测演示：ORB / SIFT / FAST 对比")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/books.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/keypoint_detection.png")
    parser.add_argument("-n", "--maxFeatures", type=int, default=1000,
                        help="各检测器的特征点上限")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("books.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 1：创建检测器。
    max_features = max(50, args.maxFeatures)
    detectors = [
        ("ORB", cv2.ORB_create(max_features), (0, 255, 0)),
        ("SIFT", cv2.SIFT_create(max_features), (0, 200, 255)),
        # FAST 没有数量上限，阈值越低点越多。
        ("FAST", cv2.FastFeatureDetector_create(20, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16),
         (255, 0, 255)),
    ]

    # 步骤 2~3：检测并绘制。
    print("输入: {} ({}x{})".format(input_path, gray.shape[1], gray.shape[0]))
    print("{:<8}{:<12}{}".format("detector", "keypoints", "time(ms)"))
    panels = [fit_panel(gray, "gray input")]
    for name, detector, color in detectors:
        keypoints, elapsed = run_detector(detector, gray)
        view = cv2.drawKeypoints(src, keypoints, None, color,
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("{:<8}{:<12}{:.2f}".format(name, len(keypoints), elapsed))
        panels.append(fit_panel(view, "{}: {}".format(name, len(keypoints))))

    # 步骤 4：对比结果。
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "keypoint_detection.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("keypoint detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
