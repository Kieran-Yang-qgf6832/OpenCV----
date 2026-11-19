# -*- coding: utf-8 -*-
"""
@file    feature_detection.py
@brief   特征检测演示：多种检测器的关键点数量分布与小角度旋转下的重复率。
@details 流程：创建不同检测器 -> 检测关键点 -> 对比数量分布 -> 评估重复性。
@note    依赖：opencv-python、numpy
         重复率的定义：把图像旋转 θ 后重新检测，把所有点映射回原坐标系，
         若距离原关键点 3 像素内存在对应点即视为重复。这个指标直接反映检测器的旋转不变性。
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


def repeatability(detector, gray, angle_deg, tolerance):
    """
    @brief  计算旋转后的重复率。
    @param  detector   检测器。
    @param  gray       灰度图。
    @param  angle_deg  旋转角度（度）。
    @param  tolerance  判定为重复的距离阈值（像素）。
    @return (重复率 0~1, 原图关键点数)
    @note   旋转后需要把新关键点用逆变换映射回原坐标系，再和原关键点比对。
    """
    base_points = detector.detect(gray, None)
    center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(gray, rotation, (gray.shape[1], gray.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated_points = detector.detect(rotated, None)
    if not base_points or not rotated_points:
        return 0.0, len(base_points)

    inverse = cv2.invertAffineTransform(rotation)
    coords = np.float32([kp.pt for kp in rotated_points]).reshape(-1, 2)
    mapped = cv2.transform(coords.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    base = np.float32([kp.pt for kp in base_points]).reshape(-1, 2)

    # 用广播一次性算出「每个基础点与所有映射点的最小距离」。
    distances = np.linalg.norm(base[:, None, :] - mapped[None, :, :], axis=2)
    matched = int((distances.min(axis=1) <= tolerance).sum())
    return matched / float(len(base)), len(base_points)


def main():
    """@brief 主流程：四种检测器的数量对比与旋转重复率评估。"""
    parser = argparse.ArgumentParser(
        description="特征检测演示：多检测器数量分布与旋转重复率")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/building.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/feature_detection.png")
    parser.add_argument("-n", "--maxFeatures", type=int, default=800,
                        help="各检测器的特征点上限")
    parser.add_argument("-a", "--angle", type=float, default=5,
                        help="重复性评估的旋转角度(度)")
    parser.add_argument("-t", "--tolerance", type=float, default=3.0,
                        help="重复判定距离阈值(像素)")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("building.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 步骤 1：创建不同检测器。
    max_features = max(50, args.maxFeatures)
    detectors = [
        ("ORB", cv2.ORB_create(max_features)),
        ("SIFT", cv2.SIFT_create(max_features)),
        ("AKAZE", cv2.AKAZE_create()),
        ("BRISK", cv2.BRISK_create()),
    ]

    rng = np.random.default_rng(20261119)
    panels = [fit_panel(gray, "gray input")]
    print("输入: {}, 旋转 {} 度".format(input_path, args.angle))
    print("{:<8}{:<12}{}".format("detector", "keypoints", "repeatability"))

    for name, detector in detectors:
        # 步骤 2：检测关键点。
        keypoints = detector.detect(gray, None)
        # 步骤 3~4：对比数量并评估重复性。
        rate, _base = repeatability(detector, gray, args.angle, args.tolerance)
        print("{:<8}{:<12}{:.2f}%".format(name, len(keypoints), rate * 100.0))

        color = tuple(int(v) for v in rng.integers(60, 255, 3))
        view = cv2.drawKeypoints(src, keypoints, None, color,
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        panels.append(fit_panel(view, "{}: {} pts, rep {:.1f}%".format(
            name, len(keypoints), rate * 100.0)))

    result = montage(panels, 3)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "feature_detection.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("feature detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
