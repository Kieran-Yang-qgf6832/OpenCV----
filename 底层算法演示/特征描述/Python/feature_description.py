# -*- coding: utf-8 -*-
"""
@file    feature_description.py
@brief   特征描述演示：计算 ORB / SIFT 描述子，分析维度并检验旋转下的匹配能力。
@details 流程：检测关键点 -> 计算描述子 -> 分析维度 -> 可视化描述子并与旋转版本匹配。
@note    依赖：opencv-python、numpy
         ORB 是 32 字节二进制描述子（汉明距离），SIFT 是 128 维浮点描述子（欧氏距离）；
         描述子的判别力要看「同一特征在变换后仍能匹配上」的比例，而不是维度高低。
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


def visualize_descriptors(descriptors, max_rows):
    """
    @brief  把描述子矩阵渲染成可视化图像（每行一个描述子）。
    @param  descriptors 描述子矩阵。
    @param  max_rows    最多显示多少行。
    @return 灰度可视化图，宽度已放大。
    """
    sliced = descriptors[:max_rows].astype(np.float32)
    normalized = cv2.normalize(sliced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 描述子通常很窄（32 或 128 列），横向放大 4 倍才看得清每个比特/维度。
    return cv2.resize(normalized, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)


def rotation_match_rate(detector, gray, norm_type):
    """
    @brief  统计旋转 15° 后的描述子匹配率。
    @param  detector  检测器（同时提供描述子）。
    @param  gray      灰度图。
    @param  norm_type 距离度量。
    @return (匹配率, 好匹配数, 原图描述子数, 匹配可视化)
    """
    kp_base, desc_base = detector.detectAndCompute(gray, None)

    center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
    rotated = cv2.warpAffine(gray, cv2.getRotationMatrix2D(center, 15.0, 1.0),
                             (gray.shape[1], gray.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    kp_rot, desc_rot = detector.detectAndCompute(rotated, None)

    if desc_base is None or desc_rot is None or len(desc_base) == 0:
        return 0.0, 0, 0, np.zeros((10, 10, 3), np.uint8)

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc_base, desc_rot, k=2)
    good = [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    view = cv2.drawMatches(gray, kp_base, rotated, kp_rot, good, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good) / float(len(desc_base)), len(good), len(desc_base), view


def main():
    """@brief 主流程：两种描述子的维度分析与旋转匹配检验。"""
    parser = argparse.ArgumentParser(
        description="特征描述演示：ORB / SIFT 描述子维度与旋转匹配率")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/church.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/feature_description.png")
    parser.add_argument("-n", "--maxFeatures", type=int, default=500,
                        help="特征点上限")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("church.png")
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    max_features = max(50, args.maxFeatures)

    # 步骤 1~2：检测关键点并计算描述子。
    orb = cv2.ORB_create(max_features)
    sift = cv2.SIFT_create(max_features)
    kp_orb, desc_orb = orb.detectAndCompute(gray, None)
    kp_sift, desc_sift = sift.detectAndCompute(gray, None)

    # 步骤 3：分析维度。
    print("输入: {}".format(input_path))
    print("  ORB  rows={} cols={} dtype={}".format(desc_orb.shape[0], desc_orb.shape[1],
                                                  desc_orb.dtype))
    print("  SIFT rows={} cols={} dtype={}".format(desc_sift.shape[0], desc_sift.shape[1],
                                                  desc_sift.dtype))

    # 步骤 4：可视化描述子并做旋转匹配检验。
    orb_rate, orb_good, orb_base, orb_matches = rotation_match_rate(orb, gray, cv2.NORM_HAMMING)
    sift_rate, sift_good, sift_base, _ = rotation_match_rate(sift, gray, cv2.NORM_L2)
    print("旋转 15° 后的匹配率:")
    print("  ORB : {:.1f}% ({}/{})".format(orb_rate * 100, orb_good, orb_base))
    print("  SIFT: {:.1f}% ({}/{})".format(sift_rate * 100, sift_good, sift_base))

    orb_view = cv2.drawKeypoints(src, kp_orb, None, (0, 255, 0),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    panels = [
        fit_panel(orb_view, "ORB keypoints: {}".format(len(kp_orb))),
        fit_panel(visualize_descriptors(desc_orb, 60),
                  "ORB descriptor {}x8bit".format(desc_orb.shape[1])),
        fit_panel(visualize_descriptors(desc_sift, 60),
                  "SIFT descriptor {}xfloat".format(desc_sift.shape[1])),
        fit_panel(orb_matches, "ORB rotated matches"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "feature_description.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("feature description", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
