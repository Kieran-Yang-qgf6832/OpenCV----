# -*- coding: utf-8 -*-
"""
@file    akaze_matching.py
@brief   AKAZE 特征匹配演示：检测关键点与描述子，用 BFMatcher + 比率测试筛选。
@details 流程：检测关键点和描述子 -> BFMatcher 匹配 -> 比率测试筛选 -> 可视化。
@note    依赖：opencv-python、numpy
         AKAZE 基于非线性尺度空间，描述子为二进制（MLDB），因此匹配用汉明距离；
         比率测试比较「最近邻」与「次近邻」，能挡掉大量只有唯一候选的误匹配。
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
    """@brief 主流程：AKAZE 匹配与筛选可视化。"""
    parser = argparse.ArgumentParser(
        description="AKAZE 匹配演示：检测 + BFMatcher + 比率测试")
    parser.add_argument("-a", "--imageA", default=None,
                        help="图像 A，缺省读取 assets/scene.png")
    parser.add_argument("-b", "--imageB", default=None,
                        help="图像 B，缺省读取 assets/scene_rotated.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/akaze_matching.png")
    parser.add_argument("-r", "--ratio", type=float, default=0.75, help="Lowe 比率阈值")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    path_a = args.imageA or locate_asset("scene.png")
    path_b = args.imageB or locate_asset("scene_rotated.png")

    image_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    image_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise FileNotFoundError("无法读取图像")

    # 步骤 1：检测关键点和描述子。
    akaze = cv2.AKAZE_create()
    kp_a, desc_a = akaze.detectAndCompute(image_a, None)
    kp_b, desc_b = akaze.detectAndCompute(image_b, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("特征提取失败")

    # 步骤 2~3：BFMatcher + 比率测试。
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < args.ratio * pair[1].distance]
    all_first = [pair[0] for pair in knn if len(pair) >= 1]

    # 步骤 4：可视化。分「全部 knn 首选」与「比率筛选后」两格对比。
    all_view = cv2.drawMatches(image_a, kp_a, image_b, kp_b, all_first, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    good_view = cv2.drawMatches(image_a, kp_a, image_b, kp_b, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print("A 关键点: {}, B 关键点: {}, 描述子维度: {} 字节".format(
        len(kp_a), len(kp_b), desc_a.shape[1]))
    print("knn 首选匹配: {}, 比率 {} 筛选后: {} ({:.1f}%)".format(
        len(all_first), args.ratio, len(good),
        100.0 * len(good) / max(1, len(all_first))))

    panels = [
        fit_panel(image_a, "image A"),
        fit_panel(image_b, "image B (rotated)"),
        fit_panel(all_view, "knn first: {}".format(len(all_first))),
        fit_panel(good_view, "ratio {:.2f}: {}".format(args.ratio, len(good))),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(path_a)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "akaze_matching.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("akaze matching", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
