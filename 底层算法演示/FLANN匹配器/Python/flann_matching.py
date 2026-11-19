# -*- coding: utf-8 -*-
"""
@file    flann_matching.py
@brief   FLANN 匹配器演示：用 KD 树索引做快速近似最近邻匹配，并与暴力匹配对比耗时。
@details 流程：选择索引算法 -> 设置搜索参数 -> kNN 匹配 -> 比率测试筛选。
@note    依赖：opencv-python、numpy
         FLANN 是「近似」最近邻：用少量精度换取速度，描述子维度越高、数据量越大收益越明显；
         SIFT 的浮点描述子用 KDTreeIndexParams，ORB/AKAZE 的二进制描述子需改用 LSH 索引。
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


def ratio_test(knn, ratio):
    """
    @brief  用比率测试从 knn 结果里筛出好匹配。
    @param  knn   每个查询点的两个最近邻。
    @param  ratio 比率阈值。
    @return 好匹配列表。
    """
    return [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def main():
    """@brief 主流程：FLANN 与 BFMatcher 的匹配质量与耗时对比。"""
    parser = argparse.ArgumentParser(
        description="FLANN 匹配器演示：KDTree 索引 + 近似 kNN")
    parser.add_argument("-a", "--imageA", default=None,
                        help="图像 A，缺省读取 assets/scene.png")
    parser.add_argument("-b", "--imageB", default=None,
                        help="图像 B，缺省读取 assets/scene_rotated.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/flann_matching.png")
    parser.add_argument("-r", "--ratio", type=float, default=0.75, help="Lowe 比率阈值")
    parser.add_argument("-t", "--trees", type=int, default=5, help="KD 树数量")
    parser.add_argument("-c", "--checks", type=int, default=50,
                        help="搜索时检查的叶节点数，越大越精确")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    path_a = args.imageA or locate_asset("scene.png")
    path_b = args.imageB or locate_asset("scene_rotated.png")

    image_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    image_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise FileNotFoundError("无法读取图像")

    # 步骤 1：用 SIFT 得到浮点描述子（FLANN 的 KD 树适用于浮点特征）。
    sift = cv2.SIFT_create(1500)
    kp_a, desc_a = sift.detectAndCompute(image_a, None)
    kp_b, desc_b = sift.detectAndCompute(image_b, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("特征提取失败")

    # 步骤 2：设置 FLANN 索引与搜索参数。
    index_params = dict(algorithm=1, trees=max(1, args.trees))  # 1 = FLANN_INDEX_KDTREE
    search_params = dict(checks=max(1, args.checks))
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 步骤 3：kNN 匹配，同时用暴力匹配作对照。
    start = time.perf_counter()
    knn_flann = flann.knnMatch(desc_a, desc_b, k=2)
    flann_ms = (time.perf_counter() - start) * 1000.0

    brute = cv2.BFMatcher(cv2.NORM_L2)
    start = time.perf_counter()
    knn_brute = brute.knnMatch(desc_a, desc_b, k=2)
    brute_ms = (time.perf_counter() - start) * 1000.0

    # 步骤 4：比率测试筛选并可视化。
    good_flann = ratio_test(knn_flann, args.ratio)
    good_brute = ratio_test(knn_brute, args.ratio)

    flann_view = cv2.drawMatches(image_a, kp_a, image_b, kp_b, good_flann, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    brute_view = cv2.drawMatches(image_a, kp_a, image_b, kp_b, good_brute, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print("描述子: A {}x{}, B {}x{}".format(desc_a.shape[0], desc_a.shape[1],
                                           desc_b.shape[0], desc_b.shape[1]))
    print("{:<10}{:<12}{}".format("matcher", "good", "time(ms)"))
    print("{:<10}{:<12}{:.2f}".format("FLANN", len(good_flann), flann_ms))
    print("{:<10}{:<12}{:.2f}".format("BFMatcher", len(good_brute), brute_ms))

    panels = [
        fit_panel(flann_view, "FLANN: {}".format(len(good_flann))),
        fit_panel(brute_view, "BFMatcher: {}".format(len(good_brute))),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(path_a)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "flann_matching.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("flann matching", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
