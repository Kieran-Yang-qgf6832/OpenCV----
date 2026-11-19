# -*- coding: utf-8 -*-
"""
@file    template_matching.py
@brief   模板匹配演示：在多尺度上滑动窗口匹配，定位模板并输出响应热力图。
@details 流程：读取模板与目标 -> 选择匹配方法 -> 多尺度滑动窗口匹配 -> 定位最佳位置。
@note    依赖：opencv-python、numpy
         TM_CCOEFF_NORMED 对亮度线性变化不敏感，但匹配结果只有「平移」自由度；
         模板尺寸一变就失效，所以工程上必须配合多尺度搜索（本演示的核心）。
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
    """@brief 主流程：多尺度模板匹配并输出最佳位置与热力图。"""
    parser = argparse.ArgumentParser(
        description="模板匹配演示：多尺度 matchTemplate 搜索")
    parser.add_argument("-s", "--scene", default=None,
                        help="场景图，缺省读取 assets/scene.png")
    parser.add_argument("-t", "--template", default=None,
                        help="模板图，缺省读取 assets/template.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/template_matching.png")
    parser.add_argument("-a", "--minScale", type=float, default=0.5, help="最小缩放系数")
    parser.add_argument("-b", "--maxScale", type=float, default=1.5, help="最大缩放系数")
    parser.add_argument("-n", "--steps", type=int, default=11, help="缩放扫描点数")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    scene_path = args.scene or locate_asset("scene.png")
    template_path = args.template or locate_asset("template.png")

    # 步骤 1：读取模板和目标。
    scene_color = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    template_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if scene_color is None or template_color is None:
        raise FileNotFoundError("无法读取图像")
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

    # 步骤 2~3：多尺度滑动窗口匹配。matchTemplate 内部就是一次全图的相关滤波。
    min_scale = max(0.1, args.minScale)
    max_scale = max(min_scale, args.maxScale)
    steps = max(2, args.steps)
    scales = np.linspace(min_scale, max_scale, steps)

    best_score, best_scale, best_loc, best_result = -1.0, 1.0, (0, 0), None
    print("尺度扫描 (TM_CCOEFF_NORMED):")
    for i, scale in enumerate(scales):
        scaled = cv2.resize(template_gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
        if scaled.shape[0] >= scene_gray.shape[0] or scaled.shape[1] >= scene_gray.shape[1]:
            continue
        result = cv2.matchTemplate(scene_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
        if i % 2 == 0:
            print("  scale={:.2f} -> peak={:.4f}".format(scale, max_v))
        if max_v > best_score:
            best_score, best_scale, best_loc, best_result = max_v, scale, max_l, result.copy()

    # 步骤 4：定位最佳位置。
    matched = cv2.resize(template_gray, None, fx=best_scale, fy=best_scale,
                         interpolation=cv2.INTER_AREA)
    x, y = best_loc
    match_box = (x, y, matched.shape[1], matched.shape[0])

    scene_view = scene_color.copy()
    cv2.rectangle(scene_view, (x, y), (x + match_box[2], y + match_box[3]),
                  (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(scene_view, "score {:.3f} scale {:.2f}".format(best_score, best_scale),
                (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 响应热力图：越红表示与模板越相似。
    heatmap_source = cv2.normalize(best_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_source, cv2.COLORMAP_JET)

    print()
    print("最佳: scale={:.2f}, score={:.4f}, 位置=({}, {})".format(
        best_scale, best_score, x, y))

    panels = [
        fit_panel(scene_view, "detected"),
        fit_panel(template_color, "template"),
        fit_panel(heatmap, "response heatmap"),
        fit_panel(scene_color[y:y + match_box[3], x:x + match_box[2]], "matched patch"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(scene_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "template_matching.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("template matching", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
