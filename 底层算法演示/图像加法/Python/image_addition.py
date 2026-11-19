# -*- coding: utf-8 -*-
"""
@file    image_addition.py
@brief   图像加法演示：直接相加、加权混合与 ROI 局部融合。
@details 流程：读取两幅图像 -> 尺寸对齐 -> cv2.add 直接相加 -> addWeighted 加权混合
         -> ROI 局部融合 -> 统计过曝比例并输出对比图。
@note    依赖：opencv-python、numpy
         必须用 cv2.add 而不是 numpy 的 a + b：uint8 相加会取模回绕（200+100=44），
         cv2.add 才会饱和到 255，与 C++ 的 cv::add / Mat 表达式保持一致。
"""

import argparse
import os

import cv2
import numpy as np

# 拼接图中标题条的高度（像素）。
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


def make_panel(image, title):
    """
    @brief  给图像加标题条，便于拼接展示。
    @param  image 三通道图像。
    @param  title 标题文字（仅支持 ASCII）。
    @return 带标题条的面板。
    """
    # 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    panel = np.full((small.shape[0] + TITLE_HEIGHT, small.shape[1], 3), 255, dtype=np.uint8)
    panel[TITLE_HEIGHT:] = small
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


def saturated_ratio(image):
    """
    @brief  统计三通道中饱和到 255 的子像素比例，衡量相加后的过曝程度。
    @param  image 三通道图像。
    @return 过曝比例（0~1）。
    """
    return float(np.count_nonzero(image == 255)) / image.size


def main():
    """@brief 主流程：读取两图、三种加法策略、ROI 融合并输出对比图。"""
    parser = argparse.ArgumentParser(
        description="图像加法演示：直接相加 / 加权混合 / ROI 融合")
    parser.add_argument("-a", "--imageA", default=None,
                        help="第一幅图像，缺省读取 assets/image_a.png")
    parser.add_argument("-b", "--imageB", default=None,
                        help="第二幅图像，缺省读取 assets/image_b.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/image_addition.png")
    parser.add_argument("--alpha", type=float, default=0.5, help="加权混合中第一幅图的权重")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    path_a = args.imageA or locate_asset("image_a.png")
    path_b = args.imageB or locate_asset("image_b.png")

    # 步骤 1：读取两幅图像并统一尺寸（逐像素运算要求尺寸与通道数一致）。
    src_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    src_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if src_a is None or src_b is None:
        missing = path_a if src_a is None else path_b
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(missing)))
    if src_a.shape != src_b.shape:
        src_b = cv2.resize(src_b, (src_a.shape[1], src_a.shape[0]), interpolation=cv2.INTER_AREA)

    # 步骤 2：直接相加。亮度整体抬高，亮部很容易饱和到 255。
    added = cv2.add(src_a, src_b)

    # 步骤 3：加权混合。alpha 越大越偏向第一幅图，权重和为 1 时整体亮度不变。
    alpha = args.alpha
    blended = cv2.addWeighted(src_a, alpha, src_b, 1.0 - alpha, 0.0)
    blend_light = cv2.addWeighted(src_a, alpha * 0.5, src_b, 1.0 - alpha * 0.5, 0.0)
    heavy = min(1.0, alpha + 0.25)
    blend_heavy = cv2.addWeighted(src_a, heavy, src_b, 1.0 - heavy, 0.0)

    # 步骤 4：ROI 融合。只在指定区域内混合，区域外保持第一幅图。
    roi_result = src_a.copy()
    small_b = cv2.resize(src_b, (src_a.shape[1] // 3, src_a.shape[0] // 3),
                         interpolation=cv2.INTER_AREA)
    x, y = 60, 60
    roi = roi_result[y:y + small_b.shape[0], x:x + small_b.shape[1]]
    roi_result[y:y + small_b.shape[0], x:x + small_b.shape[1]] = cv2.addWeighted(
        roi, 0.5, small_b, 0.5, 0.0)
    cv2.rectangle(roi_result, (x, y), (x + small_b.shape[1], y + small_b.shape[0]),
                  (0, 255, 255), 2, cv2.LINE_AA)

    # 统计：直接相加的过曝比例远高于加权混合，这是选择策略的依据。
    diff = cv2.absdiff(added, blended)
    print("尺寸: {}x{}, alpha = {}".format(src_a.shape[1], src_a.shape[0], alpha))
    print("  add 后饱和像素比例:    {:.2f}%".format(100.0 * saturated_ratio(added)))
    print("  blend 后饱和像素比例:  {:.2f}%".format(100.0 * saturated_ratio(blended)))
    print("  add 与 blend 的平均差异: {:.2f}".format(float(diff.mean())))

    panels = [
        make_panel(src_a, "image A"),
        make_panel(src_b, "image B"),
        make_panel(added, "add (saturate)"),
        make_panel(blended, "addWeighted {:.2f}".format(alpha)),
        make_panel(blend_light, "addWeighted lighter"),
        make_panel(blend_heavy, "addWeighted heavier"),
        make_panel(roi_result, "ROI blend"),
        make_panel(diff, "absdiff(add, blend)"),
    ]
    result = montage(panels, 4)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(path_a)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "image_addition.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("image addition", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
