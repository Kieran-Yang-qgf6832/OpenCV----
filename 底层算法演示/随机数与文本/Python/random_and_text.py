# -*- coding: utf-8 -*-
"""
@file    random_and_text.py
@brief   随机数与文本演示：用 numpy/cv2 随机数生成数据并绘制随机图形与文字。
@details 流程：创建 RNG -> 生成随机数矩阵 -> 绘制随机文本 -> 随机图形绘制。
@note    依赖：opencv-python、numpy
         Python 侧用 np.random.default_rng(seed) 保证可复现；
         cv2.randu / cv2.randn 也可以，但它们作用于预分配 Mat，等价写法是 numpy 直接生成。
"""

import argparse
import os

import cv2
import numpy as np

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TITLE_HEIGHT = 30

# 随机文本使用的字符表（cv2.putText 只支持 ASCII）。
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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


def random_string(rng, length):
    """@brief 生成指定长度的随机字符串。"""
    return "".join(rng.choice(list(ALPHABET), length))


def main():
    """@brief 主流程：随机数矩阵 + 随机图形 + 随机文本。"""
    parser = argparse.ArgumentParser(
        description="随机数与文本演示：随机数矩阵、随机图形与随机文本")
    parser.add_argument("-i", "--input", default=None,
                        help="画布素材，缺省读取 assets/canvas.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出图像路径，缺省写入 <目录>/output/random_and_text.png")
    parser.add_argument("-s", "--seed", type=int, default=20261119,
                        help="随机种子，相同种子结果完全一致")
    parser.add_argument("-n", "--count", type=int, default=40,
                        help="随机图形与文本的数量")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("canvas.png")
    canvas = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if canvas is None:
        canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)

    # 步骤 1：创建 RNG 对象。种子相同则整个序列完全可复现。
    rng = np.random.default_rng(args.seed)
    count = max(1, args.count)
    height, width = canvas.shape[:2]

    # 步骤 2：生成随机数矩阵。
    uniform_noise = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    gaussian_noise = rng.normal(128, 40, (256, 256)).astype(np.float32)
    gaussian_view = np.clip(gaussian_noise, 0, 255).astype(np.uint8)

    print("均匀分布 uniform(0,256): mean={:.1f}, std={:.1f}".format(
        uniform_noise.mean(), uniform_noise.std()))
    print("正态分布 normal(128,40): mean={:.1f}, std={:.1f}".format(
        gaussian_noise.mean(), gaussian_noise.std()))

    # 步骤 3：绘制随机文本。
    text_view = canvas.copy()
    for _ in range(count):
        origin = (int(rng.integers(20, max(21, width - 260))),
                  int(rng.integers(40, max(41, height - 40))))
        cv2.putText(text_view, random_string(rng, int(rng.integers(3, 10))), origin,
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.6)),
                    tuple(int(v) for v in rng.integers(30, 255, 3)),
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    cv2.putText(text_view, "seed {} text {}".format(args.seed, count), (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # 步骤 4：随机图形绘制。
    shape_view = canvas.copy()
    for _ in range(count):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(v) for v in rng.integers(30, 255, 3))
        thickness = int(rng.integers(-1, 5))  # -1 表示填充
        kind = int(rng.integers(0, 4))
        if kind == 0:
            cv2.line(shape_view, p1, p2, color, max(1, thickness), cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(shape_view, p1, int(rng.integers(10, 80)), color,
                       cv2.FILLED if thickness < 0 else thickness, cv2.LINE_AA)
        elif kind == 2:
            cv2.rectangle(shape_view, p1, p2, color,
                          cv2.FILLED if thickness < 0 else thickness, cv2.LINE_AA)
        else:
            triangle = np.array([p1, p2,
                                 (int(rng.integers(0, width)), int(rng.integers(0, height)))],
                                dtype=np.int32)
            cv2.fillPoly(shape_view, [triangle], color, cv2.LINE_AA)
    cv2.putText(shape_view, "seed {} shapes {}".format(args.seed, count), (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    panels = [
        fit_panel(uniform_noise, "randu uniform(0,256)"),
        fit_panel(gaussian_view, "randn normal(128,40)"),
        fit_panel(shape_view, "random shapes"),
        fit_panel(text_view, "random text"),
    ]
    result = montage(panels, 2)

    if args.output:
        output_path = args.output
    else:
        out_dir = output_dir_for(input_path)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "random_and_text.png")

    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("random and text", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
