# -*- coding: utf-8 -*-
"""
@file    file_io.py
@brief   文件输入输出演示：图像读写、编码参数控制与 YAML/XML 参数持久化。
@details 流程：读取图像 -> 保存不同格式与质量 -> YAML/XML 读写 -> 序列化参数。
@note    依赖：opencv-python、numpy
         cv2.FileStorage 会根据扩展名自动选择 XML 或 YAML；写入的数组会带尺寸与类型，
         读回时按节点名即可还原，是保存标定参数/模型参数的常用手段。
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
    panel = np.full((int(PANEL_HEIGHT) + TITLE_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
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
    """@brief 主流程：读写图像与参数文件。"""
    parser = argparse.ArgumentParser(
        description="文件输入输出演示：imread/imwrite/FileStorage")
    parser.add_argument("-i", "--input", default=None,
                        help="输入图像路径，缺省读取 assets/image.png")
    parser.add_argument("-o", "--output", default=None,
                        help="输出目录，缺省写入 <目录>/output")
    parser.add_argument("-q", "--quality", type=int, default=90,
                        help="JPEG 压缩质量 0~100")
    parser.add_argument("-d", "--display", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    input_path = args.input or locate_asset("image.png")
    out_dir = args.output or output_dir_for(input_path)
    os.makedirs(out_dir, exist_ok=True)
    quality = max(1, min(100, args.quality))

    # 步骤 1：读取图像。
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(input_path)))
    print("读取: {} ({}x{}, {} 通道)".format(
        input_path, image.shape[1], image.shape[0], image.shape[2]))

    # 步骤 2：保存结果。PNG 无损，JPEG 有损且受 quality 控制。
    resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    png_path = os.path.join(out_dir, "file_io_copy.png")
    jpg_path = os.path.join(out_dir, "file_io_copy.jpg")
    cv2.imwrite(png_path, image)
    cv2.imwrite(jpg_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # 重新读回 JPEG，量化有损压缩带来的差异。
    reloaded = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    mean_diff = float(cv2.absdiff(image, reloaded).mean()) if reloaded is not None else 0.0
    print("写出: {}".format(png_path))
    print("写出: {} (quality={}, 读回平均差异 {:.2f})".format(jpg_path, quality, mean_diff))

    # 步骤 3~4：YAML / XML 读写与序列化。
    sample = image[:64, :64].copy()
    xml_path = os.path.join(out_dir, "params.xml")
    yaml_path = os.path.join(out_dir, "params.yml")
    for path in (xml_path, yaml_path):
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        storage.write("jpeg_quality", quality)
        storage.write("resize_scale", 0.5)
        storage.write("sample_region", sample)
        storage.release()

    reader = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    read_quality = int(reader.getNode("jpeg_quality").real())
    read_scale = reader.getNode("resize_scale").real()
    read_region = reader.getNode("sample_region").mat()
    reader.release()
    print("从 {} 读回: jpeg_quality={}, resize_scale={}, sample_region={}x{}".format(
        xml_path, read_quality, read_scale, read_region.shape[1], read_region.shape[0]))

    panels = [
        fit_panel(image, "original"),
        fit_panel(resized, "resized 50%"),
        fit_panel(gray, "gray"),
        fit_panel(sample, "region 64x64"),
    ]
    result = montage(panels, 2)

    output_path = os.path.join(out_dir, "file_io.png")
    if not cv2.imwrite(output_path, result):
        raise IOError("无法写入输出图像: {}".format(os.path.abspath(output_path)))
    print("结果已保存至 {}".format(os.path.abspath(output_path)))

    if args.display:
        cv2.imshow("file io", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
