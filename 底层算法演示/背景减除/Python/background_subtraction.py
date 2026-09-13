# -*- coding: utf-8 -*-
"""
@file    background_subtraction.py
@brief   背景减除演示：从视频序列中分离前景运动物体。
@details 流程：读取视频 -> 创建背景减除器 -> 逐帧计算前景掩码 -> 形态学优化 -> 提取目标 -> 输出/显示。
@note    依赖：opencv-python（pip install opencv-python）
"""

import argparse
import os

import cv2

# 目标最小面积（像素）。小于该面积的轮廓视为噪声，可被过滤。
MIN_TARGET_AREA = 500


def create_subtractor(method):
    """
    @brief  创建背景减除器。
    @param  method 背景减除算法名称，可选 "MOG2" 或 "KNN"。
    @return OpenCV 背景减除器对象（BackgroundSubtractorMOG2 / BackgroundSubtractorKNN）。
    @raise  ValueError 当 method 非法时抛出。
    """
    if method == "MOG2":
        # history=500      : 用于建立背景模型的帧数；
        # varThreshold=16  : 像素与背景模型匹配的方差阈值，越小越敏感；
        # detectShadows=True: 阴影被标记为 127，确定前景为 255。
        return cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
    if method == "KNN":
        # history=500       : 用于建立背景模型的帧数；
        # dist2Threshold=400: 样本与像素的距离平方阈值，越小越敏感；
        # detectShadows=True : 阴影被标记为 127，确定前景为 255。
        return cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=True
        )
    raise ValueError("未知的背景减除方法: {} (可选 MOG2 / KNN)".format(method))


def refine_mask(mask):
    """
    @brief  形态学优化前景掩码：去除噪点并填补目标内部空洞。
    @param  mask 背景减除器输出的单通道前景掩码（取值 0 / 127 / 255）。
    @return 优化后的二值前景掩码（取值 0 / 255）。
    """
    # 1) 去阴影：仅保留确定前景（255），丢弃阴影（127）与背景（0）。
    _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # 2) 形态学优化：开运算去孤立噪点，闭运算填补目标内部空洞。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def extract_targets(mask, min_area=MIN_TARGET_AREA):
    """
    @brief  在掩码上提取运动目标轮廓，并过滤小面积噪声。
    @param  mask     二值前景掩码（单通道，0 / 255）。
    @param  min_area 目标最小面积（像素），小于该值的轮廓被忽略。
    @return 目标轮廓列表，每个元素为 Nx1x2 的点集。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_targets(mask, targets):
    """
    @brief  将二值掩码转为可视化图像，并用矩形框标出运动目标。
    @param  mask    二值前景掩码（单通道，0 / 255）。
    @param  targets 目标轮廓列表。
    @return 三通道 BGR 可视化图像。
    """
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for contour in targets:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis


def main():
    """@brief 主流程：解析参数、逐帧背景减除、保存掩码并可选显示。"""
    default_input = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "1789295519752..mp4"
    )
    parser = argparse.ArgumentParser(
        description="背景减除演示：从视频序列中分离前景运动物体"
    )
    parser.add_argument("-i", "--input", default=default_input, help="输入视频路径")
    parser.add_argument(
        "-m", "--method", choices=["MOG2", "KNN"], default="MOG2", help="背景减除方法"
    )
    parser.add_argument("-o", "--output", default=None, help="前景掩码输出视频路径")
    parser.add_argument(
        "-d", "--display", action="store_true", help="弹窗显示（默认关闭）"
    )
    args = parser.parse_args()

    # 步骤 1：打开视频序列
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError("无法打开视频: {}".format(os.path.abspath(args.input)))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出路径缺省为：<输入视频所在目录>/mask_<方法>.mp4
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "mask_{}.mp4".format(args.method)
    )
    writer = cv2.VideoWriter(
        output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=False
    )
    if not writer.isOpened():
        raise IOError("无法创建输出视频: {}".format(output))

    # 步骤 2：创建背景减除器
    subtractor = create_subtractor(args.method)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 步骤 3：逐帧获取前景掩码
        mask = subtractor.apply(frame)

        # 步骤 4：形态学优化
        binary = refine_mask(mask)

        # 步骤 5：提取目标
        targets = extract_targets(binary)

        writer.write(binary)
        if args.display:
            cv2.imshow("input", frame)
            cv2.imshow("foreground mask", draw_targets(binary, targets))
            if cv2.waitKey(30) & 0xFF == 27:  # ESC 退出
                break
        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("处理完成: {} 帧, 前景掩码已保存至 {}".format(frame_index, output))


if __name__ == "__main__":
    main()
