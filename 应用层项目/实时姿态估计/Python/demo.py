# -*- coding: utf-8 -*-
"""实时姿态估计：OpenPose(COCO 18 关键点) 在视频上逐帧估计人体骨架。

思路：
    1. readNetFromCaffe 加载 OpenPose COCO 模型，输入缩放到 288 宽再送入网络；
    2. 网络输出 19 张热力图（18 个关键点 + 1 张背景），取每张图的峰值作为关节位置；
    3. 置信度低于阈值的关节点直接丢弃，其余按骨架拓扑连线；
    4. 逐帧输出带骨架的视频，并统计各关节的平均置信度。

核心函数：cv2.dnn.readNetFromCaffe / net.forward / cv2.resize
输出：output/pose.mp4、pose_frame.png
"""

import argparse
import os
import subprocess
import sys

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖：pip install opencv-contrib-python numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ASSETS = os.path.join(PROJ, "assets")
OUT = os.path.join(PROJ, "output")
MODELS = os.path.join(os.path.dirname(PROJ), "_models")

# COCO 18 关键点顺序
NAMES = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri",
         "Rhip", "Rknee", "Rankle", "Lhip", "Lknee", "Lankle", "Reye", "Leye", "Rear", "Lear"]
PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(0, 255, 255), (0, 128, 255), (255, 128, 0), (255, 0, 128), (0, 255, 0), (128, 255, 0)]


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def write_video(path, frames, fps=25.0):
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "22", path], check=False)
    if done.returncode == 0 and os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def load_net():
    proto = os.path.join(MODELS, "openpose_pose_coco.prototxt")
    model = os.path.join(MODELS, "openpose_pose_coco.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(model)):
        sys.exit("缺少 OpenPose 模型：_models/openpose_pose_coco.*（见 README 的模型下载说明）")
    return cv2.dnn.readNetFromCaffe(proto, model)


def infer(net, frame, input_w=288, conf=0.25):
    """返回 {关节索引: (x, y, 置信度)}，坐标已映射回原图。"""
    h, w = frame.shape[:2]
    scale = input_w / float(max(h, w))
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    canvas = np.zeros((input_w, input_w, 3), np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255, (input_w, input_w), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    joints = {}
    for i in range(18):
        heat = out[0, i]
        _, score, _, loc = cv2.minMaxLoc(heat)
        if score < conf:
            continue
        # 热力图坐标 -> 输入画布坐标(用整块 input_w，不是缩放后图像) -> 原图坐标
        x = int(loc[0] * input_w / heat.shape[1] / scale)
        y = int(loc[1] * input_w / heat.shape[0] / scale)
        if 0 <= x < w and 0 <= y < h:
            joints[i] = (x, y, float(score))
    return joints


def draw_skeleton(frame, joints, conf_map):
    for k, (x, y, score) in joints.items():
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        conf_map.setdefault(k, []).append(score)
    for idx, (a, b) in enumerate(PAIRS):
        if a in joints and b in joints:
            color = COLORS[idx % len(COLORS)]
            cv2.line(frame, joints[a][:2], joints[b][:2], color, 3)
    cv2.putText(frame, "joints: %d/18" % len(joints), (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="OpenPose 人体关键点估计")
    ap.add_argument("--image", default=os.path.join(ASSETS, "person.png"),
                    help="全身人像，用来展示完整的 18 点骨架")
    ap.add_argument("--video", default=os.path.join(ASSETS, "pose.mp4"))
    ap.add_argument("--input-size", type=int, default=288)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    net = load_net()

    # 先处理全身人像：这一张能检出全部 18 个关节，作为骨架质量的主图
    still = cv2.imread(args.image)
    if still is not None:
        joints = infer(net, still, args.input_size, args.conf)
        conf_map = {}
        vis = draw_skeleton(still.copy(), joints, conf_map)
        cv2.imwrite(out_path("pose_still.png"), vis)
        print("全身人像：检出 %d/18 个关节" % len(joints))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("读不到视频：%s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames, conf_map, counts = [], {}, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        joints = infer(net, frame, args.input_size, args.conf)
        counts.append(len(joints))
        frames.append(draw_skeleton(frame, joints, conf_map))
    cap.release()
    if not frames:
        sys.exit("没有读到帧")

    write_video(out_path("pose.mp4"), frames, fps)
    cv2.imwrite(out_path("pose_frame.png"), frames[len(frames) // 2])

    print("处理 %d 帧，平均每帧检出 %.1f/18 个关节，最高 %d 个" % (len(frames), np.mean(counts), max(counts)))
    top = sorted(conf_map.items(), key=lambda kv: -np.mean(kv[1]))[:6]
    print("置信度最高的关节：", ["%s %.2f" % (NAMES[k], np.mean(v)) for k, v in top])
    print("已保存 output/pose.mp4、pose_frame.png")

    if args.display:
        for f in frames:
            cv2.imshow("pose", f)
            if cv2.waitKey(30) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
