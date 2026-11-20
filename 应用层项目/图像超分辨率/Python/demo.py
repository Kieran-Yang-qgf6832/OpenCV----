# -*- coding: utf-8 -*-
"""图像超分辨率：EDSR(x4) 与双三次插值对比，用 PSNR/SSIM 和局部放大图证明差距。

思路：
    1. 输入是参考图 1/4 尺寸的低清图（合成时用 INTER_AREA 降采样，关系明确）；
    2. 双三次插值作为基线，EDSR 用 cv2.dnn_superres 端到端放大 4 倍；
    3. 与参考图逐像素比较 PSNR、SSIM，并在纹理密集处截取局部放大对照；
    4. 输出三联图（双三次 / EDSR / 参考）与数值结论。

核心函数：cv2.dnn_superres.DnnSuperResImpl_create / readModel / upsample
输出：output/superres.png、superres_crop.png
"""

import argparse
import os
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

SCALE = 4


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def ssim(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu1, mu2 = cv2.GaussianBlur(ga, k, 1.5), cv2.GaussianBlur(gb, k, 1.5)
    s1 = cv2.GaussianBlur(ga * ga, k, 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(gb * gb, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(ga * gb, k, 1.5) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return float(m.mean())


def label(img, text, size=None):
    out = img.copy() if size is None else cv2.resize(img, size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="EDSR 超分辨率与双三次插值对比")
    ap.add_argument("--lowres", default=os.path.join(ASSETS, "lowres.png"))
    ap.add_argument("--reference", default=os.path.join(ASSETS, "reference.png"))
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    model_path = os.path.join(MODELS, "EDSR_x4.pb")
    if not os.path.exists(model_path):
        sys.exit("缺少模型：_models/EDSR_x4.pb（见 README 的模型下载说明）")

    lr = cv2.imread(args.lowres)
    ref = cv2.imread(args.reference)
    if lr is None or ref is None:
        sys.exit("读不到 lowres/reference")
    print("输入 %dx%d，参考 %dx%d" % (lr.shape[1], lr.shape[0], ref.shape[1], ref.shape[0]))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", SCALE)
    up = sr.upsample(lr)
    bicubic = cv2.resize(lr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)

    psnr_bic, psnr_es = cv2.PSNR(ref, bicubic), cv2.PSNR(ref, up)
    ssim_bic, ssim_es = ssim(ref, bicubic), ssim(ref, up)
    print("双三次：PSNR %.2f dB / SSIM %.4f" % (psnr_bic, ssim_bic))
    print("EDSR  ：PSNR %.2f dB / SSIM %.4f（提升 %.2f dB）" % (psnr_es, ssim_es, psnr_es - psnr_bic))

    h, w = ref.shape[:2]
    cx, cy, cs = int(w * 0.62), int(h * 0.55), int(min(h, w) * 0.22)
    def crop(img):
        return img[cy : cy + cs, cx : cx + cs]

    bottom = np.hstack([label(bicubic, "bicubic x4 %.2f dB" % psnr_bic, (w // 2, h // 2)),
                        label(up, "EDSR x4 %.2f dB" % psnr_es, (w // 2, h // 2))])
    full = np.vstack([bottom, label(ref, "ground truth", (w, h // 2))])
    cv2.imwrite(out_path("superres.png"), full)

    zoom = np.hstack([label(crop(bicubic), "bicubic", (300, 300)),
                      label(crop(up), "EDSR", (300, 300)),
                      label(crop(ref), "ground truth", (300, 300))])
    cv2.imwrite(out_path("superres_crop.png"), zoom)
    print("已保存 output/superres.png、superres_crop.png")

    if args.display:
        cv2.imshow("superres", zoom)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
