# 人脸识别

## 需求
建立小底库（两人）用 LBPH 训练，再识别未参与训练的人脸照片，并判断是否为库外人员。

## 作用

- **考勤/门禁**：小规模、固定人员场景，LBPH 足够且不需要 GPU。
- **相册聚类**：把人脸按人归堆。
- **访客甄别**：先判"是不是在册人员"，再走后续流程。

适用边界：LBPH 是纹理特征，对光照/姿态有一定容忍度，但**长相接近的人容易混淆**（见"已知局限"）。

## 核心函数
`cv2.face.LBPHFaceRecognizer_create()`、`train()`、`predict()`

## 关键步骤
1. res10 SSD 自动定位并裁出人脸，灰度 + 直方图均衡，统一到 128×128
2. 每人 1 张注册照 → 镜像/旋转 ±10°/亮度 ±15% 增强成 6 个训练样本
3. LBPH 训练（半径 1、8×8 网格）
4. 探针为**训练集里没出现过的**旋转+亮度组合；距离 > 70 判为库外
5. 输出 4 列网格卡片，逐张写明判给谁、距离、是否正确

## 输入
底库：`face.png`（ID-A）、`../实时姿态估计/assets/person.png`（ID-B）；
库外探针：`probe.png`

## 输出
识别结果网格图（绿=正确、红=错误）

## 素材
`assets/face.png`、`probe.png`：两张不同人物的合成人像（同一 AI 模型生成，长相偏近）。

## 运行
```bash
cd Python && python3 demo.py --display
python3 demo.py --threshold 60
```
依赖 OpenCV（含 contrib，`cv2.face` 在 contrib 里）+ NumPy；人脸检测模型见 `_models/res10_*`。

## 演示结果
`output/face_recognize.png`：7 个探针中 **6 个判对**，
ID-A/ID-B 的 6 个未见扰动全部识对（距离 15~40），库外人员被误判成 ID-A（距离 55.9）。

## 已知局限（这条演示不够干净，如实说明）
- **误纳**：库外那张脸与 ID-A 的 LBPH 距离只有 55.9，和 ID-A 自身的合法探针（40 左右）拉不开，
  阈值调到能拒绝库外，就会误拒本人。
- 试过 EigenFace、FisherFace：FisherFace 在个别扰动的距离甚至低到 7，比 LBPH 更糟。
- 要可靠拒识需要人脸 embedding（FaceNet/ArcFace 的 ONNX），那已经超出"只用 OpenCV 自带能力"。
- 因此本选题的可信部分是**小底库内的身份区分**，拒识能力只作流程演示。
