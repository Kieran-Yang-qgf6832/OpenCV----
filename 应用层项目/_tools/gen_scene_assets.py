# -*- coding: utf-8 -*-
"""合成 3D 场景素材生成器（射线投射渲染，带几何真值）。

供三个选题共用：
    立体视觉测距   -> 严格配准的左右视图 + 真实视差图
    深度相机点云   -> 彩色图 + 16 位深度图（毫米），内参已知
    多视角三维重建 -> 环绕拍摄的 6 个视角 + 相机位姿真值

场景是一个封闭房间（后墙/地面/天花板/左右墙）+ 若干立方体。每个像素的深度
都可以解析求出，所以左右视图严格配准，也能给出可直接与估计结果对比的真值。

用法：
    python3 gen_scene_assets.py
"""

import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def texture(seed, base, patches=60):
    """非周期纹理：底色被多尺度噪声调制 + 随机色块 + 散点。

    刻意不用规则棋盘：周期性图案会让立体匹配锁定到错误位移上。
    """
    import cv2

    rng = np.random.default_rng(seed)
    h = w = 320
    low = cv2.resize(rng.random((9, 9)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    mid = cv2.resize(rng.random((40, 40)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mod = 0.55 + 0.55 * low + 0.25 * mid
    img = np.array(base, np.float32)[None, None, :] * mod[..., None]
    for _ in range(patches):
        x, y = rng.integers(0, w), rng.integers(0, h)
        rw, rh = rng.integers(12, 70), rng.integers(12, 70)
        img[y : y + rh, x : x + rw] = rng.integers(20, 235, 3)
    speck = rng.random((h, w, 1)) > 0.72
    img[speck.repeat(3, axis=-1)] = rng.integers(0, 255, int(speck.sum()) * 3)
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def rotation(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rr = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rr @ Rp @ Ry


LIGHT = np.array([0.35, 0.75, -0.55])
LIGHT /= np.linalg.norm(LIGHT)


class Plane:
    """一块有限大小的平面：中心 O，两个正交基 U/V，半宽 half，纹理，朝向法线。"""

    def __init__(self, O, U, V, half, tex, normal_dir=None):
        self.O = np.asarray(O, np.float32)
        U = np.asarray(U, np.float32)
        V = np.asarray(V, np.float32)
        self.U, self.V = U / np.linalg.norm(U), V / np.linalg.norm(V)
        self.half = float(half)
        self.tex = tex
        n = np.cross(self.U, self.V)
        if normal_dir is not None and np.dot(n, np.asarray(normal_dir, np.float32)) < 0:
            n = -n
        self.normal = n
        # 朗伯着色：朝光面亮，背光面暗，让立方体的棱在灰度上可分
        self.shade = float(0.55 + 0.55 * max(0.0, float(np.dot(self.normal, LIGHT))))

    def render(self, rays_o, rays_d, depth, color):
        denom = rays_d @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.O - rays_o) @ self.normal) / denom
        valid = np.isfinite(t) & (t > 1e-3)
        hit = rays_o + t[..., None] * rays_d
        u = (hit - self.O) @ self.U / self.half
        v = (hit - self.O) @ self.V / self.half
        inside = valid & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0) & (t < depth)
        if not inside.any():
            return
        th, tw = self.tex.shape[:2]
        idx = np.nonzero(inside)
        tx = np.clip((u[idx] + 1) * 0.5 * (tw - 1), 0, tw - 1).astype(np.int32)
        ty = np.clip((v[idx] + 1) * 0.5 * (th - 1), 0, th - 1).astype(np.int32)
        depth[idx] = t[idx]
        color[idx] = self.tex[ty, tx].astype(np.float32) * self.shade


def box(center, size, tex_seed, base):
    """轴对齐立方体的 6 个面（朝内的一侧也建，保证任何视角都没有黑洞）。"""
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, np.float32) / 2.0
    t = texture(tex_seed, base)
    # 六个面各自用矩形尺寸，朝内的面也建，保证任何视角都没有黑洞
    faces = [
        ((cx, cy + sy, cz), (1, 0, 0), (0, 0, 1), (sx, sz), [0, 1, 0]),
        ((cx, cy - sy, cz), (1, 0, 0), (0, 0, 1), (sx, sz), [0, -1, 0]),
        ((cx + sx, cy, cz), (0, 0, 1), (0, 1, 0), (sz, sy), [1, 0, 0]),
        ((cx - sx, cy, cz), (0, 0, 1), (0, 1, 0), (sz, sy), [-1, 0, 0]),
        ((cx, cy, cz + sz), (1, 0, 0), (0, 1, 0), (sx, sy), [0, 0, 1]),
        ((cx, cy, cz - sz), (1, 0, 0), (0, 1, 0), (sx, sy), [0, 0, -1]),
    ]
    return [RectPlane(O, U, V, hu, hv, t, nd) for O, U, V, (hu, hv), nd in faces]


class RectPlane(Plane):
    """矩形平面：U/V 方向分别有不同的半宽。"""

    def __init__(self, O, U, V, hu, hv, tex, normal_dir=None):
        super().__init__(O, U, V, 1.0, tex, normal_dir)
        U = np.asarray(U, np.float32)
        V = np.asarray(V, np.float32)
        self.U, self.V = U / np.linalg.norm(U), V / np.linalg.norm(V)
        self.hu, self.hv = float(hu), float(hv)

    def render(self, rays_o, rays_d, depth, color):
        denom = rays_d @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.O - rays_o) @ self.normal) / denom
        valid = np.isfinite(t) & (t > 1e-3)
        hit = rays_o + t[..., None] * rays_d
        u = (hit - self.O) @ self.U / self.hu
        v = (hit - self.O) @ self.V / self.hv
        inside = valid & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0) & (t < depth)
        if not inside.any():
            return
        th, tw = self.tex.shape[:2]
        idx = np.nonzero(inside)
        tx = np.clip((u[idx] + 1) * 0.5 * (tw - 1), 0, tw - 1).astype(np.int32)
        ty = np.clip((v[idx] + 1) * 0.5 * (th - 1), 0, th - 1).astype(np.int32)
        depth[idx] = t[idx]
        color[idx] = self.tex[ty, tx].astype(np.float32) * self.shade


def render(cam_R, cam_t, K, planes, shape, noise=1.0, seed=0):
    """渲染成 (color, depth_z)。cam_t 为世界系下的 -R@eye。"""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pix = np.stack([xs, ys, np.ones_like(xs)], -1)
    cam_dirs = pix @ np.linalg.inv(K).T
    world_dirs = cam_dirs @ cam_R
    world_dirs /= np.linalg.norm(world_dirs, axis=-1, keepdims=True)
    origin = (-np.asarray(cam_R).T @ np.asarray(cam_t)).T
    depth = np.full((h, w), np.inf, np.float32)
    color = np.zeros((h, w, 3), np.float32)
    for p in planes:
        p.render(origin, world_dirs, depth, color)
    color += np.random.default_rng(seed).normal(0, noise, color.shape)
    # 平面渲染返回的是沿射线的距离 t，视差/点云需要的是光轴方向深度 Z
    cos_z = world_dirs @ np.asarray(cam_R)[2]
    depth_z = np.where(np.isfinite(depth), depth * np.maximum(cos_z, 1e-6), np.inf)
    return np.clip(color, 0, 255).astype(np.uint8), depth_z


def build_scene(seed=7, base_z=6.0):
    """一个房间 + 三个立方体：深度从 ~1.5m 到 ~6m，构图稳定、纹理丰富。"""
    planes = []
    room_w, room_h = 4.0, 2.2
    planes.append(RectPlane([0, 0, base_z], [1, 0, 0], [0, 1, 0], room_w, room_h,
                            texture(seed + 1, (150, 150, 145)), [0, 0, -1]))
    planes.append(RectPlane([0, -room_h, base_z / 2], [1, 0, 0], [0, 0, 1], room_w, base_z / 2,
                            texture(seed + 2, (95, 90, 85)), [0, 1, 0]))
    planes.append(RectPlane([0, room_h, base_z / 2], [1, 0, 0], [0, 0, 1], room_w, base_z / 2,
                            texture(seed + 3, (175, 175, 180)), [0, -1, 0]))
    planes.append(RectPlane([-room_w, 0, base_z / 2], [0, 0, 1], [0, 1, 0], base_z / 2, room_h,
                            texture(seed + 4, (120, 130, 150)), [1, 0, 0]))
    planes.append(RectPlane([room_w, 0, base_z / 2], [0, 0, 1], [0, 1, 0], base_z / 2, room_h,
                            texture(seed + 5, (140, 120, 110)), [-1, 0, 0]))
    planes += box([-1.05, -1.45, 3.2], [1.1, 0.85, 0.9], seed + 6, (200, 70, 60))
    planes += box([0.55, -1.55, 1.9], [0.9, 0.65, 0.75], seed + 7, (60, 130, 200))
    planes += box([1.35, -1.35, 4.6], [0.8, 1.05, 0.8], seed + 8, (80, 170, 90))
    return planes


def intrinsics(w, h, f):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def write(path, img):
    import cv2

    cv2.imwrite(path, img)
    print("wrote", os.path.relpath(path, ROOT))


def gen_stereo():
    """严格配准的立体对：右相机沿 +X 平移基线 b，真值视差 d = f*b/z。"""
    out = os.path.join(ROOT, "立体视觉测距", "assets")
    w, h, f, b = 1280, 720, 700.0, 0.10
    K = intrinsics(w, h, f)
    planes = build_scene()
    R = rotation(yaw=0, pitch=-4)
    left, depth = render(R, np.zeros(3), K, planes, (h, w), seed=1)
    right, _ = render(R, np.array([-b, 0, 0]), K, planes, (h, w), seed=2)
    write(os.path.join(out, "stereo_left.png"), left)
    write(os.path.join(out, "stereo_right.png"), right)
    disp = np.where(np.isfinite(depth), f * b / np.maximum(depth, 1e-3), 0)
    write(os.path.join(out, "stereo_disp_gt.png"), np.clip(disp * 16, 0, 65535).astype(np.uint16))
    np.savetxt(os.path.join(out, "stereo_params.txt"), np.array([f, b]), fmt="%.6f",
               header="focal_px baseline_m")
    print("stereo: f=%g b=%g disp %.2f~%.2f px" % (f, b, disp[depth < np.inf].min(), disp.max()))


def gen_rgbd():
    """RGB-D：彩色图 + 16 位毫米深度图。"""
    out = os.path.join(ROOT, "深度相机点云", "assets")
    w, h, f = 848, 480, 600.0
    K = intrinsics(w, h, f)
    R = rotation(yaw=4, pitch=-3)
    color, depth = render(R, np.zeros(3), K, build_scene(seed=11), (h, w), seed=3)
    write(os.path.join(out, "depth_rgb.png"), color)
    depth_mm = np.where(np.isfinite(depth), depth * 1000, 0).astype(np.uint16)
    write(os.path.join(out, "depth_map.png"), depth_mm)
    np.savetxt(os.path.join(out, "depth_intrinsics.txt"),
               np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]), fmt="%.3f", header="fx fy cx cy")
    print("rgbd: %d~%d mm" % (depth_mm[depth_mm > 0].min(), depth_mm.max()))


def build_object_scene(seed=31):
    """多视角专用场景：抬高的台面 + 多个高低错落的立方体，刻意不放大面积平面。

    近平面场景会让本质矩阵退化（单应解释更好），所以这里让特征点主要落在
    不同深度、不同朝向的立方体表面上。
    """
    planes = []
    planes.append(RectPlane([0, -1.0, 2.5], [1, 0, 0], [0, 0, 1], 900, 900,
                            texture(seed, (70, 70, 75)), [0, 1, 0]))
    layout = [(-1.5, 0.25, 2.2, 0.7), (-0.7, 0.18, 3.0, 0.55), (0.1, 0.30, 1.9, 0.8),
              (0.9, 0.20, 2.9, 0.6), (1.7, 0.15, 3.6, 0.5), (-1.9, 0.12, 3.8, 0.45),
              (-0.1, 0.10, 4.4, 0.5), (1.3, 0.35, 2.2, 0.45), (2.1, 0.10, 4.6, 0.6),
              (-1.1, 0.08, 5.0, 0.5)]
    for i, (x, y, z, sz) in enumerate(layout):
        planes += box([x, -1.0 + sz, z], [sz * 1.6, sz * 1.4, sz * 1.6], seed + i + 1,
                      (60 + 18 * i % 180, 90 + 25 * i % 160, 200 - 15 * i % 170))
    return planes


def gen_multiview():
    """环绕拍摄：相机在半径 r 的圆弧上绕场景中心旋转，全部看向同一中心。"""
    out = os.path.join(ROOT, "多视角三维重建", "assets")
    w, h, f, radius = 800, 600, 700.0, 2.6
    K = intrinsics(w, h, f)
    planes = build_object_scene(seed=21)
    center = np.array([0.0, -0.55, 3.2])
    poses = []
    for i in range(6):
        yaw = -35 + 14 * i
        ang = np.deg2rad(yaw)
        eye = center + radius * np.array([np.sin(ang), 0.35, np.cos(ang)])
        R, t = look_at(eye, center)   # 用 look_at 保证光轴真的指向场景中心
        img, _ = render(R, t, K, planes, (h, w), seed=10 + i)
        write(os.path.join(out, "view_%02d.png" % (i + 1)), img)
        poses.append(np.hstack([eye, R.ravel()]))
    np.savetxt(os.path.join(out, "cameras_gt.txt"), np.array(poses), fmt="%.5f",
               header="eye_x eye_y eye_z R00..R22")
    np.savetxt(os.path.join(out, "intrinsics_gt.txt"),
               np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]), fmt="%.3f", header="fx fy cx cy")
    print("multiview: 6 views, radius %g" % radius)


def gen_superres():
    """从高清参考图降采样出 4 倍低分辨率输入，保证超分评价有意义。"""
    import cv2

    out = os.path.join(ROOT, "图像超分辨率", "assets")
    ref = cv2.imread(os.path.join(out, "reference.png"))
    lr = cv2.resize(ref, (ref.shape[1] // 4, ref.shape[0] // 4), interpolation=cv2.INTER_AREA)
    write(os.path.join(out, "lowres.png"), lr)
    print("superres: ref %s -> lr %s" % (ref.shape[:2], lr.shape[:2]))


def gen_group_photo():
    """把现有人像拼到街景背景上，供多人脸检测演示（README 注明是拼贴合成）。"""
    import cv2

    out = os.path.join(ROOT, "DNN人脸检测", "assets")
    bg = cv2.imread(os.path.join(out, "faces.png"))
    canvas = cv2.resize(bg, (1280, 720)).astype(np.float32)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 15) * 0.45 + 55
    canvas = canvas.astype(np.uint8)
    faces = [
        (cv2.imread(os.path.join(out, "faces.png")), (356, 131, 336, 448), (40, 130, 250)),
        (cv2.imread(os.path.join(ROOT, "人脸识别", "assets", "probe.png")), (388, 168, 306, 411), (470, 110, 290)),
        (cv2.imread(os.path.join(ROOT, "实时姿态估计", "assets", "person.png")), (455, 45, 100, 125), (930, 210, 180)),
    ]
    for img, (x, y, bw, bh), (px, py, size) in faces:
        face = img[y : y + bh, x : x + bw]
        face = cv2.resize(face, (size, int(size * face.shape[0] / face.shape[1])))
        canvas[py : py + face.shape[0], px : px + face.shape[1]] = face
    write(os.path.join(out, "group_photo.png"), canvas)


def cv2_blur(img, k=31):
    import cv2
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def look_at(eye, target, up=(0, 1, 0)):
    """世界->相机的旋转：相机光轴指向 target，返回 R 与 t=-R@eye。"""
    eye = np.asarray(eye, np.float64)
    z = np.asarray(target, np.float64) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, np.float64)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def chessboard_texture(cols=9, rows=6, square=100, margin=70):
    """精确的棋盘格纹理：cols x rows 个内角点（即 (cols+1) x (rows+1) 个方格）。"""
    w = (cols + 1) * square + 2 * margin
    h = (rows + 1) * square + 2 * margin
    img = np.full((h, w, 3), 245, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = margin + r * square
                x0 = margin + c * square
                img[y0 : y0 + square, x0 : x0 + square] = 20
    return img


def aruco_board_texture(grid=3, marker_px=110, gap_px=32, margin=45):
    """3x3 ArUco 标定板纹理，标记 id 从 0 开始按行排列。"""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    span = grid * marker_px + (grid - 1) * gap_px
    w = h = span + 2 * margin
    board = np.full((h, w, 3), 250, np.uint8)
    for mid in range(grid * grid):
        row, col = divmod(mid, grid)
        marker = cv2.aruco.drawMarker(dictionary, mid, marker_px)
        y0 = margin + row * (marker_px + gap_px)
        x0 = margin + col * (marker_px + gap_px)
        board[y0 : y0 + marker_px, x0 : x0 + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return board


def render_board_views(out_dir, prefix, board_tex, w, h, f, board_mm, count=12, gaps=True):
    """把平面标定板渲染成 count 个不同位姿的视图，内参已知。"""
    K = intrinsics(w, h, f)
    bw, bh = board_tex.shape[1], board_tex.shape[0]
    board_mm_w = board_mm
    board_mm_h = board_mm * bh / bw
    # 背景用平滑渐变，噪点背景会让人误以为标定图很脏
    gh, gw = 700, 900
    yy = np.linspace(0, 1, gh)[:, None, None]
    backdrop = np.clip(70 + 90 * (1 - yy), 0, 255) * np.ones((1, gw, 3))
    backdrop = cv2_blur(np.dstack([backdrop[:, :, 0]]) if False else backdrop.astype(np.uint8))
    poses = []
    vs = [0.0, -28, 24, -16, 30, -22, 12, 26, -30, 18, -10, 8]
    ps = [0.0, 18, -20, 26, -12, 15, -26, 10, 20, -18, 24, -8]
    rs = [0.0, 12, -14, 8, 16, -18, 10, -8, 14, -12, 6, -16]
    dists = [520, 480, 560, 500, 540, 470, 580, 510, 490, 550, 530, 500]
    for i in range(count):
        yaw, pitch, roll = vs[i % len(vs)], ps[i % len(ps)], rs[i % len(rs)]
        d = dists[i % len(dists)]
        if not gaps:
            yaw *= 0.5
            pitch *= 0.5
        ang = np.deg2rad(yaw)
        eye = np.array([d * np.sin(ang) * 0.35, -d * np.sin(np.deg2rad(pitch)) * 0.3, d])
        R, t = look_at(eye, [0, 0, 0])
        Rroll = rotation(roll=roll)
        R2, t2 = look_at(Rroll.T @ eye, [0, 0, 0])
        planes = [
            RectPlane([0, 0, 0], [1, 0, 0], [0, -1, 0], board_mm_w / 2, board_mm_h / 2, board_tex, [0, 0, 1]),
            RectPlane([0, 0, -180], [1, 0, 0], [0, 1, 0], 900, 700, backdrop, [0, 0, 1]),
        ]
        img, _ = render(R2, t2, K, planes, (h, w), noise=1.5, seed=200 + i)
        write(os.path.join(out_dir, "%s_%02d.png" % (prefix, i + 1)), img)
        poses.append(np.hstack([eye, R2.ravel()]))
    np.savetxt(os.path.join(out_dir, "intrinsics_gt.txt"),
               np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]), fmt="%.3f", header="fx fy cx cy")
    print("%s: %d views, GT fx=%.1f fy=%.1f" % (prefix, count, K[0, 0], K[1, 1]))


def gen_calib_views():
    """相机标定素材：12 张几何精确的棋盘格视图（9x6 内角点，方格 25mm）。"""
    out = os.path.join(ROOT, "相机标定", "assets")
    board = chessboard_texture(9, 6, square=100, margin=70)
    # 纹理里 100 px 对应 25 mm，整块板宽按此换算成毫米
    render_board_views(out, "calib", board, 1024, 768, 900.0, board.shape[1] / 100.0 * 25.0)


def gen_aruco_views():
    """ArUco 标定素材：12 张 3x3 标定板视图（标记 40mm，间隔 12mm）。"""
    out = os.path.join(ROOT, "ArUco标定", "assets")
    board = aruco_board_texture(3, 110, 32, 45)
    mm_per_px = 40.0 / 110.0
    render_board_views(out, "board_view", board, 1024, 1024, 900.0, board.shape[1] * mm_per_px)


if __name__ == "__main__":
    gen_calib_views()
    gen_aruco_views()
    gen_stereo()
    gen_rgbd()
    gen_multiview()
    gen_superres()
    gen_group_photo()
