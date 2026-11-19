# -*- coding: utf-8 -*-
"""
@file    parallel_computing.py
@brief   并行计算演示：OpenCV 内置并行框架 + 自定义算子的分块并行，并做性能对比。
@details 流程：准备大图 -> 串行基线 -> 内置算子线程数扫描 -> 自定义算子线程/粒度扫描 ->
          正确性校验 -> 全局归约示例 -> 输出加速比表格。
@note    依赖：opencv-python（pip install opencv-python）、numpy。
        勘误：cv2.parallel_for_ 并不存在！cv::parallel_for_ 在 utility.hpp 里是
        CV_EXPORTS（非 CV_EXPORTS_W），且要求继承 cv::ParallelLoopBody 写 C++ 任务类，
        Python 绑定生成器无法导出；实测 OpenCV 4.6 / 5.0 的 cv2 均无此属性。
        Python 侧能用的两条路：
        (1) cv2.setNumThreads(n) 控制 OpenCV 内置算子的内部并行；
        (2) numpy 的 ufunc 会释放 GIL，配合 ThreadPoolExecutor 按行分块，
            这是 parallel_for_ 的 Python 等价实现。
        若循环体用纯 Python 逐像素计算，GIL 会让多线程反而更慢，务必向量化。
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# 自定义色调映射算子的参数。
GAMMA = 0.65
WEIGHT = 0.55

# 内置算子选一个计算量足够大的核，否则线程调度开销会吃掉收益。
BLUR_KERNEL = (31, 31)

# 默认分块行数：太小调度开销大，太大并行度不足，16~64 行通常合适。
DEFAULT_TILE_ROWS = 32

# 粒度扫描候选值；rows 那一档等价于「整图一个块」，即完全没有并行。
GRANULARITY_CANDIDATES = (1, 4, 16, 64)


def make_synthetic_image(rows, cols, seed=20261117):
    """
    @brief  生成确定性的合成大图，保证无外部素材也能运行。
    @param  rows 图像行数。
    @param  cols 图像列数。
    @param  seed 随机种子。
    @return 三通道 BGR 图像（横向渐变 + 纵向渐变 + 随机噪声）。
    """
    rng = np.random.default_rng(seed)
    image = np.empty((rows, cols, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, cols, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(0, 255, rows, dtype=np.uint8)[:, None]
    image[..., 2] = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    return image


def make_tiles(rows, tile_rows):
    """
    @brief  按行把 [0, rows) 切成互不重叠的块，等价于 C++ 的 cv::Range 集合。
    @param  rows      总行数。
    @param  tile_rows 每块行数。
    @return (start, end) 列表。
    """
    return [(y, min(y + tile_rows, rows)) for y in range(0, rows, tile_rows)]


def tone_map_tile(tile, gamma=GAMMA, weight=WEIGHT):
    """
    @brief  自定义逐像素色调映射算子，只作用于一个行块。
    @param  tile   输入行块（uint8 BGR）。
    @param  gamma  Gamma 校正指数。
    @param  weight 亮度权重系数。
    @return 处理后的行块（uint8 BGR）。
    @note   故意用 np.power / np.exp 让算子成为计算密集型：这些 ufunc 在 C 层执行
            且释放 GIL，多线程才有收益。若改成 cv2.LUT 查表，会退化成内存带宽密集型。
    """
    v = tile.astype(np.float32) / 255.0
    mapped = np.power(v, gamma)
    shading = 1.0 - weight + weight * np.exp(-v)
    return np.clip(255.0 * mapped * shading, 0, 255).astype(np.uint8)


def tone_map_serial(image, tile_rows):
    """
    @brief  串行基线：主线程顺序处理所有行块。
    @param  image     输入图像。
    @param  tile_rows 分块行数。
    @return 处理结果。
    """
    dst = np.empty_like(image)
    for start, end in make_tiles(image.shape[0], tile_rows):
        dst[start:end] = tone_map_tile(image[start:end])
    return dst


def tone_map_parallel(image, tile_rows, workers):
    """
    @brief  parallel_for_ 的 Python 等价实现：行块提交给线程池，主线程回填结果。
    @param  image     输入图像。
    @param  tile_rows 分块行数（相当于 nstripes = ceil(rows / tile_rows)）。
    @param  workers   工作线程数。
    @return 处理结果。
    @note   每个任务写自己的局部数组，由主线程按序写入 dst 的不重叠区域，因此无数据竞争；
            耗时中包含线程池创建开销（约 0.1 ms）。
    """
    dst = np.empty_like(image)
    tiles = make_tiles(image.shape[0], tile_rows)

    def work(bounds):
        start, end = bounds
        return start, end, tone_map_tile(image[start:end])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start, end, out in pool.map(work, tiles):
            dst[start:end] = out
    return dst


def reduce_mean_parallel(image, tile_rows, workers):
    """
    @brief  跨块有依赖时的正确做法：每个块局部累加，主线程最后合并（无共享可变状态）。
    @param  image     输入图像。
    @param  tile_rows 分块行数。
    @param  workers   工作线程数。
    @return 全图均值。
    @note   直方图、统计量这类 reduction 不能直接就地写共享数组；也不要每像素加锁，
            应像本例一样每条 stripe 只做一次合并。
    """
    tiles = make_tiles(image.shape[0], tile_rows)

    def work(bounds):
        start, end = bounds
        part = image[start:end]
        return float(part.sum(dtype=np.float64)), part.size

    total, count = 0.0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for partial_sum, partial_count in pool.map(work, tiles):
            total += partial_sum
            count += partial_count
    return total / count


def benchmark_ms(fn, repeats=3):
    """
    @brief  计时：先预热一次，再取重复执行中的最小耗时。
    @param  fn      待测操作。
    @param  repeats 重复次数。
    @return 最小耗时（毫秒）。
    """
    fn()  # 预热：首次调用会触发线程池创建、内存分配等一次性开销。
    costs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        costs.append((time.perf_counter() - t0) * 1000.0)
    return min(costs)


def thread_counts(max_threads):
    """@brief 构造 1, 2, 4, ... , max_threads 的线程数序列。"""
    counts = []
    n = 1
    while n < max_threads:
        counts.append(n)
        n *= 2
    counts.append(max_threads)
    return counts


def print_table(title, records, baseline_ms):
    """
    @brief  输出耗时/加速比表格。
    @param  title       表标题。
    @param  records     (模式, 线程数, 分块行数, 耗时) 列表。
    @param  baseline_ms 串行基线耗时。
    """
    print("\n" + title)
    print("{:<34}{:>9}{:>12}{:>14}{:>11}".format(
        "mode", "threads", "tile rows", "time(ms)", "speedup"))
    print("-" * 80)
    for label, threads, tile_rows, ms in records:
        print("{:<34}{:>9}{:>12}{:>14.2f}{:>10.2f}x".format(
            label, threads, str(tile_rows), ms, baseline_ms / ms))


def main():
    """@brief 主流程：解析参数、准备图像、跑两组对比并打印表格。"""
    parser = argparse.ArgumentParser(
        description="并行计算演示：OpenCV 并行框架与自定义算子分块加速的性能对比"
    )
    parser.add_argument("-i", "--input", default=None, help="输入图像路径，缺省生成合成大图")
    parser.add_argument("-r", "--rows", type=int, default=1200, help="合成图像行数")
    parser.add_argument("-c", "--cols", type=int, default=1600, help="合成图像列数")
    parser.add_argument(
        "-t", "--threads", type=int, default=0, help="最大线程数，0=逻辑CPU数"
    )
    parser.add_argument("-p", "--repeat", type=int, default=3, help="每项计时重复次数(取最小值)")
    parser.add_argument("-o", "--output", default=None, help="保存色调映射结果图像路径")
    args = parser.parse_args()

    repeats = max(1, args.repeat)
    max_threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    counts = thread_counts(max_threads)

    # 步骤 1：准备大图（大图像 / 批量数据才值得开线程）。
    if args.input:
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("无法读取图像: {}".format(os.path.abspath(args.input)))
    else:
        if args.rows <= 0 or args.cols <= 0:
            raise ValueError("rows / cols 必须为正数")
        image = make_synthetic_image(args.rows, args.cols)

    print("cv2 {} | numpy {} | logical CPUs: {}".format(
        cv2.__version__, np.__version__, cv2.getNumberOfCPUs()))
    print("image: {}x{} x {}ch | repeats: {}".format(
        image.shape[1], image.shape[0], image.shape[2], repeats))
    print("cv2.parallel_for_ available: {} (OpenCV Python 绑定未导出该 C++ 接口)".format(
        hasattr(cv2, "parallel_for_")))

    # ---------------- A：OpenCV 内置算子的并行框架 ----------------
    # cv2.setNumThreads(n) 是全局设置，只影响 OpenCV 内部的 parallel_for_，
    # 对 numpy 与 ThreadPoolExecutor 无效；用完必须恢复。
    original_threads = cv2.getNumThreads()
    records_a = []
    try:
        cv2.setNumThreads(1)
        baseline_a = benchmark_ms(
            lambda: cv2.GaussianBlur(image, BLUR_KERNEL, 0), repeats
        )
        records_a.append(("GaussianBlur (1 thread)", 1, "-", baseline_a))
        for n in counts:
            if n == 1:
                continue
            cv2.setNumThreads(n)
            ms = benchmark_ms(lambda: cv2.GaussianBlur(image, BLUR_KERNEL, 0), repeats)
            records_a.append(("GaussianBlur (auto stripes)", n, "-", ms))
        print_table("A. OpenCV 内置并行框架：cv2.setNumThreads(n) + cv2.GaussianBlur",
                    records_a, baseline_a)
    finally:
        cv2.setNumThreads(original_threads)

    # ---------------- B：自定义算子的分块并行 ----------------
    tile_rows = min(DEFAULT_TILE_ROWS, image.shape[0])
    serial = tone_map_serial(image, tile_rows)
    baseline_b = benchmark_ms(lambda: tone_map_serial(image, tile_rows), repeats)
    records_b = [("serial (plain tile loop)", 1, tile_rows, baseline_b)]

    # 1 个 worker 的一档用于量出线程池本身的调度开销。
    records_b.append(("thread pool (1 worker)", 1, tile_rows,
                      benchmark_ms(lambda: tone_map_parallel(image, tile_rows, 1), repeats)))
    for n in counts:
        if n == 1:
            continue
        ms = benchmark_ms(lambda: tone_map_parallel(image, tile_rows, n), repeats)
        records_b.append(("thread pool (thread sweep)", n, tile_rows, ms))

    # 粒度扫描：固定线程数，只改变分块行数。
    candidates = sorted({min(t, image.shape[0]) for t in GRANULARITY_CANDIDATES})
    candidates.append(image.shape[0])  # 整图一块 = 没有并行度
    for tile in candidates:
        ms = benchmark_ms(
            lambda t=tile: tone_map_parallel(image, t, max_threads), repeats
        )
        records_b.append(("thread pool (granularity)", max_threads, tile, ms))

    # 步骤 3：正确性校验。块间无依赖时，并行结果必须与串行逐字节一致。
    parallel = tone_map_parallel(image, tile_rows, max_threads)
    equal = np.array_equal(serial, parallel)
    max_diff = int(np.abs(serial.astype(np.int16) - parallel.astype(np.int16)).max())
    print("\ncorrectness: array_equal(serial, parallel) = {} | max diff = {} ({})".format(
        equal, max_diff, "PASS" if equal else "FAIL"))
    if not equal:
        raise RuntimeError("并行结果与串行不一致，请检查分块边界")

    print_table("B. 自定义算子：串行 vs 线程池分块（parallel_for_ 的 Python 等价实现）",
                records_b, baseline_b)

    # ---------------- C：跨块有依赖时的归约 ----------------
    mean_ref = float(np.mean(image))
    mean_par = reduce_mean_parallel(image, tile_rows, max_threads)
    print("\nreduction: stripe-local sum + merge = {:.8f} | numpy mean = {:.8f}"
          " | diff = {:.8f}".format(mean_par, mean_ref, abs(mean_par - mean_ref)))
    print("note: 积分图/形态学/连通域等有跨块依赖，不能直接切块；\n"
          "      cv2.setNumThreads() 不影响 numpy，纯 Python 逐像素循环受 GIL 限制。")

    if args.output:
        cv2.imwrite(args.output, parallel)
        print("saved: {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
