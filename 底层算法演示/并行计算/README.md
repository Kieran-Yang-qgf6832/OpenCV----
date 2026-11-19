# 并行计算

## 需求
使用OpenCV并行框架加速处理

## 核心函数
C++：`cv::parallel_for_()`、`cv::setNumThreads()`、`cv::getNumberOfCPUs()`
Python：`cv2.setNumThreads()`、`concurrent.futures.ThreadPoolExecutor`


## 关键步骤
1. 定义并行任务：C++ 继承 `cv::ParallelLoopBody`（或用 lambda 重载），Python 定义块处理函数
2. 设置迭代范围：`cv::Range(0, rows)`，按行切块，保证块间无数据依赖
3. 并行执行：`parallel_for_(range, body, nstripes)`，`nstripes` 控制分块粒度，`-1` 为自动
4. 性能对比：串行基线 → 线程数扫描 → 分块粒度扫描 → 耗时/加速比表格
5. 正确性校验：并行结果与串行逐字节一致；有全局累加时用「局部缓冲 + 合并」

## 输入
大图像或批量数据（缺省由程序内生成 1600x1200 合成图，也可用 `-i` 指定图片）

## 输出
加速处理结果（可选 `-o` 保存结果图）与耗时/加速比表格

## 使用情景
- **自定义逐像素算子**：内置函数覆盖不到的 Gamma/色调映射、逐像素条件判断、查表组合
- **图像金字塔 / 分块处理**：多尺度、多 ROI、多 tile 上重复同一套计算
- **批量图片预处理**：N 张图做同样的纯 CPU 计算（注意别和下游推理的线程池重复开线程）
- **点云 / 关键点集合**：描述子计算、匹配打分、重投影误差等元素间独立的运算
- **多相机 / 多参数任务**：左右图、多视角、参数网格搜索

## 不适用
- 有跨块依赖的算子：积分图、形态学、连通域标记、扫描线递推
- 循环体太轻或图太小：线程调度开销大于收益，内存带宽瓶颈的纯拷贝/简单加减也一样
- 下游已自带线程池：`dnn` 推理、视频解码等，再套一层会造成线程过订阅
- 需要全局累加时直接写共享数组：应改成本演示的局部累加 + 一次合并

## 运行
```bash
# C++（必须 Release 构建，否则 -O0 下加速比无意义）
cd C++ && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/parallel_computing --rows=1200 --cols=1600 --threads=8 --repeat=3
# 参数需用 --key=value 形式：OpenCV CommandLineParser 不支持空格分隔

# Python（pip install opencv-python numpy）
cd Python && python parallel_computing.py --rows 1200 --cols 1600 --threads 8 --repeat 3
```

两份程序都支持 `-i/--input` 换成真实图片、`-o/--output` 保存结果、`-t/--threads` 限制线程数。

## 素材
`assets/`：`street.png`
