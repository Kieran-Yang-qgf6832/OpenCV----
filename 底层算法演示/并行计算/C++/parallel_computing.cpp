/**
 * @file    parallel_computing.cpp
 * @brief   并行计算演示：用 cv::parallel_for_ 分块加速自定义逐像素算子，并与串行做性能对比。
 * @details 流程：准备大图 -> 定义并行任务(ParallelLoopBody / lambda) -> 串行基线 ->
 *          正确性校验 -> 线程数扫描 -> 分块粒度扫描 -> 全局归约示例 -> 输出加速比表格。
 * @note    依赖：OpenCV 4.x（core / imgproc），并行后端由编译时的 TBB / OpenMP / pthreads 决定。
 *          cv::parallel_for_ 在 utility.hpp 中声明为 CV_EXPORTS（非 CV_EXPORTS_W），
 *          且要求继承 cv::ParallelLoopBody 编写 C++ 任务类，Python 绑定并未导出，
 *          所以 Python 版演示改用 cv2.setNumThreads() + 线程池分块，见 Python/parallel_computing.py。
 *          计时务必使用 Release 构建，否则 -O0 下的加速比没有参考意义。
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

/// Gamma 校正指数。
constexpr float kGamma = 0.65f;
/// 亮度权重系数，用于让算子需要额外的 exp 计算。
constexpr float kWeight = 0.55f;

/// 一条性能记录。
struct BenchRecord {
    std::string label;     ///< 模式名称（表格用 ASCII，避免中文宽度导致错位）。
    int threads;           ///< cv::setNumThreads() 设置的线程数。
    int tileRows;          ///< 等效分块行数，0 表示由 OpenCV 自动决定。
    double milliseconds;   ///< 多次计时中的最小值。
};

/**
 * @brief  生成确定性的合成大图，保证不依赖外部素材也能运行。
 * @param  rows 图像行数。
 * @param  cols 图像列数。
 * @return 三通道 BGR 图像（横向渐变 + 纵向渐变 + 随机噪声）。
 */
cv::Mat makeSyntheticImage(int rows, int cols)
{
    cv::Mat image(rows, cols, CV_8UC3);
    cv::RNG rng(20261117);
    for (int y = 0; y < rows; ++y) {
        uchar* row = image.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            row[3 * x + 0] = static_cast<uchar>((x * 255) / std::max(1, cols - 1));
            row[3 * x + 1] = static_cast<uchar>((y * 255) / std::max(1, rows - 1));
            row[3 * x + 2] = static_cast<uchar>(rng.uniform(0, 256));
        }
    }
    return image;
}

/**
 * @brief  自定义逐像素色调映射算子，只处理 [rows.start, rows.end) 这段行。
 * @param  src   输入 BGR 图像。
 * @param  dst   输出 BGR 图像（尺寸、类型与 src 一致，需预先分配）。
 * @param  rows  本次要处理的行区间。
 * @note   串行与并行共用这一个核函数，才能确保两种结果逐字节一致。
 *         这里故意保留 pow/exp，让算子成为计算密集型；若改写成 256 项查表(LUT)，
 *         算子会退化成内存带宽密集型，多线程收益会明显缩水。
 */
void toneMapRows(const cv::Mat& src, cv::Mat& dst, const cv::Range& rows)
{
    const int length = src.cols * src.channels();
    for (int y = rows.start; y < rows.end; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int i = 0; i < length; ++i) {
            const float v = s[i] / 255.0f;
            const float mapped = std::pow(v, kGamma);
            const float shading = 1.0f - kWeight + kWeight * std::exp(-v);
            d[i] = cv::saturate_cast<uchar>(255.0f * mapped * shading);
        }
    }
}

/**
 * @brief  cv::parallel_for_ 的任务体：把分配到的行区间交给同一个核函数。
 * @note   operator() 是 const，OpenCV 会在多个工作线程里同时调用同一个对象，
 *         因此任务体不能持有跨线程共享的可变状态；输出按行写入互不重叠的地址即可。
 */
class ToneMapInvoker : public cv::ParallelLoopBody
{
public:
    ToneMapInvoker(const cv::Mat& src, cv::Mat& dst) : src_(src), dst_(dst) {}

    void operator()(const cv::Range& range) const override
    {
        toneMapRows(src_, dst_, range);
    }

private:
    const cv::Mat& src_;
    cv::Mat& dst_;
};

/** @brief 串行基线：直接遍历全部行，完全不经过并行框架。 */
void toneMapSerial(const cv::Mat& src, cv::Mat& dst)
{
    toneMapRows(src, dst, cv::Range(0, src.rows));
}

/**
 * @brief  并行版本（继承 ParallelLoopBody 的经典写法）。
 * @param  nstripes 期望的 stripe 数量；-1 表示由 OpenCV 按线程数自动决定。
 *                  设为 ceil(rows / tileRows) 即可间接控制分块行数。
 */
void toneMapParallel(const cv::Mat& src, cv::Mat& dst, double nstripes)
{
    // 步骤 1~3：定义任务 -> 设置迭代范围 Range(0, rows) -> 并行执行。
    cv::parallel_for_(cv::Range(0, src.rows), ToneMapInvoker(src, dst), nstripes);
}

/**
 * @brief  并行版本（OpenCV 4.x 提供的 std::function / lambda 重载，写法更紧凑）。
 * @param  nstripes 同上。
 */
void toneMapParallelLambda(const cv::Mat& src, cv::Mat& dst, double nstripes)
{
    cv::parallel_for_(cv::Range(0, src.rows),
                      [&](const cv::Range& range) { toneMapRows(src, dst, range); },
                      nstripes);
}

/**
 * @brief  跨块有依赖时的归约容器：局部累加 + 一次性合并。
 * @note   绝不能在每个像素上加锁；每条 stripe 先在自己的栈上求和，最后加锁一次。
 */
class MeanAccumulator
{
public:
    void add(double partialSum, std::size_t partialCount)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sum_ += partialSum;
        count_ += partialCount;
    }

    double mean() const { return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_); }

private:
    std::mutex mutex_;
    double sum_ = 0.0;
    std::size_t count_ = 0;
};

/** @brief 计算图像全局均值的并行任务体，演示 reduction 的正确写法。 */
class MeanInvoker : public cv::ParallelLoopBody
{
public:
    MeanInvoker(const cv::Mat& src, MeanAccumulator& acc) : src_(src), acc_(acc) {}

    void operator()(const cv::Range& range) const override
    {
        double localSum = 0.0;
        std::size_t localCount = 0;
        const std::size_t length = static_cast<std::size_t>(src_.cols) * src_.channels();
        for (int y = range.start; y < range.end; ++y) {
            const uchar* s = src_.ptr<uchar>(y);
            for (std::size_t i = 0; i < length; ++i) {
                localSum += s[i];
                ++localCount;
            }
        }
        acc_.add(localSum, localCount);
    }

private:
    const cv::Mat& src_;
    MeanAccumulator& acc_;
};

/**
 * @brief  计时：先预热一次，再取重复执行中的最小耗时。
 * @param  fn      待测操作。
 * @param  repeats 重复次数。
 * @return 最小耗时（毫秒）。
 */
double benchmarkMs(const std::function<void()>& fn, int repeats)
{
    fn();  // 预热：首次调用会触发线程池创建、缺页等一次性开销。
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < repeats; ++i) {
        const int64 t0 = cv::getTickCount();
        fn();
        const int64 t1 = cv::getTickCount();
        best = std::min(best, static_cast<double>(t1 - t0) * 1000.0 / cv::getTickFrequency());
    }
    return best;
}

/** @brief 构造 1, 2, 4, ... , maxThreads 的线程数序列。 */
std::vector<int> makeThreadCounts(int maxThreads)
{
    std::vector<int> counts;
    for (int n = 1; n < maxThreads; n *= 2) {
        counts.push_back(n);
    }
    counts.push_back(maxThreads);
    return counts;
}

/**
 * @brief  输出耗时/加速比表格。
 * @param  records    所有性能记录。
 * @param  baselineMs 串行基线耗时。
 */
void printTable(const std::vector<BenchRecord>& records, double baselineMs)
{
    std::cout << std::left << std::setw(32) << "mode"
              << std::right << std::setw(8) << "threads"
              << std::setw(12) << "tile rows" << std::setw(14) << "time(ms)"
              << std::setw(12) << "speedup" << "\n";
    std::cout << std::string(78, '-') << "\n";
    for (const auto& r : records) {
        std::cout << std::left << std::setw(32) << r.label
                  << std::right << std::setw(8) << r.threads
                  << std::setw(12) << (r.tileRows > 0 ? std::to_string(r.tileRows) : "auto")
                  << std::fixed << std::setprecision(2) << std::setw(14) << r.milliseconds
                  << std::setw(11) << (baselineMs / r.milliseconds) << "x\n";
    }
}

/** @brief 从 cv::getBuildInformation() 中读出实际启用的并行后端名称。 */
std::string parallelFrameworkName()
{
    const std::string marker = "Parallel framework:";
    std::istringstream stream(cv::getBuildInformation());
    std::string line;
    while (std::getline(stream, line)) {
        const std::size_t pos = line.find(marker);
        if (pos == std::string::npos) {
            continue;
        }
        const std::string value = line.substr(pos + marker.size());
        const std::size_t first = value.find_first_not_of(" \t");
        const std::size_t last = value.find_last_not_of(" \t\r");
        if (first == std::string::npos) {
            break;
        }
        return value.substr(first, last - first + 1);
    }
    return "unknown";
}

}  // namespace

/**
 * @brief  主流程：解析参数、准备图像、跑完三组对比并打印表格。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    // 注意：OpenCV 4.6 的 CommandLineParser 只接受 --key=value / -k=value，
    // 写成 --rows 600 会把 600 当成位置参数而忽略，因此这里不使用 @位置参数。
    const cv::String keys =
        "{help h usage ? |        | 打印帮助信息 }"
        "{i input        |        | 输入图像路径, 缺省则程序内生成合成大图 }"
        "{r rows         | 1200   | 合成图像行数 }"
        "{c cols         | 1600   | 合成图像列数 }"
        "{t threads      | 0      | 最大线程数, 0=逻辑CPU数 }"
        "{p repeat       | 3      | 每项计时重复次数(取最小值) }"
        "{o output       |        | 保存色调映射结果图像路径 }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("并行计算演示: cv::parallel_for_ 分块加速自定义逐像素算子\n"
                 "参数请使用 --key=value 形式，例如 --rows=600 --threads=8");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    const std::string inputPath = parser.get<std::string>("input");
    const int rows = parser.get<int>("rows");
    const int cols = parser.get<int>("cols");
    const int repeats = std::max(1, parser.get<int>("repeat"));
    const int maxThreads = parser.get<int>("threads") > 0
                               ? parser.get<int>("threads")
                               : cv::getNumberOfCPUs();

    // 步骤 1：准备大图（大图像 / 批量数据才值得开线程）。
    cv::Mat src;
    if (!inputPath.empty()) {
        src = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (src.empty()) {
            std::cerr << "无法读取图像: " << inputPath << std::endl;
            return -1;
        }
    } else {
        if (rows <= 0 || cols <= 0) {
            std::cerr << "rows / cols 必须为正数。若写成了 --rows 600，请改成 --rows=600"
                         "（OpenCV CommandLineParser 不支持空格分隔）" << std::endl;
            return -1;
        }
        src = makeSyntheticImage(rows, cols);
    }
    cv::Mat serial(src.size(), src.type());
    cv::Mat parallel(src.size(), src.type());

    const int originalThreads = cv::getNumThreads();
    std::cout << "OpenCV " << CV_VERSION << " | parallel framework: " << parallelFrameworkName()
              << " | logical CPUs: " << cv::getNumberOfCPUs() << "\n";
    std::cout << "image: " << src.cols << "x" << src.rows << " x " << src.channels()
              << "ch | repeats: " << repeats << "\n\n";

    // 步骤 2：串行基线。
    cv::setNumThreads(1);
    const double baselineMs = benchmarkMs([&]() { toneMapSerial(src, serial); }, repeats);

    // 步骤 3：正确性校验。块间无依赖时，并行结果必须与串行逐字节相同。
    cv::setNumThreads(maxThreads);
    toneMapParallel(src, parallel, -1.0);
    const double maxDiff = cv::norm(serial, parallel, cv::NORM_INF);
    std::cout << "correctness: max|serial - parallel| = " << maxDiff
              << " (" << (maxDiff == 0.0 ? "PASS" : "FAIL") << ")\n";
    if (maxDiff != 0.0) {
        cv::setNumThreads(originalThreads);
        return -1;
    }

    // 步骤 4：性能对比 —— 线程数扫描 + 分块粒度扫描。
    std::vector<BenchRecord> records;
    records.push_back({"serial (plain loop)", 1, 0, baselineMs});

    for (const int n : makeThreadCounts(maxThreads)) {
        cv::setNumThreads(n);
        const std::string label = (n == 1) ? "parallel_for_ (1 thread)"
                                           : "parallel_for_ (auto stripes)";
        records.push_back({label, n, 0,
                           benchmarkMs([&]() { toneMapParallel(src, parallel, -1.0); }, repeats)});
    }

    cv::setNumThreads(maxThreads);
    records.push_back({"parallel_for_ (lambda form)", maxThreads, 0,
                       benchmarkMs([&]() { toneMapParallelLambda(src, parallel, -1.0); }, repeats)});

    // nstripes = ceil(rows / tileRows)：分块太细会被调度开销吃掉，太粗则无并行度。
    const int tileCandidates[] = {1, 4, 16, 64, src.rows};
    for (const int tile : tileCandidates) {
        const double nstripes = static_cast<double>((src.rows + tile - 1) / tile);
        records.push_back({"parallel_for_ (fixed stripes)", maxThreads, tile,
                           benchmarkMs([&]() { toneMapParallel(src, parallel, nstripes); }, repeats)});
    }

    std::cout << "\n";
    printTable(records, baselineMs);

    // 步骤 5：跨块有依赖（全局归约）的正确做法：局部累加，每条 stripe 只加锁一次。
    MeanAccumulator acc;
    cv::parallel_for_(cv::Range(0, src.rows), MeanInvoker(src, acc), -1.0);
    const cv::Scalar mean = cv::mean(src);
    const double reference = (mean[0] + mean[1] + mean[2]) / 3.0;
    std::cout << "\nreduction: stripe-local sum + mutex merge = " << std::setprecision(8)
              << acc.mean() << " | cv::mean = " << reference
              << " | diff = " << std::fabs(acc.mean() - reference) << "\n";
    std::cout << "note: 积分图/形态学/连通域等有跨块依赖，不能这样直接切块；\n"
                 "      直方图等需要全局累加时，必须按上面的局部缓冲 + 合并方式处理。\n";

    const std::string outputPath = parser.get<std::string>("output");
    if (!outputPath.empty()) {
        cv::imwrite(outputPath, parallel);
        std::cout << "saved: " << outputPath << "\n";
    }

    cv::setNumThreads(originalThreads);
    return 0;
}
