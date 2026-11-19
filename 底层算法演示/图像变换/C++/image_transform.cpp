/**
 * @file    image_transform.cpp
 * @brief   图像变换演示：用查找表实现对数变换、伽马变换与对比度拉伸。
 * @details 流程：读取灰度图 -> 构建 256 项查找表 -> 对数变换 / 伽马变换 / 对比度拉伸
 *          -> 拼接对比图并统计各变换后的灰度分布。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          灰度变换是逐像素的单调映射，用 LUT 只需算 256 次，比逐像素算 pow/log 快两个量级。
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

/// 拼接图中标题条的高度（像素）。
const int kTitleHeight = 30;

/**
 * @brief  在若干候选位置中查找素材，便于从不同工作目录运行。
 * @param  fileName 素材文件名。
 * @return 第一个存在的路径；都不存在时返回首选候选。
 */
std::string locateAsset(const std::string& fileName)
{
    const std::vector<std::string> candidates = {
        "../../assets/" + fileName, "../assets/" + fileName, "assets/" + fileName};
    for (const auto& path : candidates) {
        if (cv::utils::fs::exists(path)) {
            return path;
        }
    }
    return candidates.front();
}

/**
 * @brief  由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。
 * @param  inputPath 素材路径。
 * @return 输出目录路径（尚未创建）。
 */
std::string outputDirFor(const std::string& inputPath)
{
    std::string dir = inputPath;
    const size_t slash = dir.find_last_of("/\\");
    dir = (slash == std::string::npos) ? "." : dir.substr(0, slash);
    const size_t slash2 = dir.find_last_of("/\\");
    const std::string leaf = (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
    if (leaf == "assets") {
        dir = (slash2 == std::string::npos) ? "." : dir.substr(0, slash2);
    }
    return dir + "/output";
}

/**
 * @brief  构造对数变换查找表：s = c * ln(1 + r)，c 使 255 映射到 255。
 * @return 1x256 的 CV_8U 查找表。
 * @note   对数变换压缩高灰度、拉伸低灰度，适合整体偏暗、暗部细节丢失的图像。
 */
cv::Mat buildLogLut()
{
    cv::Mat lut(1, 256, CV_8U);
    const double c = 255.0 / std::log(1.0 + 255.0);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(c * std::log(1.0 + i));
    }
    return lut;
}

/**
 * @brief  构造伽马变换查找表：s = 255 * (r / 255) ^ gamma。
 * @param  gamma 伽马值；< 1 提亮暗部，> 1 压暗。
 * @return 1x256 的 CV_8U 查找表。
 */
cv::Mat buildGammaLut(double gamma)
{
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, gamma));
    }
    return lut;
}

/**
 * @brief  构造对比度拉伸查找表：把 [lo, hi] 线性映射到 [0, 255]。
 * @param  lo 输入图的实际最小灰度。
 * @param  hi 输入图的实际最大灰度。
 * @return 1x256 的 CV_8U 查找表。
 * @note   与 convertScaleAbs 不同，这里按图像实际动态范围拉伸，不会截断有效信息。
 */
cv::Mat buildStretchLut(double lo, double hi)
{
    cv::Mat lut(1, 256, CV_8U);
    const double span = (hi > lo) ? (hi - lo) : 1.0;
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>((i - lo) * 255.0 / span);
    }
    return lut;
}

/**
 * @brief  按百分位计算有效动态范围，避免极少数像素主导拉伸区间。
 * @param  gray    单通道图像。
 * @param  lowPct  低端百分位，如 0.01 表示 1%。
 * @param  highPct 高端百分位，如 0.99 表示 99%。
 * @param  lo      输出的低端灰度。
 * @param  hi      输出的高端灰度。
 * @note   直接取 min/max 时，一个过曝亮点就会让拉伸几乎失效；按百分位裁剪更稳。
 */
void percentileRange(const cv::Mat& gray, double lowPct, double highPct, double& lo, double& hi)
{
    const int histSize = 256;
    float range[] = {0.0f, 256.0f};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    const double total = static_cast<double>(gray.total());
    const double lowTarget = total * lowPct;
    const double highTarget = total * highPct;
    double accumulated = 0.0;
    bool lowFound = false;
    lo = 0.0;
    hi = 255.0;
    for (int i = 0; i < 256; ++i) {
        accumulated += hist.at<float>(i);
        if (!lowFound && accumulated >= lowTarget) {
            lo = i;
            lowFound = true;
        }
        if (accumulated >= highTarget) {
            hi = i;
            break;
        }
    }
    if (hi <= lo) {
        hi = lo + 1.0;
    }
}

/**
 * @brief  给单通道图加上标题条并转成 BGR，便于拼接展示。
 * @param  gray  单通道图像。
 * @param  title 标题文字（仅支持 ASCII）。
 * @return 带标题条的三通道面板。
 */
cv::Mat makePanel(const cv::Mat& gray, const std::string& title)
{
    // 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat body = panel(cv::Rect(0, kTitleHeight, small.cols, small.rows));
    cv::cvtColor(small, body, cv::COLOR_GRAY2BGR);
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/**
 * @brief  按列数把若干等尺寸面板拼成一张大图。
 * @param  panels 面板列表。
 * @param  cols   每行面板数。
 * @return 拼接结果。
 */
cv::Mat montage(const std::vector<cv::Mat>& panels, int cols)
{
    const int rows = static_cast<int>((panels.size() + cols - 1) / cols);
    cv::Mat canvas(rows * panels[0].rows, cols * panels[0].cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < panels.size(); ++i) {
        const int r = static_cast<int>(i) / cols;
        const int c = static_cast<int>(i) % cols;
        panels[i].copyTo(canvas(cv::Rect(c * panels[0].cols, r * panels[i].rows,
                                         panels[i].cols, panels[i].rows)));
    }
    return canvas;
}

/**
 * @brief  打印图像灰度统计，用于对比不同变换的效果。
 * @param  name 变换名称。
 * @param  img  单通道图像。
 */
void printStats(const std::string& name, const cv::Mat& img)
{
    double lo = 0.0;
    double hi = 0.0;
    cv::minMaxLoc(img, &lo, &hi);
    std::cout << "  " << std::left << std::setw(18) << name
              << " min=" << std::setw(4) << static_cast<int>(lo)
              << " max=" << std::setw(4) << static_cast<int>(hi)
              << " mean=" << std::fixed << std::setprecision(1) << cv::mean(img)[0] << "\n";
}

}  // namespace

/**
 * @brief  主流程：读取灰度图、构建查找表、逐项变换并拼接输出。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/portrait.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_transform.png }"
        "{g gamma        | 0.4  | 伽马变换的主参数 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像变换演示: LUT 实现对数 / 伽马 / 对比度拉伸\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string inputPath = parser.get<std::string>("input");
    if (inputPath.empty()) {
        inputPath = locateAsset("portrait.png");
    }

    // 步骤 1：读取并转为灰度图。
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    std::cout << "输入: " << inputPath << " (" << gray.cols << "x" << gray.rows << ")\n";
    printStats("original", gray);

    // 步骤 2：构建查找表。三种变换都是 256 项单调映射，先算表再一次 LUT。
    double lo = 0.0;
    double hi = 0.0;
    percentileRange(gray, 0.01, 0.99, lo, hi);  // 按 1%~99% 裁剪后的有效动态范围
    std::cout << "拉伸区间: [" << lo << ", " << hi << "]\n";
    const double mainGamma = parser.get<double>("gamma");

    // 步骤 3：逐项变换。
    cv::Mat logResult;
    cv::Mat gammaLow;
    cv::Mat gammaOne;
    cv::Mat gammaHigh;
    cv::Mat stretched;
    cv::LUT(gray, buildLogLut(), logResult);
    cv::LUT(gray, buildGammaLut(mainGamma), gammaLow);
    cv::LUT(gray, buildGammaLut(1.0), gammaOne);
    cv::LUT(gray, buildGammaLut(1.0 / mainGamma), gammaHigh);
    cv::LUT(gray, buildStretchLut(lo, hi), stretched);

    std::cout << "变换后统计:\n";
    printStats("log", logResult);
    printStats("gamma " + std::to_string(mainGamma).substr(0, 3), gammaLow);
    printStats("gamma 1.0", gammaOne);
    printStats("gamma " + std::to_string(1.0 / mainGamma).substr(0, 4), gammaHigh);
    printStats("stretch", stretched);

    // 步骤 4：拼接对比图并保存。
    const std::vector<cv::Mat> panels = {
        makePanel(gray, "original"),
        makePanel(logResult, "log"),
        makePanel(gammaLow, "gamma " + std::to_string(mainGamma).substr(0, 3)),
        makePanel(gammaOne, "gamma 1.0 (identity)"),
        makePanel(gammaHigh, "gamma " + std::to_string(1.0 / mainGamma).substr(0, 4)),
        makePanel(stretched, "stretch 1%-99%")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image transform", result);
        cv::waitKey(0);
    }
    return 0;
}
