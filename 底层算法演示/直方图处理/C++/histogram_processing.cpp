/**
 * @file    histogram_processing.cpp
 * @brief   直方图处理演示：直方图统计、全局均衡化与 CLAHE 自适应增强的对比。
 * @details 流程：计算直方图 -> 全局均衡化 -> 自适应 CLAHE -> 对比效果。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          全局均衡化按整幅图的累积分布重映射灰度，会同时放大噪声；
 *          CLAHE 只在局部小块内做均衡并裁剪直方图峰值，因此更适合雾霾、逆光这类光照不均的图像。
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

const int kPanelWidth = 480;
const int kPanelHeight = 360;
const int kTitleHeight = 30;

/** @brief 在若干候选位置中查找素材，便于从不同工作目录运行。 */
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

/** @brief 由素材路径推导输出目录：<demo>/assets/x.png -> <demo>/output。 */
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

/** @brief 等比缩放并居中留白到固定面板。 */
cv::Mat fitPanel(const cv::Mat& image, const std::string& title)
{
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }
    const double scale =
        std::min(kPanelWidth / static_cast<double>(bgr.cols),
                 kPanelHeight / static_cast<double>(bgr.rows));
    cv::Mat scaled;
    cv::resize(bgr, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::Mat panel(kPanelHeight + kTitleHeight, kPanelWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    const int x = (kPanelWidth - scaled.cols) / 2;
    const int y = kTitleHeight + (kPanelHeight - scaled.rows) / 2;
    scaled.copyTo(panel(cv::Rect(x, y, scaled.cols, scaled.rows)));
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/** @brief 按列数拼接等尺寸面板。 */
cv::Mat montage(const std::vector<cv::Mat>& panels, int cols)
{
    const int rows = static_cast<int>((panels.size() + cols - 1) / cols);
    cv::Mat canvas(rows * panels[0].rows, cols * panels[0].cols, CV_8UC3, cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < panels.size(); ++i) {
        const int r = static_cast<int>(i) / cols;
        const int c = static_cast<int>(i) % cols;
        panels[i].copyTo(canvas(cv::Rect(c * panels[i].cols, r * panels[i].rows,
                                         panels[i].cols, panels[i].rows)));
    }
    return canvas;
}

/**
 * @brief  把 256 级直方图画成柱状图，便于直观对比分布变化。
 * @param  gray   灰度图。
 * @param  width  输出宽度。
 * @param  height 输出高度。
 * @return 柱状图图像（BGR）。
 */
cv::Mat drawHistogram(const cv::Mat& gray, int width, int height)
{
    const int histSize = 256;
    float range[] = {0.0f, 256.0f};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    double maxValue = 0.0;
    cv::minMaxLoc(hist, nullptr, &maxValue);
    hist.convertTo(hist, CV_32F, (height - 10) / std::max(1.0, maxValue));

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    const int binWidth = std::max(1, width / histSize);
    for (int i = 0; i < histSize; ++i) {
        const int value = static_cast<int>(hist.at<float>(i));
        cv::rectangle(canvas, cv::Rect(i * binWidth, height - value, binWidth, value),
                      cv::Scalar(80, 80, 80), cv::FILLED);
    }
    return canvas;
}

}  // namespace

/**
 * @brief  主流程：直方图、均衡化与 CLAHE 对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/haze.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/histogram_processing.png }"
        "{l clipLimit    | 2.0  | CLAHE 的对比度限制 }"
        "{t tileSize     | 8    | CLAHE 的网格数 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("直方图处理演示: calcHist / equalizeHist / CLAHE\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("haze.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 1：计算直方图并记录原始分布统计。
    cv::Scalar meanOriginal;
    cv::Scalar stdOriginal;
    cv::meanStdDev(gray, meanOriginal, stdOriginal);
    const cv::Mat histOriginal = drawHistogram(gray, 512, kPanelHeight);

    // 步骤 2：全局均衡化。
    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);
    cv::Scalar meanEq;
    cv::Scalar stdEq;
    cv::meanStdDev(equalized, meanEq, stdEq);
    const cv::Mat histEqualized = drawHistogram(equalized, 512, kPanelHeight);

    // 步骤 3：自适应 CLAHE。按小块分别均衡，避免全局拉伸放大噪声。
    const int tile = std::max(1, parser.get<int>("tileSize"));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(parser.get<double>("clipLimit"),
                                               cv::Size(tile, tile));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    cv::Scalar meanClahe;
    cv::Scalar stdClahe;
    cv::meanStdDev(enhanced, meanClahe, stdClahe);
    const cv::Mat histClahe = drawHistogram(enhanced, 512, kPanelHeight);

    // 步骤 4：对比效果，标准差越大说明对比度被拉得越开。
    std::cout << "输入: " << inputPath << "\n";
    std::cout << std::left << std::setw(12) << "method" << std::setw(10) << "mean"
              << "std\n";
    const std::pair<std::string, std::pair<double, double>> stats[] = {
        {"original", {meanOriginal[0], stdOriginal[0]}},
        {"equalized", {meanEq[0], stdEq[0]}},
        {"clahe", {meanClahe[0], stdClahe[0]}}};
    for (const auto& item : stats) {
        std::cout << std::left << std::setw(12) << item.first << std::fixed
                  << std::setprecision(1) << std::setw(10) << item.second.first
                  << item.second.second << "\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(histOriginal, "histogram"),
        fitPanel(equalized, "equalizeHist"),
        fitPanel(histEqualized, "histogram (eq)"),
        fitPanel(enhanced, "CLAHE"),
        fitPanel(histClahe, "histogram (clahe)")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/histogram_processing.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("histogram processing", result);
        cv::waitKey(0);
    }
    return 0;
}
