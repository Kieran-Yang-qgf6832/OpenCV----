/**
 * @file    image_thresholding.cpp
 * @brief   图像阈值化演示：全局阈值、Otsu 自动阈值与自适应阈值对比。
 * @details 流程：灰度转换 -> 固定全局阈值 -> Otsu 自动阈值 -> 自适应阈值(均值/高斯)
 *          -> 多阈值结果拼接对比，并统计各方法的前景像素占比。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          素材纸面上有明显光照不均：全局阈值（含 Otsu）会在暗部整片丢失，
 *          自适应阈值按局部邻域取阈值，因此能保留暗部文字——这是本演示的重点。
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

/// 拼接图中标题条的高度（像素）。
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

/**
 * @brief  给单通道二值图加标题条并转成 BGR，便于拼接。
 * @param  image 单通道图像。
 * @param  title 标题文字（仅支持 ASCII）。
 * @return 带标题条的面板。
 */
cv::Mat makePanel(const cv::Mat& image, const std::string& title)
{
    // 展示时缩到一半，控制拼接图体积。
    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::cvtColor(small, panel(cv::Rect(0, kTitleHeight, small.cols, small.rows)),
                 cv::COLOR_GRAY2BGR);
    cv::putText(panel, title, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return panel;
}

/**
 * @brief  按列数拼接等尺寸面板。
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
        panels[i].copyTo(canvas(cv::Rect(c * panels[i].cols, r * panels[i].rows,
                                         panels[i].cols, panels[i].rows)));
    }
    return canvas;
}

/**
 * @brief  打印二值图的前景像素占比。
 * @param  name 方法名称。
 * @param  mask 二值图。
 */
void printForegroundRatio(const std::string& name, const cv::Mat& mask)
{
    std::cout << "  " << std::left << std::setw(28) << name
              << std::fixed << std::setprecision(2)
              << 100.0 * cv::countNonZero(mask) / mask.total() << "%\n";
}

}  // namespace

/**
 * @brief  主流程：四种阈值策略对比并输出拼接图。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/paper.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_thresholding.png }"
        "{b block        | 31   | 自适应阈值的邻域尺寸，必须是大于 1 的奇数 }"
        "{c constant     | 5    | 自适应阈值的常数偏移 C }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像阈值化演示: 全局 / Otsu / 自适应阈值\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("paper.png");
    }

    // 步骤 1：灰度转换。
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int blockSize = parser.get<int>("block");
    if (blockSize < 3 || blockSize % 2 == 0) {
        std::cerr << "block 必须是大于 1 的奇数，已改用 31" << std::endl;
        blockSize = 31;
    }
    const double constant = parser.get<double>("constant");

    // 步骤 2：Otsu 全局阈值。阈值由灰度直方图自动确定，适用于双峰分布。
    cv::Mat otsu;
    const double otsuThreshold =
        cv::threshold(gray, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 步骤 3：固定全局阈值，作为对照。
    cv::Mat fixed;
    cv::threshold(gray, fixed, 127, 255, cv::THRESH_BINARY);

    // 步骤 4：自适应阈值。以每个像素邻域的加权均值减 C 作为局部阈值。
    cv::Mat adaptiveMean;
    cv::Mat adaptiveGaussian;
    cv::Mat adaptiveFine;
    cv::adaptiveThreshold(gray, adaptiveMean, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY, blockSize, constant);
    cv::adaptiveThreshold(gray, adaptiveGaussian, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, blockSize, constant);
    cv::adaptiveThreshold(gray, adaptiveFine, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 11, constant);

    std::cout << "输入: " << inputPath << ", Otsu 自动阈值 = " << otsuThreshold << "\n";
    std::cout << "前景(白)像素占比:\n";
    printForegroundRatio("fixed 127", fixed);
    printForegroundRatio("otsu", otsu);
    printForegroundRatio("adaptive mean (block=" + std::to_string(blockSize) + ")",
                         adaptiveMean);
    printForegroundRatio("adaptive gauss (block=" + std::to_string(blockSize) + ")",
                         adaptiveGaussian);
    printForegroundRatio("adaptive gauss (block=11)", adaptiveFine);

    const std::vector<cv::Mat> panels = {
        makePanel(gray, "gray input"),
        makePanel(fixed, "fixed 127"),
        makePanel(otsu, "otsu " + std::to_string(static_cast<int>(otsuThreshold))),
        makePanel(adaptiveMean, "adaptive mean"),
        makePanel(adaptiveGaussian, "adaptive gaussian"),
        makePanel(adaptiveFine, "adaptive gaussian b=11")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_thresholding.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image thresholding", result);
        cv::waitKey(0);
    }
    return 0;
}
