/**
 * @file    morphology.cpp
 * @brief   形态学操作演示：腐蚀、膨胀、开闭运算与形态学梯度。
 * @details 流程：定义结构元素 -> 腐蚀去噪 -> 膨胀填充 -> 开闭运算组合 -> 输出对比。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          腐蚀 = 取邻域最小值（前景收缩），膨胀 = 取邻域最大值（前景扩张）；
 *          开运算 = 先腐蚀后膨胀（去小噪点），闭运算 = 先膨胀后腐蚀（填小孔洞）。
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

/** @brief 打印前景（非零）像素数，量化各操作对前景面积的影响。 */
void printForeground(const std::string& name, const cv::Mat& binary)
{
    std::cout << "  " << std::left << std::setw(12) << name << cv::countNonZero(binary) << "\n";
}

}  // namespace

/**
 * @brief  主流程：六种形态学操作的对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/morphology.png }"
        "{k kernel       | 5    | 结构元素尺寸（奇数） }"
        "{s shape        | ellipse | 结构元素形状: rect / ellipse / cross }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("形态学操作演示: erode / dilate / open / close / gradient / tophat\n"
                 "参数请使用 --key=value 形式");
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
        inputPath = locateAsset("binary.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 素材是白底黑内容，取反让文字/形状成为前景。
    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY_INV);

    // 步骤 1：定义结构元素。
    int kernel = parser.get<int>("kernel");
    if (kernel < 3 || kernel % 2 == 0) {
        std::cerr << "kernel 必须是不小于 3 的奇数，已改用 5" << std::endl;
        kernel = 5;
    }
    const std::string shape = parser.get<std::string>("shape");
    int shapeCode = cv::MORPH_ELLIPSE;
    if (shape == "rect") {
        shapeCode = cv::MORPH_RECT;
    } else if (shape == "cross") {
        shapeCode = cv::MORPH_CROSS;
    }
    const cv::Mat element = cv::getStructuringElement(shapeCode, cv::Size(kernel, kernel));

    // 步骤 2~4：腐蚀、膨胀、开闭运算与梯度。
    cv::Mat eroded;
    cv::Mat dilated;
    cv::Mat opened;
    cv::Mat closed;
    cv::Mat gradient;
    cv::Mat tophat;
    cv::erode(binary, eroded, element);
    cv::dilate(binary, dilated, element);
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, element);
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, element);
    cv::morphologyEx(binary, gradient, cv::MORPH_GRADIENT, element);
    cv::morphologyEx(binary, tophat, cv::MORPH_TOPHAT, element);

    std::cout << "输入: " << inputPath << ", kernel=" << kernel << " (" << shape << ")\n";
    std::cout << "前景像素数:\n";
    printForeground("input", binary);
    printForeground("erode", eroded);
    printForeground("dilate", dilated);
    printForeground("open", opened);
    printForeground("close", closed);
    printForeground("gradient", gradient);

    const std::vector<cv::Mat> panels = {
        fitPanel(binary, "binary input"),
        fitPanel(eroded, "erode"),
        fitPanel(dilated, "dilate"),
        fitPanel(opened, "open (remove specks)"),
        fitPanel(closed, "close (fill holes)"),
        fitPanel(gradient, "morph gradient")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/morphology.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("morphology", result);
        cv::waitKey(0);
    }
    return 0;
}
