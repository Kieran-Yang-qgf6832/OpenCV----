/**
 * @file    grabcut.cpp
 * @brief   GrabCut 演示：用矩形框初始化，迭代优化 GMM 模型并提取前景。
 * @details 流程：定义初始矩形 -> 迭代优化模型 -> 涂鸦修正（可选）-> 提取前景。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          GrabCut 在 RGB 空间建 GMM，先验只有「矩形内=可能前景、外=确定背景」，
 *          所以矩形必须把目标完整框住且尽量贴紧，否则背景颜色会混进前景模型。
 */

#include <algorithm>
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

}  // namespace

/**
 * @brief  主流程：矩形初始化的 GrabCut 前景提取。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/subject.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/grabcut.png }"
        "{m margin       | 0.10 | 矩形相对边界的留白比例 }"
        "{n iters        | 5    | 迭代次数 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("GrabCut 演示: 矩形初始化 + GMM 迭代\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("subject.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：定义初始矩形。留一圈边距，保证背景像素也能进入背景 GMM。
    const double margin = std::max(0.0, std::min(0.4, parser.get<double>("margin")));
    const cv::Rect rect(static_cast<int>(src.cols * margin), static_cast<int>(src.rows * margin),
                        static_cast<int>(src.cols * (1.0 - 2 * margin)),
                        static_cast<int>(src.rows * (1.0 - 2 * margin)));

    // 步骤 2：迭代优化模型。
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat bgdModel;
    cv::Mat fgdModel;
    const int iterations = std::max(1, parser.get<int>("iters"));
    cv::grabCut(src, mask, rect, bgdModel, fgdModel, iterations, cv::GC_INIT_WITH_RECT);

    // 步骤 3~4：按标签提取前景。PR_FGD 表示「可能前景」，与 FGD 一并保留。
    cv::Mat binary = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
    cv::Mat foreground(src.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    src.copyTo(foreground, binary);
    cv::Mat background(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    src.copyTo(background, ~binary);

    // 标签可视化：确定前景绿、可能前景浅绿、确定背景黑、可能背景灰。
    cv::Mat labelView(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    labelView.setTo(cv::Scalar(0, 60, 0), mask == cv::GC_FGD);
    labelView.setTo(cv::Scalar(0, 160, 0), mask == cv::GC_PR_FGD);
    labelView.setTo(cv::Scalar(60, 60, 60), mask == cv::GC_PR_BGD);

    const double ratio = 100.0 * cv::countNonZero(binary) / binary.total();
    std::cout << "输入: " << inputPath << ", 迭代 " << iterations << " 次\n";
    std::cout << "  前景占比: " << ratio << "%\n";

    cv::Mat rectView = src.clone();
    cv::rectangle(rectView, rect, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

    const std::vector<cv::Mat> panels = {
        fitPanel(rectView, "initial rect"),
        fitPanel(labelView, "grabCut labels"),
        fitPanel(foreground, "foreground"),
        fitPanel(background, "background")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/grabcut.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("grabcut", result);
        cv::waitKey(0);
    }
    return 0;
}
