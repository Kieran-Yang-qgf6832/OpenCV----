/**
 * @file    template_matching.cpp
 * @brief   模板匹配演示：在多尺度上滑动窗口匹配，定位模板并输出响应热力图。
 * @details 流程：读取模板与目标 -> 选择匹配方法 -> 多尺度滑动窗口匹配 -> 定位最佳位置。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          TM_CCOEFF_NORMED 对亮度线性变化不敏感，但匹配结果只有「平移」自由度；
 *          模板尺寸一变就失效，所以工程上必须配合多尺度搜索（本演示的核心）。
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

}  // namespace

/**
 * @brief  主流程：多尺度模板匹配并输出最佳位置与热力图。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{s scene        |      | 场景图，缺省读取 assets/scene.png }"
        "{t template     |      | 模板图，缺省读取 assets/template.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/template_matching.png }"
        "{a minScale     | 0.5  | 最小缩放系数 }"
        "{b maxScale     | 1.5  | 最大缩放系数 }"
        "{n steps        | 11   | 缩放扫描点数 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("模板匹配演示: 多尺度 matchTemplate 搜索\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string scenePath = parser.get<std::string>("scene");
    std::string templatePath = parser.get<std::string>("template");
    if (scenePath.empty()) {
        scenePath = locateAsset("scene.png");
    }
    if (templatePath.empty()) {
        templatePath = locateAsset("template.png");
    }

    // 步骤 1：读取模板和目标。
    const cv::Mat sceneColor = cv::imread(scenePath, cv::IMREAD_COLOR);
    const cv::Mat templateColor = cv::imread(templatePath, cv::IMREAD_COLOR);
    if (sceneColor.empty() || templateColor.empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return -1;
    }
    cv::Mat sceneGray;
    cv::Mat templateGray;
    cv::cvtColor(sceneColor, sceneGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(templateColor, templateGray, cv::COLOR_BGR2GRAY);

    // 步骤 2~3：多尺度滑动窗口匹配。matchTemplate 内部就是一次全图的相关滤波。
    const double minScale = std::max(0.1, parser.get<double>("minScale"));
    const double maxScale = std::max(minScale, parser.get<double>("maxScale"));
    const int steps = std::max(2, parser.get<int>("steps"));
    const double stepSize = (maxScale - minScale) / (steps - 1);

    double bestScore = -1.0;
    double bestScale = 1.0;
    cv::Point bestLocation;
    cv::Mat bestResult;
    std::cout << "尺度扫描 (TM_CCOEFF_NORMED):\n";
    for (int i = 0; i < steps; ++i) {
        const double scale = minScale + i * stepSize;
        cv::Mat scaledTemplate;
        cv::resize(templateGray, scaledTemplate, cv::Size(), scale, scale, cv::INTER_AREA);
        if (scaledTemplate.cols >= sceneGray.cols || scaledTemplate.rows >= sceneGray.rows) {
            continue;
        }
        cv::Mat result;
        cv::matchTemplate(sceneGray, scaledTemplate, result, cv::TM_CCOEFF_NORMED);
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::Point minLocation;
        cv::Point maxLocation;
        cv::minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation);
        if (i % 2 == 0) {
            std::cout << "  scale=" << std::fixed << std::setprecision(2) << scale
                      << " -> peak=" << maxValue << "\n";
        }
        if (maxValue > bestScore) {
            bestScore = maxValue;
            bestScale = scale;
            bestLocation = maxLocation;
            bestResult = result.clone();
        }
    }

    // 步骤 4：定位最佳位置。
    cv::Mat matchedTemplate;
    cv::resize(templateGray, matchedTemplate, cv::Size(), bestScale, bestScale, cv::INTER_AREA);
    const cv::Rect matchBox(bestLocation.x, bestLocation.y,
                            matchedTemplate.cols, matchedTemplate.rows);

    cv::Mat sceneView = sceneColor.clone();
    cv::rectangle(sceneView, matchBox, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    cv::putText(sceneView,
                "score " + std::to_string(bestScore).substr(0, 5) + " scale " +
                    std::to_string(bestScale).substr(0, 4),
                cv::Point(matchBox.x, std::max(20, matchBox.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    // 响应热力图：越红表示与模板越相似。
    cv::Mat heatmapSource;
    cv::normalize(bestResult, heatmapSource, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat heatmap;
    cv::applyColorMap(heatmapSource, heatmap, cv::COLORMAP_JET);

    std::cout << "\n最佳: scale=" << bestScale << ", score=" << bestScore
              << ", 位置=(" << bestLocation.x << ", " << bestLocation.y << ")\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(sceneView, "detected"),
        fitPanel(templateColor, "template"),
        fitPanel(heatmap, "response heatmap"),
        fitPanel(sceneColor(matchBox), "matched patch")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(scenePath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/template_matching.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("template matching", result);
        cv::waitKey(0);
    }
    return 0;
}
