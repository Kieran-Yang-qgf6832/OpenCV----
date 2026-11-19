/**
 * @file    watershed_segmentation.cpp
 * @brief   分水岭分割演示：用距离变换 + 标记控制的分水岭分离相互接触的物体。
 * @details 流程：二值化 -> 距离变换 -> 标记前景背景 -> 应用分水岭 -> 着色输出。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          直接对梯度图跑分水岭会因噪声产生大量过分割，所以必须先用距离变换找出
 *          每个物体的「种子」（确定的内部区域），再把未知区域交给分水岭漫水填充。
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
 * @brief  按分水岭标记给每个物体上色，边界画成红色。
 * @param  markers 分水岭输出的标记图（-1 表示边界，1 表示背景）。
 * @param  labelCount 连通域数量。
 * @return 着色结果。
 */
cv::Mat colorizeMarkers(const cv::Mat& markers, int labelCount)
{
    std::vector<cv::Vec3b> colors(static_cast<size_t>(labelCount) + 1, cv::Vec3b(255, 255, 255));
    cv::RNG rng(20261119);
    for (size_t i = 2; i < colors.size(); ++i) {
        colors[i] = cv::Vec3b(static_cast<uchar>(rng.uniform(40, 255)),
                              static_cast<uchar>(rng.uniform(40, 255)),
                              static_cast<uchar>(rng.uniform(40, 255)));
    }

    cv::Mat colorized(markers.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        cv::Vec3b* out = colorized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < markers.cols; ++x) {
            if (row[x] == -1) {
                out[x] = cv::Vec3b(0, 0, 255);  // 分水岭边界
            } else if (row[x] >= 0 && row[x] < static_cast<int>(colors.size())) {
                out[x] = colors[static_cast<size_t>(row[x])];
            }
        }
    }
    return colorized;
}

}  // namespace

/**
 * @brief  主流程：二值化、距离变换、标记、分水岭与着色。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/coins.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/watershed_segmentation.png }"
        "{f fgRatio      | 0.4  | 前景种子阈值，占最大距离的比例 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("分水岭算法演示: 距离变换 + 标记控制的分水岭分割\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("coins.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }

    // 步骤 1：二值化。硬币比背景亮，Otsu 能自动找到阈值。
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    cv::Mat binary;
    const double otsu = cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    // 步骤 2：距离变换。每个前景像素到最近背景的距离，峰值即各物体的中心。
    cv::Mat distance;
    cv::distanceTransform(binary, distance, cv::DIST_L2, 5);
    double maxDistance = 0.0;
    cv::minMaxLoc(distance, nullptr, &maxDistance);
    cv::Mat distanceView;
    cv::normalize(distance, distanceView, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat sureFg;
    cv::threshold(distance, sureFg, parser.get<double>("fgRatio") * maxDistance, 255,
                  cv::THRESH_BINARY);
    sureFg.convertTo(sureFg, CV_8U);
    cv::morphologyEx(sureFg, sureFg, cv::MORPH_OPEN, kernel);

    // 步骤 3：标记前景与背景。膨胀后的二值图是确定背景，两者之差是未知区域。
    cv::Mat sureBg;
    cv::dilate(binary, sureBg, kernel, cv::Point(-1, -1), 3);
    cv::Mat unknown;
    cv::subtract(sureBg, sureFg, unknown);

    cv::Mat labels;
    const int labelCount = cv::connectedComponents(sureFg, labels);
    cv::Mat markers = labels + 1;  // 背景标记为 1，物体从 2 开始
    markers.setTo(0, unknown == 255);  // 未知区域置 0，交给分水岭决定归属

    // 步骤 4：应用分水岭，边界会被标记成 -1。
    cv::watershed(src, markers);

    // 着色与叠加。
    const cv::Mat colorized = colorizeMarkers(markers, labelCount);
    cv::Mat overlay = src.clone();
    overlay.setTo(cv::Scalar(0, 0, 255), markers == -1);

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(binary, "otsu binary"),
        fitPanel(distanceView, "distance transform"),
        fitPanel(sureFg, "sure foreground (seeds)"),
        fitPanel(colorized, "watershed labels"),
        fitPanel(overlay, "boundaries on original")};
    const cv::Mat result = montage(panels, 3);

    std::cout << "输入: " << inputPath << ", Otsu 阈值 = " << otsu << "\n";
    std::cout << "检测到物体数: " << labelCount - 1 << "\n";

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/watershed_segmentation.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("watershed segmentation", result);
        cv::waitKey(0);
    }
    return 0;
}
