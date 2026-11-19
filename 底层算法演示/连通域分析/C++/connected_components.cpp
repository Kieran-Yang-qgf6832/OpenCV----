/**
 * @file    connected_components.cpp
 * @brief   连通域分析演示：标记二值图中的连通区域并统计面积、包围盒、质心。
 * @details 流程：二值化 -> 连通域标记 -> 计算统计信息 -> 按条件过滤并可视化。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          connectedComponents 只返回标记图与数量；需要面积/包围盒/质心时用
 *          connectedComponentsWithStats，它一次性给出全部统计量，避免再遍历像素。
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
    const size_t slash2 = dir.find_last_of("\\/");
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
 * @brief  按标记值给连通域上色（同一标签同色）。
 * @param  labels 标记图（CV_32S）。
 * @param  count  标签数（含背景 0）。
 * @return BGR 着色图。
 */
cv::Mat colorizeLabels(const cv::Mat& labels, int count)
{
    std::vector<cv::Vec3b> palette(static_cast<size_t>(count), cv::Vec3b(0, 0, 0));
    cv::RNG rng(20261119);
    for (int i = 1; i < count; ++i) {
        palette[static_cast<size_t>(i)] =
            cv::Vec3b(static_cast<uchar>(rng.uniform(50, 255)),
                      static_cast<uchar>(rng.uniform(50, 255)),
                      static_cast<uchar>(rng.uniform(50, 255)));
    }

    cv::Mat colored(labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < labels.rows; ++y) {
        const int* row = labels.ptr<int>(y);
        cv::Vec3b* out = colored.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] > 0 && row[x] < count) {
                out[x] = palette[static_cast<size_t>(row[x])];
            }
        }
    }
    return colored;
}

}  // namespace

/**
 * @brief  主流程：标记连通域、统计并按面积过滤。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/connected_components.png }"
        "{a minArea      | 200  | 最小面积过滤阈值，小于该值视为噪声 }"
        "{c connectivity | 8    | 邻接方式，4 或 8 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("连通域分析演示: 标记、统计与面积过滤\n参数请使用 --key=value 形式");
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

    // 步骤 1：二值化。素材是白底黑形状，取反让形状成为前景。
    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY_INV);

    // 步骤 2~3：标记并一次性取出统计信息。
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int connectivity = parser.get<int>("connectivity") == 4 ? 4 : 8;
    const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids,
                                                       connectivity, CV_32S);

    // 步骤 4：按面积过滤。
    const int minArea = parser.get<int>("minArea");
    cv::Mat filtered = binary.clone();
    cv::Mat overlay;
    cv::cvtColor(src, overlay, cv::COLOR_GRAY2BGR);
    int kept = 0;
    std::cout << "输入: " << inputPath << ", 邻接方式: " << connectivity << " 邻接\n";
    std::cout << "  原始连通域(含背景): " << count << "\n";
    std::cout << std::left << std::setw(6) << "label" << std::setw(10) << "area"
              << std::setw(20) << "bbox(x,y,w,h)" << "centroid\n";
    for (int i = 1; i < count; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < minArea) {
            continue;
        }
        ++kept;
        const cv::Rect box(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                           stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        cv::rectangle(overlay, box, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(overlay, std::to_string(i), cv::Point(box.x + 4, box.y + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        if (kept <= 10) {
            std::cout << std::left << std::setw(6) << i << std::setw(10) << area
                      << std::setw(20)
                      << (std::to_string(box.x) + "," + std::to_string(box.y) + "," +
                          std::to_string(box.width) + "," + std::to_string(box.height))
                      << "(" << std::fixed << std::setprecision(1)
                      << centroids.at<double>(i, 0) << ", " << centroids.at<double>(i, 1) << ")\n";
        }
    }
    std::cout << "  面积 >= " << minArea << " 的连通域: " << kept << " 个\n";

    const cv::Mat colored = colorizeLabels(labels, count);
    const std::vector<cv::Mat> panels = {
        fitPanel(binary, "binary input"),
        fitPanel(colored, "labels: " + std::to_string(count - 1)),
        fitPanel(overlay, "area >= " + std::to_string(minArea) + ": " + std::to_string(kept))};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/connected_components.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("connected components", result);
        cv::waitKey(0);
    }
    return 0;
}
