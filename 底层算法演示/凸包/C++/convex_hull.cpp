/**
 * @file    convex_hull.cpp
 * @brief   凸包演示：计算轮廓的凸包与凸缺陷，并用实心度衡量形状的凹陷程度。
 * @details 流程：提取轮廓点集 -> 计算凸包 -> 计算凸缺陷 -> 可视化。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          convexityDefects 返回的深度是定点数，需除以 256 才是像素值；
 *          并且它要求凸包用 returnPoints=false 得到索引，不能直接用点集。
 */

#include <algorithm>
#include <cmath>
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
 * @brief  主流程：凸包、凸缺陷与实心度统计。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 二值图，缺省读取 assets/binary.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/convex_hull.png }"
        "{a minArea      | 500  | 最小面积过滤阈值 }"
        "{d minDepth     | 1.0  | 凸缺陷的最小深度(像素)，滤掉抖动 }"
        "{w display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("凸包演示: convexHull + convexityDefects\n参数请使用 --key=value 形式");
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

    // 步骤 1：提取轮廓点集。素材白底黑形状，取反后形状成为前景。
    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat binaryView;
    cv::cvtColor(binary, binaryView, cv::COLOR_GRAY2BGR);
    cv::Mat hullView = binaryView.clone();
    cv::Mat defectView = binaryView.clone();

    const int minArea = parser.get<int>("minArea");
    const double minDepth = parser.get<double>("minDepth");
    int kept = 0;
    int totalDefects = 0;
    std::cout << "输入: " << inputPath << ", 轮廓总数: " << contours.size() << "\n";
    std::cout << std::left << std::setw(6) << "id" << std::setw(10) << "area"
              << std::setw(12) << "hullArea" << std::setw(10) << "solidity"
              << "defects\n";

    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area < minArea) {
            continue;
        }
        ++kept;

        // 步骤 2：计算凸包。两次调用：一次拿点集画图，一次拿索引算缺陷。
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull, false, true);
        std::vector<int> hullIndices;
        cv::convexHull(contours[i], hullIndices, false, false);

        const double hullArea = cv::contourArea(hull);
        const double solidity = hullArea > 0.0 ? area / hullArea : 0.0;

        // 步骤 3：计算凸缺陷。
        int defects = 0;
        if (hullIndices.size() > 3) {
            std::vector<cv::Vec4i> defectList;
            try {
                cv::convexityDefects(contours[i], hullIndices, defectList);
            } catch (const cv::Exception&) {
                defectList.clear();
            }
            for (const auto& defect : defectList) {
                const double depth = defect[3] / 256.0;  // 定点数转像素
                if (depth < minDepth) {
                    continue;
                }
                ++defects;
                const cv::Point start = contours[i][defect[0]];
                const cv::Point end = contours[i][defect[1]];
                const cv::Point far = contours[i][defect[2]];
                cv::line(defectView, start, far, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::line(defectView, end, far, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                cv::circle(defectView, far, 5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }
        }
        totalDefects += defects;

        cv::drawContours(hullView, std::vector<std::vector<cv::Point>>{contours[i]}, -1,
                         cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
        cv::drawContours(hullView, std::vector<std::vector<cv::Point>>{hull}, -1,
                         cv::Scalar(0, 180, 0), 2, cv::LINE_AA);

        if (kept <= 16) {
            std::cout << std::left << std::setw(6) << i << std::setw(10)
                      << static_cast<int>(area) << std::setw(12)
                      << static_cast<int>(hullArea) << std::fixed << std::setprecision(3)
                      << std::setw(10) << solidity << defects << "\n";
        }
    }
    std::cout << "面积 >= " << minArea << " 的轮廓: " << kept
              << " 个, 深度 >= " << minDepth << " 的凸缺陷共 " << totalDefects << " 个\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(binary, "binary contours"),
        fitPanel(hullView, "convex hull (green)"),
        fitPanel(defectView, "defects (red)")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/convex_hull.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("convex hull", result);
        cv::waitKey(0);
    }
    return 0;
}
