/**
 * @file    image_addition.cpp
 * @brief   图像加法演示：直接相加、加权混合与 ROI 局部融合。
 * @details 流程：读取两幅图像 -> 尺寸对齐 -> cv::add 直接相加 -> addWeighted 加权混合
 *          -> ROI 局部融合 -> 统计过曝比例并输出对比图。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          cv::add 与 C++ 的 Mat 表达式 a+b 都会饱和到 255；numpy 的 uint8 相加则会
 *          取模回绕（200+100=44），Python 版因此显式使用 cv2.add 保证结果一致。
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
 * @brief  给图像加标题条，便于拼接展示。
 * @param  image 三通道图像。
 * @param  title 标题文字（仅支持 ASCII）。
 * @return 带标题条的面板。
 */
cv::Mat makePanel(const cv::Mat& image, const std::string& title)
{
    // 展示时缩到一半，控制拼接图体积，同时保留足够可读性。
    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    small.copyTo(panel(cv::Rect(0, kTitleHeight, small.cols, small.rows)));
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
 * @brief  统计三通道中饱和到 255 的子像素比例，衡量相加后的过曝程度。
 * @param  image 三通道图像。
 * @return 过曝比例（0~1）。
 */
double saturatedRatio(const cv::Mat& image)
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    double saturated = 0.0;
    for (const auto& channel : channels) {
        saturated += static_cast<double>(cv::countNonZero(channel == 255));
    }
    return saturated / (3.0 * image.total());
}

}  // namespace

/**
 * @brief  主流程：读取两图、三种加法策略、ROI 融合并输出对比图。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{a imageA       |      | 第一幅图像，缺省读取 assets/image_a.png }"
        "{b imageB       |      | 第二幅图像，缺省读取 assets/image_b.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_addition.png }"
        "{alpha          | 0.5  | 加权混合中第一幅图的权重 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像加法演示: 直接相加 / 加权混合 / ROI 融合\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string pathA = parser.get<std::string>("imageA");
    std::string pathB = parser.get<std::string>("imageB");
    if (pathA.empty()) {
        pathA = locateAsset("image_a.png");
    }
    if (pathB.empty()) {
        pathB = locateAsset("image_b.png");
    }

    // 步骤 1：读取两幅图像并统一尺寸（逐像素运算要求尺寸与通道数一致）。
    const cv::Mat srcA = cv::imread(pathA, cv::IMREAD_COLOR);
    cv::Mat srcB = cv::imread(pathB, cv::IMREAD_COLOR);
    if (srcA.empty() || srcB.empty()) {
        std::cerr << "无法读取图像: " << (srcA.empty() ? pathA : pathB) << std::endl;
        return -1;
    }
    if (srcA.size() != srcB.size()) {
        cv::resize(srcB, srcB, srcA.size(), 0, 0, cv::INTER_AREA);
    }

    // 步骤 2：直接相加。亮度整体抬高，亮部很容易饱和到 255。
    cv::Mat added;
    cv::add(srcA, srcB, added);

    // 步骤 3：加权混合。alpha 越大越偏向第一幅图，权重和为 1 时整体亮度不变。
    const double alpha = parser.get<double>("alpha");
    cv::Mat blended;
    cv::addWeighted(srcA, alpha, srcB, 1.0 - alpha, 0.0, blended);
    cv::Mat blendLight;
    cv::Mat blendHeavy;
    cv::addWeighted(srcA, alpha * 0.5, srcB, 1.0 - alpha * 0.5, 0.0, blendLight);
    cv::addWeighted(srcA, std::min(1.0, alpha + 0.25), srcB,
                    1.0 - std::min(1.0, alpha + 0.25), 0.0, blendHeavy);

    // 步骤 4：ROI 融合。只在指定区域内混合，区域外保持第一幅图。
    cv::Mat roiResult = srcA.clone();
    cv::Mat smallB;
    cv::resize(srcB, smallB, cv::Size(srcA.cols / 3, srcA.rows / 3), 0, 0, cv::INTER_AREA);
    const cv::Rect roi(60, 60, smallB.cols, smallB.rows);
    cv::Mat blendedRoi;
    cv::addWeighted(roiResult(roi), 0.5, smallB, 0.5, 0.0, blendedRoi);
    blendedRoi.copyTo(roiResult(roi));
    cv::rectangle(roiResult, roi, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    // 统计：直接相加的过曝比例远高于加权混合，这是选择策略的依据。
    cv::Mat diff;
    cv::absdiff(added, blended, diff);
    std::cout << "尺寸: " << srcA.cols << "x" << srcA.rows << ", alpha = " << alpha << "\n";
    std::cout << "  add 后饱和像素比例:    " << std::fixed << std::setprecision(2)
              << 100.0 * saturatedRatio(added) << "%\n";
    std::cout << "  blend 后饱和像素比例:  " << 100.0 * saturatedRatio(blended) << "%\n";
    const cv::Scalar diffMean = cv::mean(diff);  // 取三通道平均，与 Python 的 diff.mean() 对齐
    std::cout << "  add 与 blend 的平均差异: "
              << (diffMean[0] + diffMean[1] + diffMean[2]) / 3.0 << "\n";

    const std::vector<cv::Mat> panels = {
        makePanel(srcA, "image A"),
        makePanel(srcB, "image B"),
        makePanel(added, "add (saturate)"),
        makePanel(blended, "addWeighted " + std::to_string(alpha).substr(0, 4)),
        makePanel(blendLight, "addWeighted lighter"),
        makePanel(blendHeavy, "addWeighted heavier"),
        makePanel(roiResult, "ROI blend"),
        makePanel(diff, "absdiff(add, blend)")};
    const cv::Mat result = montage(panels, 4);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(pathA);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_addition.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image addition", result);
        cv::waitKey(0);
    }
    return 0;
}
