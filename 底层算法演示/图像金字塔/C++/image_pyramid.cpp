/**
 * @file    image_pyramid.cpp
 * @brief   图像金字塔演示：构建高斯金字塔、拉普拉斯金字塔并用拉普拉斯金字塔重建原图。
 * @details 流程：高斯模糊降采样 -> 上采样重建 -> 拉普拉斯计算 -> 由拉普拉斯金字塔重建。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          pyrDown = 高斯模糊 + 隔行隔列采样，pyrUp 只是插值放大（不可逆），
 *          所以拉普拉斯层保存的正是「上采样丢失的高频」；把它们逐层加回去即可无损重建。
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

/**
 * @brief  把带正负值的拉普拉斯层可视化（零点映射到中灰）。
 * @param  laplacian 拉普拉斯层（CV_32F）。
 * @return 8 位灰度可视化图。
 */
cv::Mat laplacianView(const cv::Mat& laplacian)
{
    cv::Mat normalized;
    cv::normalize(laplacian, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat view;
    normalized.convertTo(view, CV_8U);
    return view;
}

/** @brief 计算两幅 8 位灰度图的 PSNR(dB)。 */
double psnr(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    const double mse = cv::mean(diff.mul(diff))[0];
    return mse <= 0.0 ? 99.0 : 10.0 * std::log2(255.0 * 255.0 / mse) / std::log2(10.0);
}

}  // namespace

/**
 * @brief  主流程：构建两层金字塔并做无损重建。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/skyline.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_pyramid.png }"
        "{l levels       | 3    | 金字塔层数（不含第 0 层） }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像金字塔演示: 高斯金字塔 + 拉普拉斯金字塔重建\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("skyline.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    const int levels = std::max(1, std::min(4, parser.get<int>("levels")));

    // 步骤 1：高斯模糊降采样。每层尺寸减半，细节逐层减少。
    std::vector<cv::Mat> gaussian = {gray};
    for (int i = 0; i < levels; ++i) {
        cv::Mat down;
        cv::pyrDown(gaussian.back(), down);
        gaussian.push_back(down);
    }

    // 步骤 2~3：上采样重建并求拉普拉斯层（上采样丢失的高频）。
    std::vector<cv::Mat> laplacian(static_cast<size_t>(levels));
    for (int i = 0; i < levels; ++i) {
        // 全程用浮点做上采样，避免与重建阶段的整数上采样产生不一致的舍入误差。
        cv::Mat current;
        cv::Mat nextFloat;
        gaussian[static_cast<size_t>(i)].convertTo(current, CV_32F);
        gaussian[static_cast<size_t>(i) + 1].convertTo(nextFloat, CV_32F);
        cv::Mat upFloat;
        cv::pyrUp(nextFloat, upFloat, gaussian[static_cast<size_t>(i)].size());
        laplacian[static_cast<size_t>(i)] = current - upFloat;
    }

    // 步骤 4：由拉普拉斯金字塔逐层重建。
    cv::Mat reconstructed = gaussian.back().clone();
    for (int i = levels - 1; i >= 0; --i) {
        cv::Mat up;
        cv::pyrUp(reconstructed, up, gaussian[static_cast<size_t>(i)].size());
        cv::Mat upFloat;
        up.convertTo(upFloat, CV_32F);
        reconstructed = upFloat + laplacian[static_cast<size_t>(i)];
    }
    cv::Mat rebuilt;
    reconstructed.convertTo(rebuilt, CV_8U);

    std::cout << "输入: " << inputPath << ", 层数: " << levels + 1 << "\n";
    for (size_t i = 0; i < gaussian.size(); ++i) {
        std::cout << "  level " << i << ": " << gaussian[i].cols << "x" << gaussian[i].rows << "\n";
    }
    std::cout << "  重建 PSNR: " << std::fixed << std::setprecision(2) << psnr(rebuilt, gray)
              << " dB (拉普拉斯金字塔可无损重建)\n";

    cv::Mat error;
    cv::absdiff(rebuilt, gray, error);
    error.convertTo(error, CV_8U, 8.0);  // 放大 8 倍才能看清残差

    std::vector<cv::Mat> panels = {
        fitPanel(gray, "level 0 (original)"),
        fitPanel(gaussian[1], "level 1 (pyrDown)"),
        fitPanel(gaussian[2], "level 2 (pyrDown)")};
    if (laplacian.size() >= 1) {
        panels.push_back(fitPanel(laplacianView(laplacian[0]), "laplacian level 0"));
    }
    panels.push_back(fitPanel(rebuilt, "reconstructed"));
    panels.push_back(fitPanel(error, "abs error x8"));
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_pyramid.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image pyramid", result);
        cv::waitKey(0);
    }
    return 0;
}
