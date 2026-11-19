/**
 * @file    image_filtering.cpp
 * @brief   图像滤波演示：高斯、中值、双边滤波与锐化的效果对比。
 * @details 流程：选择滤波器类型 -> 设定核大小 -> 卷积运算 -> 对比效果。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          高斯/均值按空间距离加权，边缘一起被模糊；中值对椒盐噪声最有效；
 *          双边滤波同时考虑空间距离与灰度差，因此能「保边去噪」。
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
 * @brief  用拉普拉斯方差衡量图像的高频（细节/噪声）强度。
 * @param  gray 灰度图。
 * @return 拉普拉斯响应的方差，越大越锐利。
 */
double sharpness(const cv::Mat& gray)
{
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_32F, 3);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

}  // namespace

/**
 * @brief  主流程：四种滤波 + 锐化的对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/texture.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/image_filtering.png }"
        "{k kernel       | 7    | 高斯/中值滤波的核大小（奇数） }"
        "{s sigma        | 0    | 高斯标准差，0 表示由核大小推导 }"
        "{e sigmaColor   | 60   | 双边滤波的颜色标准差 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("图像滤波演示: 高斯 / 中值 / 双边 / 锐化\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("texture.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int kernel = parser.get<int>("kernel");
    if (kernel < 3 || kernel % 2 == 0) {
        std::cerr << "kernel 必须是不小于 3 的奇数，已改用 7" << std::endl;
        kernel = 7;
    }
    const double sigma = parser.get<double>("sigma");

    // 步骤 1~2：按核大小与标准差构造各滤波器。
    // 为了体现降噪效果，先人为加入高斯噪声作为输入。
    cv::Mat noisy;
    cv::Mat noise(gray.size(), CV_32F);
    cv::RNG rng(20261119);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, 18.0);
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    cv::Mat noisyFloat = grayFloat + noise;
    noisyFloat.convertTo(noisy, CV_8U);

    // 步骤 3：卷积运算。
    cv::Mat gaussian;
    cv::Mat median;
    cv::Mat bilateral;
    cv::Mat sharpened;
    cv::GaussianBlur(noisy, gaussian, cv::Size(kernel, kernel), sigma);
    cv::medianBlur(noisy, median, kernel);
    cv::bilateralFilter(noisy, bilateral, kernel, parser.get<double>("sigmaColor"),
                        kernel / 2.0);
    // 非锐化掩模：原图减去模糊图得到高频，再按比例加回。
    cv::Mat blurredForSharpen;
    cv::GaussianBlur(gray, blurredForSharpen, cv::Size(0, 0), 2.0);
    cv::addWeighted(gray, 1.6, blurredForSharpen, -0.6, 0.0, sharpened);

    // 步骤 4：对比效果。锐度指标用拉普拉斯方差。
    std::cout << "输入: " << inputPath << ", kernel=" << kernel << "\n";
    std::cout << std::left << std::setw(12) << "image" << "sharpness\n";
    const std::pair<std::string, cv::Mat> items[] = {
        {"original", gray}, {"noisy", noisy},  {"gaussian", gaussian},
        {"median", median}, {"bilateral", bilateral}, {"sharpened", sharpened}};
    for (const auto& item : items) {
        std::cout << std::left << std::setw(12) << item.first << std::fixed
                  << std::setprecision(1) << sharpness(item.second) << "\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "original"),
        fitPanel(noisy, "noisy (sigma 18)"),
        fitPanel(gaussian, "gaussian"),
        fitPanel(median, "median"),
        fitPanel(bilateral, "bilateral"),
        fitPanel(sharpened, "unsharp mask")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/image_filtering.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("image filtering", result);
        cv::waitKey(0);
    }
    return 0;
}
