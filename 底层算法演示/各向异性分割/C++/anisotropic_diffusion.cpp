/**
 * @file    anisotropic_diffusion.cpp
 * @brief   各向异性扩散演示：Perona-Malik 扩散在平滑噪声的同时保留边缘。
 * @details 流程：设置扩散参数 -> 迭代扩散 -> 与高斯滤波对比 -> 用 PSNR 与边缘强度量化评估。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）+ opencv_contrib 的 ximgproc
 *          各向异性扩散的扩散系数随梯度增大而减小，因此平坦区被强平滑、边缘处几乎不扩散；
 *          高斯滤波对全图一视同仁，必然把边缘一起糊掉。
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ximgproc.hpp>

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
 * @brief  计算两幅图之间的峰值信噪比。
 * @param  a 图像 A。
 * @param  b 图像 B（参考图）。
 * @return PSNR( dB )。
 */
double psnr(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_32F);
    const double mse = cv::mean(diff.mul(diff))[0];
    return mse <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

/**
 * @brief  用梯度幅值的 99 分位数衡量边缘保留程度。
 * @param  gray 灰度图。
 * @return 梯度幅值的 99 分位数，越大说明物体边界越锐利。
 * @note   不能用梯度均值：均值被残余噪声主导，滤波越弱反而越大。
 *         真边缘位于高分位，取 99 分位才能把「边界陡不陡」单独拎出来。
 */
double edgeGradientP99(const cv::Mat& gray)
{
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    cv::Mat flat = magnitude.reshape(1, 1).clone();
    cv::sort(flat, flat, cv::SORT_ASCENDING | cv::SORT_EVERY_ROW);
    const int index = std::min(static_cast<int>(flat.total()) - 1,
                               static_cast<int>(flat.total() * 0.99));
    return flat.ptr<float>(0)[index];
}

/// 前置声明：随后定义，供 matchGaussianSigma 内部使用。
double flatRegionNoise(const cv::Mat& gray);

/**
 * @brief  自动匹配高斯滤波的 sigma，使其平坦区噪声与扩散结果相当。
 * @param  src          含噪彩色图。
 * @param  targetNoise  目标平坦区噪声标准差。
 * @return 最接近目标的高斯 sigma。
 * @note   只有把两者的平滑力度拉到同一水平，比较边缘锐利度才有意义。
 */
double matchGaussianSigma(const cv::Mat& src, double targetNoise)
{
    auto noiseOf = [](double sigma, const cv::Mat& image) {
        cv::Mat blurred;
        cv::GaussianBlur(image, blurred, cv::Size(0, 0), sigma);
        cv::Mat gray;
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
        return flatRegionNoise(gray);
    };

    double bestSigma = 1.0;
    double bestDiff = 1e9;
    for (double sigma = 0.2; sigma <= 3.0 + 1e-9; sigma += 0.1) {
        const double diff = std::fabs(noiseOf(sigma, src) - targetNoise);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestSigma = sigma;
        }
    }
    for (double sigma = std::max(0.05, bestSigma - 0.1);
         sigma <= bestSigma + 0.1 + 1e-9; sigma += 0.01) {
        const double diff = std::fabs(noiseOf(sigma, src) - targetNoise);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestSigma = sigma;
        }
    }
    return bestSigma;
}

/**
 * @brief  取左上角一块平坦背景，用其标准差衡量残余噪声。
 * @param  gray 灰度图。
 * @return 标准差，越小说明噪声被压得越干净。
 */
double flatRegionNoise(const cv::Mat& gray)
{
    const int size = std::min(200, std::min(gray.rows, gray.cols));
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray(cv::Rect(0, 0, size, size)), mean, stddev);
    return stddev[0];
}

}  // namespace

/**
 * @brief  主流程：各向异性扩散与高斯滤波的定量对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |       | 打印帮助信息 }"
        "{i input        |       | 含噪图像，缺省读取 assets/noisy.png }"
        "{r reference    |       | 干净参考图，缺省读取 assets/smooth.png }"
        "{o output       |       | 输出图像路径，缺省写入 <目录>/output/anisotropic_diffusion.png }"
        "{a alpha        | 0.05  | 扩散速率，越大平滑越快，超过 0.25 会不稳定 }"
        "{k K            | 0.10  | 边缘敏感度(归一化尺度)，梯度超过 K 后几乎不再扩散 }"
        "{n niters       | 10    | 迭代次数 }"
        "{s sigma        | 0     | 对照用高斯滤波的标准差，0 表示自动匹配噪声水平 }"
        "{d display      | 0     | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("各向异性扩散演示: 边缘保持平滑 vs 高斯滤波\n参数请使用 --key=value 形式");
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
    std::string refPath = parser.get<std::string>("reference");
    if (inputPath.empty()) {
        inputPath = locateAsset("noisy.png");
    }
    if (refPath.empty()) {
        refPath = locateAsset("smooth.png");
    }

    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 1：设置扩散参数（alpha 决定速率，K 决定把多大的梯度当成边缘）。
    const double alpha = parser.get<double>("alpha");
    const double k = parser.get<double>("k");
    const int iterations = std::max(1, parser.get<int>("niters"));

    // 步骤 2：迭代扩散。该函数只接受 8UC3 彩色输入，因此直接处理彩色图。
    cv::Mat filtered;
    cv::ximgproc::anisotropicDiffusion(src, filtered, alpha, k, iterations);

    // 步骤 3：对比高斯滤波。先把 sigma 调到与扩散结果相同的平坦区噪声，
    //        否则「平滑力度不同」会让边缘锐利度的比较失去意义。
    //        指标一律在灰度图上计算，与参考图保持同一口径。
    cv::Mat filteredGray;
    cv::cvtColor(filtered, filteredGray, cv::COLOR_BGR2GRAY);
    double sigma = parser.get<double>("sigma");
    if (sigma <= 0.0) {
        sigma = matchGaussianSigma(src, flatRegionNoise(filteredGray));
    }
    cv::Mat gaussian;
    cv::GaussianBlur(src, gaussian, cv::Size(0, 0), sigma);
    cv::Mat gaussianGray;
    cv::cvtColor(gaussian, gaussianGray, cv::COLOR_BGR2GRAY);

    // 步骤 4：量化评估。
    cv::Mat reference = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
    const bool hasRef = !reference.empty() && reference.size() == gray.size();

    std::cout << "输入: " << inputPath << ", alpha = " << alpha << ", K = " << k
              << ", niters = " << iterations << ", 匹配 sigma = " << sigma << "\n";
    std::cout << std::left << std::setw(22) << "method" << std::setw(14) << "flat noise"
              << std::setw(14) << "grad p99" << "PSNR(dB)\n";
    auto report = [&](const std::string& name, const cv::Mat& image) {
        std::cout << std::left << std::setw(22) << name
                  << std::fixed << std::setprecision(2)
                  << std::setw(16) << flatRegionNoise(image)
                  << std::setw(16) << edgeGradientP99(image);
        if (hasRef) {
            std::cout << std::setprecision(2) << psnr(image, reference);
        } else {
            std::cout << "-";
        }
        std::cout << "\n";
    };
    report("noisy input", gray);
    report("anisotropic", filteredGray);
    report("gaussian", gaussianGray);
    if (hasRef) {
        report("clean reference", reference);
    } else {
        std::cout << "参考图缺失或尺寸不一致，跳过 PSNR\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "noisy input"),
        fitPanel(hasRef ? reference : gray, "clean reference"),
        fitPanel(filtered, "anisotropic diffusion"),
        fitPanel(gaussian, "gaussian sigma " + std::to_string(sigma).substr(0, 4))};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/anisotropic_diffusion.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("anisotropic diffusion", result);
        cv::waitKey(0);
    }
    return 0;
}
