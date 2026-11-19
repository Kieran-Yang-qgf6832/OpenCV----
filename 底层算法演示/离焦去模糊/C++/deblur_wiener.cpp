/**
 * @file    deblur_wiener.cpp
 * @brief   离焦去模糊演示：用维纳滤波在频域复原高斯离焦模糊。
 * @details 流程：模拟/读取离焦模糊 -> 估计点扩散函数 -> 维纳滤波复原 -> 用 PSNR 评估质量。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          维纳滤波 F = G·H* / (|H|² + K)：K 是噪声与信号的功率比，取太小会放大噪声，
 *          取太大则退化为低通、几乎没有锐化效果，所以 K 必须和 PSF 一起调。
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
const int kPsfSize = 63;

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

/** @brief 四个象限对角互换，把零频搬到画布中心。 */
void fftShift(cv::Mat& mat)
{
    const int cx = mat.cols / 2;
    const int cy = mat.rows / 2;
    cv::Mat q0(mat, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mat, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mat, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mat, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/**
 * @brief  构造高斯点扩散函数的频域形式。
 * @param  size  图像尺寸。
 * @param  sigma 高斯半径，即离焦程度。
 * @return 双通道复数频谱（零频在左上角）。
 * @note   先把空间域的核放在画布中心，再做 fftShift 把中心搬到原点，
 *         否则负频部分会落在错误的位置，复原结果完全错误。
 */
cv::Mat gaussianPsfSpectrum(cv::Size size, double sigma)
{
    const cv::Mat kernel1d = cv::getGaussianKernel(kPsfSize, sigma);
    cv::Mat psf = kernel1d * kernel1d.t();  // 二维可分离核

    cv::Mat padded = cv::Mat::zeros(size, CV_32F);
    const int half = kPsfSize / 2;
    psf.copyTo(padded(cv::Rect(size.width / 2 - half, size.height / 2 - half,
                               kPsfSize, kPsfSize)));
    fftShift(padded);

    cv::Mat spectrum;
    cv::dft(padded, spectrum, cv::DFT_COMPLEX_OUTPUT);
    return spectrum;
}

/**
 * @brief  维纳滤波复原。
 * @param  gray    模糊灰度图（CV_8U）。
 * @param  sigma   高斯 PSF 的标准差。
 * @param  k       噪声与信号功率比。
 * @return 复原结果（CV_8U）。
 */
cv::Mat wienerDeblur(const cv::Mat& gray, double sigma, double k)
{
    cv::Mat srcFloat;
    gray.convertTo(srcFloat, CV_32F);
    cv::Mat srcSpectrum;
    cv::dft(srcFloat, srcSpectrum, cv::DFT_COMPLEX_OUTPUT);

    const cv::Mat psfSpectrum = gaussianPsfSpectrum(gray.size(), sigma);

    // 分子：G · conj(H)
    cv::Mat numerator;
    cv::mulSpectrums(srcSpectrum, psfSpectrum, numerator, 0, true);

    // 分母：|H|² + K（实数）
    std::vector<cv::Mat> psfPlanes;
    cv::split(psfSpectrum, psfPlanes);
    cv::Mat denominator;
    cv::magnitude(psfPlanes[0], psfPlanes[1], denominator);
    denominator = denominator.mul(denominator) + k;

    std::vector<cv::Mat> numPlanes;
    cv::split(numerator, numPlanes);
    numPlanes[0] = numPlanes[0] / denominator;
    numPlanes[1] = numPlanes[1] / denominator;

    cv::Mat filtered;
    cv::merge(numPlanes, filtered);
    cv::Mat restored;
    cv::idft(filtered, restored, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    cv::Mat result;
    restored.convertTo(result, CV_8U);  // 饱和截断到 0~255
    return result;
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
 * @brief  主流程：扫描 PSF 与噪声比，取最优复原结果。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 模糊图像，缺省读取 assets/blurred.png }"
        "{r reference    |      | 清晰参考图，缺省读取 assets/sharp.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/deblur_wiener.png }"
        "{s sigma        | 0    | PSF 标准差，0 表示自动搜索 }"
        "{k K            | 0    | 噪声功率比，0 表示自动搜索 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("离焦去模糊演示: 维纳滤波复原高斯离焦模糊\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("blurred.png");
    }
    if (refPath.empty()) {
        refPath = locateAsset("sharp.png");
    }

    cv::Mat blurred = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (blurred.empty()) {
        std::cerr << "无法读取模糊图像: " << inputPath << std::endl;
        return -1;
    }
    const cv::Mat reference = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
    const bool hasRef = !reference.empty() && reference.size() == blurred.size();

    // 步骤 1~2：确定 PSF。有参考图时直接搜索「让 PSNR 最高」的 sigma 与 K。
    double bestSigma = parser.get<double>("sigma");
    double bestK = parser.get<double>("K");
    cv::Mat bestRestored;
    double bestPsnr = -1.0;

    if (bestSigma > 0.0 && bestK > 0.0) {
        bestRestored = wienerDeblur(blurred, bestSigma, bestK);
        bestPsnr = hasRef ? psnr(bestRestored, reference) : 0.0;
    } else {
        const double sigmas[] = {1.5, 2.0, 2.5, 3.0, 4.0};
        const double ratios[] = {1e-4, 1e-2, 1e-1};
        std::cout << "参数扫描 (sigma, K) -> PSNR(dB):\n";
        for (const double sigma : sigmas) {
            std::cout << "  sigma=" << std::fixed << std::setprecision(1) << std::setw(4) << sigma;
            for (const double k : ratios) {
                const cv::Mat restored = wienerDeblur(blurred, sigma, k);
                const double value = hasRef ? psnr(restored, reference) : 0.0;
                std::cout << "  K=" << std::scientific << std::setprecision(0) << k
                          << std::defaultfloat << std::fixed << ":" << std::setprecision(2)
                          << value;
                if (value > bestPsnr) {
                    bestPsnr = value;
                    bestSigma = sigma;
                    bestK = k;
                    bestRestored = restored;
                }
            }
            std::cout << "\n";
        }
    }

    // 步骤 3：用最优参数复原，并给出一组过激进的参数作为对照。
    if (bestRestored.empty()) {
        bestRestored = wienerDeblur(blurred, std::max(0.5, bestSigma), std::max(1e-6, bestK));
    }
    const cv::Mat overSharp = wienerDeblur(blurred, bestSigma, 1e-5);

    // 步骤 4：评估质量。
    std::cout << "\n输入: " << inputPath << "\n";
    std::cout << "  最优参数: sigma=" << std::fixed << std::setprecision(2) << bestSigma
              << ", K=" << std::scientific << bestK << std::defaultfloat << "\n";
    if (hasRef) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  模糊图 PSNR:   " << psnr(blurred, reference) << " dB\n";
        std::cout << "  复原图 PSNR:   " << psnr(bestRestored, reference) << " dB\n";
        std::cout << "  过激 K=1e-5:   " << psnr(overSharp, reference) << " dB (噪声被放大)\n";
    } else {
        std::cout << "  参考图缺失，跳过 PSNR 评估\n";
    }

    const std::vector<cv::Mat> panels = {
        fitPanel(blurred, "blurred input"),
        fitPanel(bestRestored, "wiener restored"),
        fitPanel(overSharp, "wiener K too small"),
        fitPanel(hasRef ? reference : blurred, "sharp reference")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/deblur_wiener.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("deblur wiener", result);
        cv::waitKey(0);
    }
    return 0;
}
