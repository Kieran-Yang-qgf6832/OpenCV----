/**
 * @file    motion_deblur.cpp
 * @brief   运动去模糊演示：用直线运动模糊核 + 维纳滤波复原运动模糊图像。
 * @details 流程：构造运动模糊核 -> 扫描角度与长度 -> 维纳滤波复原 -> 用 PSNR 选最优参数。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          与离焦模糊（各向同性高斯核）不同，运动模糊的 PSF 是一条线段，
 *          方向和长度都未知，所以必须二维扫描；核的能量归一化为 1，否则复原图亮度会漂移。
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
const int kKernelSize = 63;

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
 * @brief  构造直线运动模糊核。
 * @param  length 运动长度（像素）。
 * @param  angle  运动方向（度）。
 * @return kKernelSize×kKernelSize 的浮点核，能量归一化为 1。
 */
cv::Mat motionKernel(double length, double angleDeg)
{
    cv::Mat kernel = cv::Mat::zeros(kKernelSize, kKernelSize, CV_32F);
    const cv::Point2f center(kKernelSize / 2.0f, kKernelSize / 2.0f);
    const double rad = angleDeg * CV_PI / 180.0;
    const cv::Point2f offset(static_cast<float>(std::cos(rad) * length / 2.0),
                             static_cast<float>(std::sin(rad) * length / 2.0));
    cv::line(kernel, center - offset, center + offset, cv::Scalar::all(1.0), 1, cv::LINE_AA);
    const double total = cv::sum(kernel)[0];
    if (total > 0.0) {
        kernel /= total;  // 归一化，保证复原图亮度不漂移
    }
    return kernel;
}

/**
 * @brief  维纳滤波复原。
 * @param  gray   模糊灰度图。
 * @param  kernel 空间域 PSF（尺寸可小于图像）。
 * @param  k      噪声功率比。
 * @return 复原结果（CV_8U）。
 */
cv::Mat wiener(const cv::Mat& gray, const cv::Mat& kernel, double k)
{
    cv::Mat srcFloat;
    gray.convertTo(srcFloat, CV_32F);
    cv::Mat srcSpectrum;
    cv::dft(srcFloat, srcSpectrum, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat padded = cv::Mat::zeros(gray.size(), CV_32F);
    const int half = kernel.rows / 2;
    kernel.copyTo(padded(cv::Rect(gray.cols / 2 - half, gray.rows / 2 - half,
                                  kernel.cols, kernel.rows)));
    fftShift(padded);
    cv::Mat psfSpectrum;
    cv::dft(padded, psfSpectrum, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat numerator;
    cv::mulSpectrums(srcSpectrum, psfSpectrum, numerator, 0, true);

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
    restored.convertTo(result, CV_8U);
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
 * @brief  主流程：扫描运动核的角度与长度并取最优复原。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 模糊图像，缺省读取 assets/motion_blurred.png }"
        "{r reference    |      | 清晰参考图，缺省读取 assets/sharp.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/motion_deblur.png }"
        "{k K            | 0.01 | 噪声功率比 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("运动去模糊演示: 运动核扫描 + 维纳滤波\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("motion_blurred.png");
    }
    if (refPath.empty()) {
        refPath = locateAsset("sharp.png");
    }

    const cv::Mat blurred = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (blurred.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    const cv::Mat reference = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
    const bool hasRef = !reference.empty() && reference.size() == blurred.size();
    const double k = std::max(1e-8, parser.get<double>("K"));

    // 步骤 1~2：两阶段扫描运动核。先粗扫定位大致方向，再在附近细化，
    //          否则角度/长度的量化误差会让复原质量比不处理还差。
    double bestPsnr = -1.0;
    double bestAngle = 0.0;
    double bestLength = 10.0;
    cv::Mat bestRestored;

    auto evaluate = [&](double angle, double length) {
        const cv::Mat restored = wiener(blurred, motionKernel(length, angle), k);
        const double value = hasRef ? psnr(restored, reference) : 0.0;
        if (value > bestPsnr) {
            bestPsnr = value;
            bestAngle = angle;
            bestLength = length;
            bestRestored = restored;
        }
    };

    std::cout << "粗扫 (角度步长 15, 长度 10/20/30):\n";
    for (int angle = 0; angle < 180; angle += 15) {
        for (double length : {10.0, 20.0, 30.0}) {
            evaluate(angle, length);
        }
    }
    std::cout << "  粗扫最优: 角度 " << bestAngle << " 度, 长度 " << bestLength
              << " 像素, PSNR " << std::fixed << std::setprecision(2) << bestPsnr << " dB\n";

    const double coarseAngle = bestAngle;
    const double coarseLength = bestLength;
    for (double angle = coarseAngle - 10.0; angle <= coarseAngle + 10.0 + 1e-9; angle += 5.0) {
        for (double length = std::max(2.0, coarseLength - 6.0);
             length <= coarseLength + 6.0 + 1e-9; length += 2.0) {
            evaluate(angle, length);
        }
    }
    std::cout << "  细化后最优: 角度 " << bestAngle << " 度, 长度 " << bestLength
              << " 像素, PSNR " << bestPsnr << " dB\n";

    // 步骤 3~4：用最优核复原并评估。
    if (bestRestored.empty()) {
        bestRestored = wiener(blurred, motionKernel(16.0, 0.0), k);
    }
    std::cout << "\n输入: " << inputPath << "\n";
    std::cout << "  最优运动核: 角度 " << bestAngle << " 度, 长度 " << bestLength << " 像素, K="
              << std::scientific << k << std::defaultfloat << "\n";
    if (hasRef) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  模糊图 PSNR: " << psnr(blurred, reference) << " dB\n";
        std::cout << "  复原图 PSNR: " << psnr(bestRestored, reference) << " dB\n";
    } else {
        std::cout << "  参考图缺失，跳过 PSNR 评估\n";
    }

    const cv::Mat kernelView = motionKernel(bestLength, bestAngle);
    cv::Mat kernelDisplay;
    cv::normalize(kernelView, kernelDisplay, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(kernelDisplay, kernelDisplay, cv::Size(240, 240), 0, 0, cv::INTER_NEAREST);

    const std::vector<cv::Mat> panels = {
        fitPanel(blurred, "motion blurred"),
        fitPanel(bestRestored, "wiener restored"),
        fitPanel(hasRef ? reference : blurred, "sharp reference"),
        fitPanel(kernelDisplay, "estimated kernel")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/motion_deblur.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("motion deblur", result);
        cv::waitKey(0);
    }
    return 0;
}
