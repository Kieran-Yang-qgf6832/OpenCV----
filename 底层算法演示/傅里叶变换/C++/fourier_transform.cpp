/**
 * @file    fourier_transform.cpp
 * @brief   傅里叶变换演示：离散傅里叶变换、频谱中心化与频域低通/高通滤波。
 * @details 流程：灰度转换 -> 离散傅里叶变换 -> 频谱中心化与对数幅度谱 -> 频域滤波 -> 逆变换。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          幅度谱动态范围极大，必须取 log(1+|F|) 才能看见细节；
 *          dft 在每个维度上按 2、3、5 的幂最快，所以先用 getOptimalDFTSize 补边。
 */

#include <algorithm>
#include <cmath>
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
 * @brief  把零频分量搬到画布中心（四个象限对角互换）。
 * @param  mat 单通道浮点矩阵。
 * @note   dft 的零频在左上角，中心化只是为了人眼看频谱方便，不影响滤波结果。
 */
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
 * @brief  用给定掩膜过滤复数频谱并做逆变换。
 * @param  spectrum 双通道复数频谱（零频在左上角）。
 * @param  mask     单通道浮点掩膜，尺寸与频谱一致。
 * @param  size     原始图像尺寸，用于裁掉补边。
 * @return 滤波后的灰度图（CV_8U）。
 */
cv::Mat filterSpectrum(const cv::Mat& spectrum, const cv::Mat& mask, cv::Size size)
{
    std::vector<cv::Mat> planes;
    cv::split(spectrum, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);

    cv::Mat filtered;
    cv::merge(planes, filtered);
    cv::Mat inverse;
    cv::idft(filtered, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);  // 实数输出并归一化

    cv::Mat result;
    inverse(cv::Rect(0, 0, size.width, size.height)).convertTo(result, CV_8U);
    return result;
}

}  // namespace

/**
 * @brief  主流程：计算频谱、构造低通/高通掩膜并做逆变换。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/texture.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/fourier_transform.png }"
        "{r radius       | 60   | 低通掩膜的截止半径(像素) }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("傅里叶变换演示: DFT / 频谱中心化 / 低通与高通滤波\n参数请使用 --key=value 形式");
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

    // 步骤 1：灰度转换。
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 2：离散傅里叶变换。先补边到最优尺寸，再用双通道浮点矩阵存实部与虚部。
    const int optimalRows = cv::getOptimalDFTSize(gray.rows);
    const int optimalCols = cv::getOptimalDFTSize(gray.cols);
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 0, optimalRows - gray.rows, 0, optimalCols - gray.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    std::vector<cv::Mat> planes = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImage;
    cv::merge(planes, complexImage);
    cv::dft(complexImage, complexImage);

    // 步骤 3：频谱中心化。幅度取对数后才能看到高频细节。
    cv::split(complexImage, planes);
    cv::Mat magnitudeImage;
    cv::magnitude(planes[0], planes[1], magnitudeImage);
    cv::Mat spectrumView;
    magnitudeImage += cv::Scalar::all(1);
    cv::log(magnitudeImage, spectrumView);
    fftShift(spectrumView);  // 只影响显示
    cv::normalize(spectrumView, spectrumView, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 步骤 4：频域滤波。掩膜在中心化坐标系下画圆，再反中心化以对齐零频在左上角的频谱。
    const int radius = std::max(1, parser.get<int>("radius"));
    cv::Mat lowMask(padded.size(), CV_32F, cv::Scalar(0.0f));
    cv::circle(lowMask, cv::Point(optimalCols / 2, optimalRows / 2), radius,
               cv::Scalar::all(1.0f), cv::FILLED, cv::LINE_8);
    fftShift(lowMask);
    const cv::Mat highMask = cv::Scalar::all(1.0) - lowMask;

    const cv::Mat lowPass = filterSpectrum(complexImage, lowMask, gray.size());
    const cv::Mat highPass = filterSpectrum(complexImage, highMask, gray.size());

    const double lowEnergy = cv::sum(lowMask)[0] / lowMask.total();
    std::cout << "输入: " << inputPath << " (" << gray.cols << "x" << gray.rows
              << ") -> 补边后 " << optimalCols << "x" << optimalRows << "\n";
    std::cout << "低通截止半径: " << radius << " 像素, 掩膜能量占比: "
              << lowEnergy * 100.0 << "%\n";

    const std::vector<cv::Mat> panels = {
        fitPanel(gray, "gray input"),
        fitPanel(spectrumView, "log magnitude spectrum"),
        fitPanel(lowPass, "low pass (blur)"),
        fitPanel(highPass, "high pass (edges)")};
    const cv::Mat result = montage(panels, 2);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/fourier_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("fourier transform", result);
        cv::waitKey(0);
    }
    return 0;
}
