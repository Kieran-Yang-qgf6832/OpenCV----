/**
 * @file    linear_transform.cpp
 * @brief   线性变换演示：用增益 alpha 与偏置 beta 调整对比度和亮度。
 * @details 流程：读取灰度图 -> 设定增益和偏置 -> 逐像素变换 -> 裁剪有效范围
 *          -> 拼接对比图，可选滑动条实时调节。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          cv::convertScaleAbs 的语义是 |alpha * x + beta| 再饱和到 0~255，
 *          中间的绝对值会让负增益出现反直觉结果（alpha=-1, beta=0 时图像不变），
 *          本演示专门把这一点和真正的截断版本放在一起对比。
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
 * @brief  标准线性变换：|alpha * x + beta| 后饱和到 0~255。
 * @param  gray  输入灰度图。
 * @param  alpha 增益（对比度）。
 * @param  beta  偏置（亮度）。
 * @return 变换结果。
 */
cv::Mat linearAbs(const cv::Mat& gray, double alpha, double beta)
{
    cv::Mat dst;
    cv::convertScaleAbs(gray, dst, alpha, beta);
    return dst;
}

/**
 * @brief  严格线性变换：alpha * x + beta 直接截断到 0~255（不做绝对值）。
 * @param  gray  输入灰度图。
 * @param  alpha 增益。
 * @param  beta  偏置。
 * @return 变换结果。
 * @note   这才是数学意义上的线性映射；convertScaleAbs 因为取绝对值，负增益会失真。
 */
cv::Mat linearClamped(const cv::Mat& gray, double alpha, double beta)
{
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        const double value = alpha * i + beta;
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(value);  // 负值截断为 0，超过 255 截断为 255
    }
    cv::Mat dst;
    cv::LUT(gray, lut, dst);
    return dst;
}

/**
 * @brief  给单通道图加标题条并转成 BGR。
 * @param  image 单通道图像。
 * @param  title 标题文字（仅支持 ASCII）。
 * @return 带标题条的面板。
 */
cv::Mat makePanel(const cv::Mat& image, const std::string& title)
{
    // 展示时缩到一半，控制拼接图体积。
    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat panel(small.rows + kTitleHeight, small.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::cvtColor(small, panel(cv::Rect(0, kTitleHeight, small.cols, small.rows)),
                 cv::COLOR_GRAY2BGR);
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

/** @brief 打印均值与标准差，量化对比度变化。 */
void printStats(const std::string& name, const cv::Mat& img)
{
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(img, mean, stddev);
    std::cout << "  " << std::left << std::setw(26) << name
              << " mean=" << std::fixed << std::setprecision(1) << std::setw(6) << mean[0]
              << " std=" << std::setw(6) << stddev[0] << "\n";
}

/**
 * @brief  弹窗 + 滑动条实时调节增益与偏置。
 * @param  gray 输入灰度图。
 * @note   仅在 --display=1 时进入；ESC 或 q 退出。
 */
void interactiveTune(const cv::Mat& gray)
{
    const std::string window = "linear transform (ESC/q to quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    int alpha100 = 150;  // 实际 alpha = alpha100 / 100
    int betaOffset = 0;  // 实际 beta = betaOffset - 100
    cv::createTrackbar("alpha x100", window, &alpha100, 400);
    cv::createTrackbar("beta +100", window, &betaOffset, 200);

    while (true) {
        const cv::Mat dst = linearAbs(gray, alpha100 / 100.0, betaOffset - 100.0);
        cv::imshow(window, dst);
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q') {
            break;
        }
    }
}

}  // namespace

/**
 * @brief  主流程：多组 alpha/beta 组合对比，并演示绝对值语义的坑。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/low_contrast.png }"
        "{r reference    |      | 参考图像路径，缺省读取 assets/portrait.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/linear_transform.png }"
        "{a alpha        | 1.6  | 主参数：增益（对比度） }"
        "{b beta         | 0    | 主参数：偏置（亮度） }"
        "{d display      | 0    | 是否弹窗并用滑动条实时调节 }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("线性变换演示: 增益/偏置调整对比度与亮度\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("low_contrast.png");
    }

    // 步骤 1：读取灰度图。
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 步骤 2：设定增益和偏置，并做多组对比。
    const double alpha = parser.get<double>("alpha");
    const double beta = parser.get<double>("beta");

    // 步骤 3：逐像素变换（convertScaleAbs 内部就是 y = |alpha * x + beta| 的饱和版本）。
    const cv::Mat mainResult = linearAbs(gray, alpha, beta);

    // 步骤 4：裁剪有效范围。对比 convertScaleAbs 与严格截断在负增益下的差异。
    const cv::Mat absNeg = linearAbs(gray, -1.0, 0.0);       // 取绝对值：与原图相同
    const cv::Mat clampNeg = linearClamped(gray, -1.0, 0.0);  // 严格截断：全部为 0

    std::cout << "输入: " << inputPath << ", alpha = " << alpha << ", beta = " << beta << "\n";
    printStats("input", gray);
    printStats("alpha 0.5", linearAbs(gray, 0.5, 0.0));
    printStats("alpha " + std::to_string(alpha).substr(0, 4), mainResult);
    printStats("alpha 2.0", linearAbs(gray, 2.0, 0.0));
    printStats("beta -60", linearAbs(gray, 1.0, -60.0));
    printStats("beta +60", linearAbs(gray, 1.0, 60.0));
    printStats("convertScaleAbs a=-1", absNeg);
    printStats("clamped a=-1", clampNeg);

    // 参考图：同一场景的正常曝光版本，用来目测对比度是否已拉回正常水平。
    cv::Mat reference;
    std::string referencePath = parser.get<std::string>("reference");
    if (referencePath.empty()) {
        referencePath = locateAsset("portrait.png");
    }
    const cv::Mat refColor = cv::imread(referencePath, cv::IMREAD_COLOR);
    if (!refColor.empty()) {
        cv::cvtColor(refColor, reference, cv::COLOR_BGR2GRAY);
    } else {
        reference = gray.clone();
    }

    const std::vector<cv::Mat> panels = {
        makePanel(gray, "low contrast input"),
        makePanel(reference, "reference (normal)"),
        makePanel(mainResult, "alpha " + std::to_string(alpha).substr(0, 4) + " beta " +
                                  std::to_string(static_cast<int>(beta))),
        makePanel(linearAbs(gray, 0.5, 0.0), "alpha 0.5"),
        makePanel(linearAbs(gray, 2.0, 0.0), "alpha 2.0"),
        makePanel(linearAbs(gray, 1.0, -60.0), "beta -60"),
        makePanel(linearAbs(gray, 1.0, 60.0), "beta +60"),
        makePanel(absNeg, "convertScaleAbs a=-1 (abs)"),
        makePanel(clampNeg, "clamped a=-1 (black)")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/linear_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    // 步骤 5：实时调节（可选）。
    if (parser.get<int>("display") != 0) {
        interactiveTune(gray);
    }
    return 0;
}
