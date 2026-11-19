/**
 * @file    affine_transform.cpp
 * @brief   仿射变换演示：旋转、缩放、平移、剪切与插值方式对比。
 * @details 流程：定义对应点 -> 计算仿射矩阵 -> 应用变换 -> 插值方式对比 -> 输出拼接图。
 * @note    依赖：OpenCV 4.x（core / imgproc / imgcodecs / highgui）
 *          仿射变换只有 6 个自由度，保持平行线平行；要产生消失点必须用 8 自由度的单应矩阵。
 */

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
    const double scale =
        std::min(kPanelWidth / static_cast<double>(image.cols),
                 kPanelHeight / static_cast<double>(image.rows));
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
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

/** @brief 打印 2x3 仿射矩阵。 */
void printMatrix(const std::string& name, const cv::Mat& m)
{
    std::cout << name << ":\n";
    for (int r = 0; r < m.rows; ++r) {
        std::cout << "  [";
        for (int c = 0; c < m.cols; ++c) {
            std::cout << std::setw(11) << std::fixed << std::setprecision(4)
                      << m.at<double>(r, c);
        }
        std::cout << " ]\n";
    }
}

}  // namespace

/**
 * @brief  主流程：五种变换与两种插值方式的对比。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |      | 打印帮助信息 }"
        "{i input        |      | 输入图像路径，缺省读取 assets/scene.png }"
        "{o output       |      | 输出图像路径，缺省写入 <目录>/output/affine_transform.png }"
        "{a angle        | 25   | 旋转角度(度) }"
        "{s scale        | 0.85 | 旋转时同时施加的缩放 }"
        "{d display      | 0    | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("仿射变换演示: 旋转/缩放/平移/剪切与插值对比\n参数请使用 --key=value 形式");
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
        inputPath = locateAsset("scene.png");
    }
    const cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "无法读取图像: " << inputPath << std::endl;
        return -1;
    }
    const cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    const cv::Size size = src.size();

    // 步骤 1：定义对应点。三点不共线即可唯一确定一个仿射矩阵。
    const std::vector<cv::Point2f> srcTri = {
        {100.0f, 100.0f}, {924.0f, 120.0f}, {120.0f, 924.0f}};
    const std::vector<cv::Point2f> dstTri = {
        {180.0f, 160.0f}, {880.0f, 60.0f}, {120.0f, 940.0f}};

    // 步骤 2：计算仿射矩阵。
    const cv::Mat rotation =
        cv::getRotationMatrix2D(center, parser.get<double>("angle"), parser.get<double>("scale"));
    const cv::Mat scaling = cv::getRotationMatrix2D(center, 0.0, 1.5);
    const cv::Mat translation = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 80.0, 0.0, 1.0, -60.0);
    const cv::Mat shear = cv::getAffineTransform(srcTri, dstTri);
    printMatrix("rotation", rotation);
    printMatrix("shear (from 3 point pairs)", shear);

    // 步骤 3：应用变换。
    cv::Mat rotated;
    cv::Mat scaled;
    cv::Mat translated;
    cv::Mat sheared;
    cv::warpAffine(src, rotated, rotation, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));
    cv::warpAffine(src, scaled, scaling, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));
    cv::warpAffine(src, translated, translation, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));
    cv::warpAffine(src, sheared, shear, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));

    // 步骤 4：插值处理。缩小后放大才能看出最近邻的锯齿与双线性的平滑。
    cv::Mat small;
    cv::resize(src, small, cv::Size(), 0.12, 0.12, cv::INTER_NEAREST);
    cv::Mat nearest;
    cv::Mat linear;
    cv::resize(small, nearest, size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(small, linear, size, 0.0, 0.0, cv::INTER_LINEAR);

    // 链式变换：先旋转再平移，等价于矩阵相乘（注意顺序）。
    cv::Mat combined;
    cv::warpAffine(rotated, combined, translation, size, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    const std::vector<cv::Mat> panels = {
        fitPanel(src, "original"),
        fitPanel(rotated, "rotate " + std::to_string(static_cast<int>(parser.get<double>("angle"))) + "deg"),
        fitPanel(scaled, "scale 1.5x"),
        fitPanel(translated, "translate (+80, -60)"),
        fitPanel(sheared, "shear (3 point pairs)"),
        fitPanel(combined, "rotate then translate"),
        fitPanel(nearest, "INTER_NEAREST (12% then back)"),
        fitPanel(linear, "INTER_LINEAR (12% then back)")};
    const cv::Mat result = montage(panels, 4);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(inputPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/affine_transform.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "输入: " << inputPath << " (" << src.cols << "x" << src.rows << ")\n";
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("affine transform", result);
        cv::waitKey(0);
    }
    return 0;
}
