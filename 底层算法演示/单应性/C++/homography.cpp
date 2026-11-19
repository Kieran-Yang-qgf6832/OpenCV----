/**
 * @file    homography.cpp
 * @brief   单应性演示：特征匹配 + RANSAC 估计 3x3 单应矩阵，用于透视矫正与图像拼接。
 * @details 流程：特征匹配(ORB) -> RANSAC 估计 H -> 透视变换(矫正) -> 图像拼接 -> 输出对比图。
 * @note    依赖：OpenCV 4.x（core / features2d / imgproc / imgcodecs / highgui）
 *          单应矩阵只在「同一平面」或「相机纯旋转」时成立；有视差的三维场景应改用基础矩阵。
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

namespace {

/// 拼接图里每个面板的尺寸（长边等比缩放后居中留白）。
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

/**
 * @brief  把图像等比缩放到固定面板内并居中留白，保证拼接时各面板尺寸一致。
 * @param  image 输入图像（任意尺寸）。
 * @param  title 标题文字（仅支持 ASCII）。
 * @return 固定尺寸的面板。
 */
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
 * @brief  用 ORB 做特征匹配，并用 Lowe 比率筛选。
 * @param  refImage 参考图（模板）。
 * @param  dstImage 目标图（场景）。
 * @param  kpRef    输出：参考图关键点。
 * @param  kpDst    输出：目标图关键点。
 * @param  matches  输出：筛选后的匹配对。
 * @param  ratio    比率阈值，0.75 是常用值。
 */
void matchFeatures(const cv::Mat& refImage, const cv::Mat& dstImage,
                   std::vector<cv::KeyPoint>& kpRef, std::vector<cv::KeyPoint>& kpDst,
                   std::vector<cv::DMatch>& matches, double ratio)
{
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    cv::Mat descRef;
    cv::Mat descDst;
    orb->detectAndCompute(refImage, cv::noArray(), kpRef, descRef);
    orb->detectAndCompute(dstImage, cv::noArray(), kpDst, descDst);
    if (descRef.empty() || descDst.empty()) {
        return;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descRef, descDst, knn, 2);
    for (const auto& pair : knn) {
        // 最近邻明显优于次近邻才保留，可过滤掉大量错误匹配。
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            matches.push_back(pair[0]);
        }
    }
}

/**
 * @brief  打印归一化后的 3x3 单应矩阵。
 * @param  H 单应矩阵。
 * @note   单应矩阵只在相差一个尺度因子的意义下唯一，统一把 h33 归一化为 1 才便于比较。
 */
void printHomography(const cv::Mat& H)
{
    const double scale = H.at<double>(2, 2) != 0.0 ? H.at<double>(2, 2) : 1.0;
    std::cout << "H (归一化 h33=1):\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "  [";
        for (int c = 0; c < 3; ++c) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(6)
                      << H.at<double>(r, c) / scale;
        }
        std::cout << " ]\n";
    }
}

}  // namespace

/**
 * @brief  主流程：匹配、估计 H、透视矫正与拼接。
 * @param  argc 参数个数。
 * @param  argv 参数列表。
 * @return 0 表示成功，非 0 表示失败。
 */
int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? |       | 打印帮助信息 }"
        "{r ref          |       | 参考图，缺省读取 assets/poster.png }"
        "{s scene        |       | 场景图，缺省读取 assets/poster_perspective.png }"
        "{o output       |       | 输出图像路径，缺省写入 <目录>/output/homography.png }"
        "{c ratio        | 0.75  | Lowe 比率阈值 }"
        "{t thresh       | 3.0   | RANSAC 重投影误差阈值(像素) }"
        "{d display      | 0     | 是否弹窗显示 (1=显示, 0=不显示) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("单应性演示: ORB 匹配 + RANSAC + 透视矫正/拼接\n参数请使用 --key=value 形式");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    std::string refPath = parser.get<std::string>("ref");
    std::string scenePath = parser.get<std::string>("scene");
    if (refPath.empty()) {
        refPath = locateAsset("poster.png");
    }
    if (scenePath.empty()) {
        scenePath = locateAsset("poster_perspective.png");
    }

    const cv::Mat refImage = cv::imread(refPath, cv::IMREAD_COLOR);
    const cv::Mat sceneImage = cv::imread(scenePath, cv::IMREAD_COLOR);
    if (refImage.empty() || sceneImage.empty()) {
        std::cerr << "无法读取图像: " << (refImage.empty() ? refPath : scenePath) << std::endl;
        return -1;
    }

    // 步骤 1：特征匹配。
    std::vector<cv::KeyPoint> kpRef;
    std::vector<cv::KeyPoint> kpScene;
    std::vector<cv::DMatch> matches;
    matchFeatures(refImage, sceneImage, kpRef, kpScene, matches, parser.get<double>("ratio"));
    if (matches.size() < 4) {
        std::cerr << "匹配点不足 4 组，无法求解单应矩阵（当前 " << matches.size() << "）" << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> ptsRef;
    std::vector<cv::Point2f> ptsScene;
    for (const auto& m : matches) {
        ptsRef.push_back(kpRef[m.queryIdx].pt);
        ptsScene.push_back(kpScene[m.trainIdx].pt);
    }

    // 步骤 2：RANSAC 估计单应矩阵，mask 标记内点。
    cv::Mat inlierMask;
    const cv::Mat H = cv::findHomography(ptsRef, ptsScene, cv::RANSAC,
                                         parser.get<double>("thresh"), inlierMask);
    if (H.empty()) {
        std::cerr << "RANSAC 求解失败" << std::endl;
        return -1;
    }
    const int inliers = cv::countNonZero(inlierMask);
    std::cout << "匹配数: " << matches.size() << ", RANSAC 内点: " << inliers
              << " (" << std::fixed << std::setprecision(1)
              << 100.0 * inliers / matches.size() << "%)\n";
    printHomography(H);

    // 步骤 3：透视变换。H 是 ref -> scene，矫正场景图要用它的逆矩阵。
    cv::Mat rectified;
    cv::warpPerspective(sceneImage, rectified, H.inv(), refImage.size());

    // 步骤 4：图像拼接。把参考图按 H 投到场景图上，只在有效区域内做 50% 混合。
    cv::Mat warped;
    cv::warpPerspective(refImage, warped, H, sceneImage.size());
    cv::Mat warpedMask;
    cv::cvtColor(warped, warpedMask, cv::COLOR_BGR2GRAY);
    cv::threshold(warpedMask, warpedMask, 1, 255, cv::THRESH_BINARY);
    cv::Mat blended = sceneImage.clone();
    cv::Mat blendedRoi;
    cv::addWeighted(sceneImage, 0.5, warped, 0.5, 0.0, blendedRoi);
    blendedRoi.copyTo(blended, warpedMask);

    // 对比图：全部匹配 vs 仅内点。
    cv::Mat allMatchImage;
    cv::Mat inlierImage;
    cv::drawMatches(refImage, kpRef, sceneImage, kpScene, matches, allMatchImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    const std::vector<char> inlierFlags(inlierMask.begin<uchar>(), inlierMask.end<uchar>());
    cv::drawMatches(refImage, kpRef, sceneImage, kpScene, matches, inlierImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), inlierFlags,
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::vector<cv::Mat> panels = {
        fitPanel(refImage, "reference (planar)"),
        fitPanel(sceneImage, "scene (with perspective)"),
        fitPanel(allMatchImage, "matches: " + std::to_string(matches.size())),
        fitPanel(inlierImage, "inliers: " + std::to_string(inliers)),
        fitPanel(rectified, "rectified by inv(H)"),
        fitPanel(blended, "stitched")};
    const cv::Mat result = montage(panels, 3);

    std::string outputPath = parser.get<std::string>("output");
    if (outputPath.empty()) {
        const std::string outDir = outputDirFor(refPath);
        cv::utils::fs::createDirectories(outDir);
        outputPath = outDir + "/homography.png";
    }
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法写入输出图像: " << outputPath << std::endl;
        return -1;
    }
    std::cout << "结果已保存至 " << outputPath << std::endl;

    if (parser.get<int>("display") != 0) {
        cv::imshow("homography", result);
        cv::waitKey(0);
    }
    return 0;
}
