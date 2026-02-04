#ifndef VIDICANT_VIDEO_HPP
#define VIDICANT_VIDEO_HPP

#include <opencv2/core.hpp>
#include <string>
#include <utility>

namespace cv {
class Mat;
} // namespace cv

namespace vidicant {

int getVideoFrameCount(const std::string &filename);
double getVideoFPS(const std::string &filename);
std::pair<int, int> getVideoResolution(const std::string &filename);
double getVideoDuration(const std::string &filename);
cv::Mat extractFirstFrame(const std::string &filename);
double getVideoAverageBrightness(const std::string &filename);
bool isVideoGrayscale(const std::string &filename);
bool saveFirstFrameAsImage(const std::string &videoPath, const std::string &imagePath);
double getVideoMotionScore(const std::string &filename);
std::vector<std::array<double, 3>> getVideoDominantColors(const std::string &filename);

} // namespace vidicant

#endif // VIDICANT_VIDEO_HPP