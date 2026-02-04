#ifndef VIDICANT_VIDEO_HPP
#define VIDICANT_VIDEO_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace cv {
class Mat;
} // namespace cv

class IVideoLoader {
public:
    virtual ~IVideoLoader() = default;
    virtual bool open(const std::string& filename) = 0;
    virtual int getFrameCount() = 0;
    virtual double getFPS() = 0;
    virtual std::pair<int, int> getResolution() = 0;
    virtual cv::Mat readFrame() = 0;
};

class OpenCVVideoLoader : public IVideoLoader {
public:
    bool open(const std::string& filename) override;
    int getFrameCount() override;
    double getFPS() override;
    std::pair<int, int> getResolution() override;
    cv::Mat readFrame() override;
private:
    cv::VideoCapture cap_;
};

class VideoHandler {
private:
    std::unique_ptr<IVideoLoader> loader_;
    std::string filename_;  // Store filename for reopening if needed

public:
    explicit VideoHandler(std::unique_ptr<IVideoLoader> loader);
    bool open(const std::string& filename);
    int getFrameCount();
    double getFPS();
    std::pair<int, int> getResolution();
    double getDuration();
    cv::Mat extractFirstFrame();
    double getAverageBrightness();
    bool isGrayscale();
    bool saveFirstFrameAsImage(const std::string& imagePath);
    double getMotionScore();
    std::vector<std::array<double, 3>> getDominantColors();
};

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
