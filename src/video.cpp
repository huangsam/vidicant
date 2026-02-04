#include "vidicant/video.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

bool OpenCVVideoLoader::open(const std::string& filename) {
    cap_.open(filename, cv::CAP_FFMPEG);
    return cap_.isOpened();
}

int OpenCVVideoLoader::getFrameCount() {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
}

double OpenCVVideoLoader::getFPS() {
    return cap_.get(cv::CAP_PROP_FPS);
}

std::pair<int, int> OpenCVVideoLoader::getResolution() {
    int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return {width, height};
}

cv::Mat OpenCVVideoLoader::readFrame() {
    cv::Mat frame;
    cap_ >> frame;
    return frame;
}

VideoHandler::VideoHandler(std::unique_ptr<IVideoLoader> loader) : loader_(std::move(loader)) {}

bool VideoHandler::open(const std::string& filename) {
    filename_ = filename;
    return loader_->open(filename);
}

int VideoHandler::getFrameCount() {
    return loader_->getFrameCount();
}

double VideoHandler::getFPS() {
    return loader_->getFPS();
}

std::pair<int, int> VideoHandler::getResolution() {
    return loader_->getResolution();
}

double VideoHandler::getDuration() {
    int frameCount = loader_->getFrameCount();
    double fps = loader_->getFPS();
    if (fps <= 0) return -1.0;
    return frameCount / fps;
}

cv::Mat VideoHandler::extractFirstFrame() {
    // Create a temporary loader to read the first frame
    auto tempLoader = std::make_unique<OpenCVVideoLoader>();
    if (!tempLoader->open(filename_)) return cv::Mat();
    return tempLoader->readFrame();
}

double VideoHandler::getAverageBrightness() {
    auto tempLoader = std::make_unique<OpenCVVideoLoader>();
    if (!tempLoader->open(filename_)) return -1.0;
    cv::Mat frame;
    double totalBrightness = 0.0;
    int frameCount = 0;
    frame = tempLoader->readFrame();
    while (!frame.empty()) {
        cv::Scalar mean = cv::mean(frame);
        double brightness = (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
        totalBrightness += brightness;
        frameCount++;
        if (frameCount > 100) break;  // Limit to first 100 frames for speed
        frame = tempLoader->readFrame();
    }
    return frameCount > 0 ? totalBrightness / frameCount : -1.0;
}

bool VideoHandler::isGrayscale() {
    cv::Mat frame = extractFirstFrame();
    return frame.channels() == 1;
}

bool VideoHandler::saveFirstFrameAsImage(const std::string& imagePath) {
    cv::Mat frame = extractFirstFrame();
    if (frame.empty()) return false;
    return cv::imwrite(imagePath, frame);
}

double VideoHandler::getMotionScore() {
    auto tempLoader = std::make_unique<OpenCVVideoLoader>();
    if (!tempLoader->open(filename_)) return -1.0;
    cv::Mat prevFrame, currFrame;
    prevFrame = tempLoader->readFrame();
    if (prevFrame.empty()) return 0.0;
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    double totalMotion = 0.0;
    int frameCount = 1;
    currFrame = tempLoader->readFrame();
    while (!currFrame.empty() && frameCount < 50) {  // Limit to 50 frames
        cv::Mat grayCurr;
        cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::absdiff(prevFrame, grayCurr, diff);
        cv::Scalar meanDiff = cv::mean(diff);
        totalMotion += meanDiff[0];
        prevFrame = grayCurr;
        frameCount++;
        currFrame = tempLoader->readFrame();
    }
    return frameCount > 1 ? totalMotion / (frameCount - 1) : 0.0;
}

std::vector<std::array<double, 3>> VideoHandler::getDominantColors() {
    auto tempLoader = std::make_unique<OpenCVVideoLoader>();
    if (!tempLoader->open(filename_)) return {};
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    int count = 0;
    frame = tempLoader->readFrame();
    while (!frame.empty() && count < 10) {  // Sample first 10 frames
        frames.push_back(frame.clone());
        count++;
        frame = tempLoader->readFrame();
    }
    if (frames.empty()) return {};

    // Concatenate all frames into one big image for k-means
    cv::Mat data;
    for (const auto& f : frames) {
        cv::Mat temp;
        f.convertTo(temp, CV_32F);
        temp = temp.reshape(1, temp.total());
        if (data.empty()) {
            data = temp;
        } else {
            cv::vconcat(data, temp, data);
        }
    }

    std::vector<int> labels;
    cv::Mat centers;
    cv::kmeans(data, 3, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<std::array<double, 3>> dominantColors;
    for (int i = 0; i < 3; ++i) {
        dominantColors.push_back({centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2)});
    }
    return dominantColors;
}

namespace vidicant {
int getVideoFrameCount(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return -1;
    return handler.getFrameCount();
}

double getVideoFPS(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return -1.0;
    return handler.getFPS();
}

std::pair<int, int> getVideoResolution(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return {-1, -1};
    return handler.getResolution();
}

double getVideoDuration(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return -1.0;
    return handler.getDuration();
}

cv::Mat extractFirstFrame(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return cv::Mat();
    return handler.extractFirstFrame();
}

double getVideoAverageBrightness(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return -1.0;
    return handler.getAverageBrightness();
}

bool isVideoGrayscale(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return false;
    return handler.isGrayscale();
}

bool saveFirstFrameAsImage(const std::string &videoPath, const std::string &imagePath) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(videoPath)) return false;
    return handler.saveFirstFrameAsImage(imagePath);
}

double getVideoMotionScore(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return -1.0;
    return handler.getMotionScore();
}

std::vector<std::array<double, 3>> getVideoDominantColors(const std::string &filename) {
    VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
    if (!handler.open(filename)) return {};
    return handler.getDominantColors();
}

} // namespace vidicant
