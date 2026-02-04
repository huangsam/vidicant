#include "vidicant/video.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

namespace vidicant {

int getVideoFrameCount(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return -1;
    }
    return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
}

double getVideoFPS(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return -1.0;
    }
    return cap.get(cv::CAP_PROP_FPS);
}

std::pair<int, int> getVideoResolution(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return {-1, -1};
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return {width, height};
}

double getVideoDuration(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return -1.0;
    }
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) return -1.0;
    return frameCount / fps;
}

cv::Mat extractFirstFrame(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return cv::Mat();
    }
    cv::Mat frame;
    cap >> frame;  // Read first frame
    return frame;
}

double getVideoAverageBrightness(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return -1.0;
    }
    cv::Mat frame;
    double totalBrightness = 0.0;
    int frameCount = 0;
    while (cap.read(frame)) {
        cv::Scalar mean = cv::mean(frame);
        double brightness = (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
        totalBrightness += brightness;
        frameCount++;
        if (frameCount > 100) break;  // Limit to first 100 frames for speed
    }
    return frameCount > 0 ? totalBrightness / frameCount : -1.0;
}

bool isVideoGrayscale(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return false;
    }
    cv::Mat frame;
    cap >> frame;  // Check first frame
    return frame.channels() == 1;
}

bool saveFirstFrameAsImage(const std::string &videoPath, const std::string &imagePath) {
    cv::Mat frame = extractFirstFrame(videoPath);
    if (frame.empty()) return false;
    return cv::imwrite(imagePath, frame);
}

double getVideoMotionScore(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return -1.0;
    }
    cv::Mat prevFrame, currFrame;
    cap >> prevFrame;
    if (prevFrame.empty()) return 0.0;
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    double totalMotion = 0.0;
    int frameCount = 1;
    while (cap.read(currFrame) && frameCount < 50) {  // Limit to 50 frames
        cv::Mat grayCurr;
        cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::absdiff(prevFrame, grayCurr, diff);
        cv::Scalar meanDiff = cv::mean(diff);
        totalMotion += meanDiff[0];
        prevFrame = grayCurr;
        frameCount++;
    }
    return frameCount > 1 ? totalMotion / (frameCount - 1) : 0.0;
}

std::vector<std::array<double, 3>> getVideoDominantColors(const std::string &filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << filename << std::endl;
        return {};
    }
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    int count = 0;
    while (cap.read(frame) && count < 10) {  // Sample first 10 frames
        frames.push_back(frame.clone());
        count++;
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

} // namespace vidicant