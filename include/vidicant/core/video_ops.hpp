// File: video_ops.hpp
// Core computer vision algorithmic operations for video frames.

#ifndef VIDICANT_CORE_VIDEO_OPS_HPP
#define VIDICANT_CORE_VIDEO_OPS_HPP

#include "vidicant/types.hpp"
#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace vidicant::core {

constexpr int kOpticalFlowMaxPairs = 20;
constexpr int kMaxBrightnessCurveFrames = 100;
constexpr int kVideoCompareSampleCount = 10;

double calculateFrameMotion(const cv::Mat &prevGray, const cv::Mat &currGray);
std::vector<std::array<double, 3>>
extractVideoDominantColors(const std::vector<cv::Mat> &frames, int k = 3);
double calculateColorConsistency(const std::vector<double> &brightnesses);
double calculateOpticalFlowMagnitude(const std::vector<cv::Mat> &frames,
                                     int maxPairs = kOpticalFlowMaxPairs);
ShotLengthStats calculateShotLengthStats(const std::vector<int> &sceneChanges,
                                         int totalFrames);
double calculateFlickerScore(const std::vector<double> &brightnesses);
int evaluateThumbnailFrame(const cv::Mat &frame, int frameIndex,
                           double &bestScore, int &bestIndex);
double compareVideoHistograms(const std::vector<cv::Mat> &frames1,
                              const std::vector<cv::Mat> &frames2,
                              int sampleCount = kVideoCompareSampleCount);
std::string decodeFourcc(double fourccCode);

} // namespace vidicant::core

#endif // VIDICANT_CORE_VIDEO_OPS_HPP
