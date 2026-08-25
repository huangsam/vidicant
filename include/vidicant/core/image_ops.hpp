// File: image_ops.hpp
// Core computer vision algorithmic operations on cv::Mat instances.

#ifndef VIDICANT_CORE_IMAGE_OPS_HPP
#define VIDICANT_CORE_IMAGE_OPS_HPP

#include "vidicant/types.hpp"
#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace vidicant::core {

// Brightness & color
double calculateAverageBrightness(const cv::Mat &image);
std::vector<std::array<double, 3>> extractDominantColors(const cv::Mat &image,
                                                         int k = 3);
double calculateContrastRatio(const cv::Mat &image);
double calculateSaturationLevel(const cv::Mat &image);
std::vector<std::vector<int>> calculateHistogram(const cv::Mat &image);
double calculateWhiteBalanceScore(const cv::Mat &image);
std::vector<int> calculateHueHistogram(const cv::Mat &image);

// Texture, edges & information theory
int calculateEdgeCount(const cv::Mat &image);
double calculateBlurScore(const cv::Mat &image);
double calculateEntropy(const cv::Mat &image);
double calculateNoiseEstimate(const cv::Mat &image);
double calculateSymmetryScore(const cv::Mat &image);
TextureFeatures calculateTextureFeatures(const cv::Mat &image);
double calculateSharpnessScore(const cv::Mat &image);
std::string classifyNoiseType(const cv::Mat &image);

// Perceptual hashing & similarity
uint64_t calculatePerceptualHash(const cv::Mat &image);
double calculateSSIM(const cv::Mat &img1, const cv::Mat &img2);

} // namespace vidicant::core

#endif // VIDICANT_CORE_IMAGE_OPS_HPP
