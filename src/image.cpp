// File: image.cpp
// Implementation of image loaders, ImageHandler coordinator, and public image
// functions.

#include "vidicant/image.hpp"
#include "vidicant/core/image_ops.hpp"
#include "vidicant/dnn/dnn_engine.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <thread>

namespace vidicant {

cv::Mat OpenCVImageLoader::imread(const std::string &filename) {
  return cv::imread(filename);
}

MemoryImageLoader::MemoryImageLoader(cv::Mat image)
    : image_(std::move(image)) {}

cv::Mat MemoryImageLoader::imread(const std::string &) { return image_; }

cv::Mat ImageHandler::loadCached(const std::string &filename) {
  auto it = cache_.find(filename);
  if (it != cache_.end()) {
    return it->second;
  }
  cv::Mat image = loader_->imread(filename);
  if (!image.empty()) {
    cache_[filename] = image;
  }
  return image;
}

ImageHandler::ImageHandler(std::unique_ptr<IImageLoader> loader)
    : loader_(std::move(loader)) {}

std::pair<int, int> ImageHandler::getDimensions(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty()) {
    std::cerr << "Could not open or find the image: " << filename << std::endl;
    return {-1, -1};
  }
  return {image.cols, image.rows};
}

bool ImageHandler::isGrayscale(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return false;
  return image.channels() == 1;
}

double ImageHandler::getAverageBrightness(const std::string &filename) {
  return core::calculateAverageBrightness(loadCached(filename));
}

int ImageHandler::getNumberOfChannels(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1;
  return image.channels();
}

int ImageHandler::getEdgeCount(const std::string &filename) {
  return core::calculateEdgeCount(loadCached(filename));
}

std::vector<std::array<double, 3>>
ImageHandler::getDominantColors(const std::string &filename, int k) {
  return core::extractDominantColors(loadCached(filename), k);
}

double ImageHandler::getBlurScore(const std::string &filename) {
  return core::calculateBlurScore(loadCached(filename));
}

double ImageHandler::getContrastRatio(const std::string &filename) {
  return core::calculateContrastRatio(loadCached(filename));
}

double ImageHandler::getSaturationLevel(const std::string &filename) {
  return core::calculateSaturationLevel(loadCached(filename));
}

std::vector<std::vector<int>>
ImageHandler::getHistogram(const std::string &filename) {
  return core::calculateHistogram(loadCached(filename));
}

double ImageHandler::getAspectRatio(const std::string &filename) {
  auto [width, height] = getDimensions(filename);
  return height > 0 ? static_cast<double>(width) / height : 0.0;
}

double ImageHandler::getImageEntropy(const std::string &filename) {
  return core::calculateEntropy(loadCached(filename));
}

double ImageHandler::getNoiseEstimate(const std::string &filename) {
  return core::calculateNoiseEstimate(loadCached(filename));
}

double ImageHandler::getSymmetryScore(const std::string &filename) {
  return core::calculateSymmetryScore(loadCached(filename));
}

TextureFeatures ImageHandler::getTextureFeatures(const std::string &filename) {
  return core::calculateTextureFeatures(loadCached(filename));
}

uint64_t ImageHandler::getPerceptualHash(const std::string &filename) {
  return core::calculatePerceptualHash(loadCached(filename));
}

double ImageHandler::getWhiteBalanceScore(const std::string &filename) {
  return core::calculateWhiteBalanceScore(loadCached(filename));
}

std::vector<int> ImageHandler::getHueHistogram(const std::string &filename) {
  return core::calculateHueHistogram(loadCached(filename));
}

double ImageHandler::getSharpnessScore(const std::string &filename) {
  return core::calculateSharpnessScore(loadCached(filename));
}

double ImageHandler::compareImages(const std::string &filename1,
                                   const std::string &filename2) {
  return core::calculateSSIM(loadCached(filename1), loadCached(filename2));
}

std::string ImageHandler::getNoiseType(const std::string &filename) {
  return core::classifyNoiseType(loadCached(filename));
}

void ImageHandler::runDNNInference(const std::string &filename,
                                   const std::string &model_path,
                                   ImageMetrics &metrics,
                                   const std::string &task, int top_k,
                                   float conf_threshold, float nms_threshold) {
  cv::Mat image = loadCached(filename);
  dnn::DnnEngine::runInference(image, model_path, metrics, task, top_k,
                               conf_threshold, nms_threshold);
}

std::pair<double, double>
ImageHandler::assessQualityDNN(const std::string &filename,
                               const std::string &model_path) {
  cv::Mat image = loadCached(filename);
  return dnn::DnnEngine::assessQuality(image, model_path);
}

ImageMetrics ImageHandler::getMetrics(const std::string &filename,
                                      const std::string &model_path,
                                      const std::string &task, int top_k,
                                      float conf_threshold,
                                      float nms_threshold) {
  ImageMetrics m{};
  auto [w, h] = getDimensions(filename);
  m.width = w;
  m.height = h;
  if (w == -1)
    return m;

  m.is_grayscale = isGrayscale(filename);
  m.average_brightness = getAverageBrightness(filename);
  m.channels = getNumberOfChannels(filename);
  m.edge_count = getEdgeCount(filename);
  m.dominant_colors = getDominantColors(filename);
  m.blur_score = getBlurScore(filename);
  m.contrast_ratio = getContrastRatio(filename);
  m.saturation_level = getSaturationLevel(filename);
  m.histogram = getHistogram(filename);
  m.aspect_ratio = getAspectRatio(filename);
  m.entropy = getImageEntropy(filename);
  m.noise_estimate = getNoiseEstimate(filename);
  m.symmetry_score = getSymmetryScore(filename);
  m.texture = getTextureFeatures(filename);
  m.perceptual_hash = getPerceptualHash(filename);
  m.white_balance_score = getWhiteBalanceScore(filename);
  m.hue_histogram = getHueHistogram(filename);
  m.sharpness_score = getSharpnessScore(filename);
  m.noise_type = getNoiseType(filename);

  if (!model_path.empty()) {
    runDNNInference(filename, model_path, m, task, top_k, conf_threshold,
                    nms_threshold);
  } else {
    m.aesthetic_score = -1.0;
    m.technical_quality_score = -1.0;
    m.ml_evaluated = false;
  }

  return m;
}

std::pair<int, int> getImageDimensions(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDimensions(filename);
}

bool isImageGrayscale(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.isGrayscale(filename);
}

double getImageAverageBrightness(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAverageBrightness(filename);
}

int getImageNumberOfChannels(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNumberOfChannels(filename);
}

int getImageEdgeCount(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getEdgeCount(filename);
}

std::vector<std::array<double, 3>>
getImageDominantColors(const std::string &filename, int k) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDominantColors(filename, k);
}

double getImageBlurScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getBlurScore(filename);
}

double getImageContrastRatio(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getContrastRatio(filename);
}

double getImageSaturationLevel(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSaturationLevel(filename);
}

std::vector<std::vector<int>> getImageHistogram(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHistogram(filename);
}

double getImageAspectRatio(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAspectRatio(filename);
}

double getImageEntropy(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getImageEntropy(filename);
}

double getImageNoiseEstimate(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseEstimate(filename);
}

double getImageSymmetryScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSymmetryScore(filename);
}

TextureFeatures getImageTextureFeatures(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getTextureFeatures(filename);
}

uint64_t getImagePerceptualHash(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getPerceptualHash(filename);
}

double getImageWhiteBalanceScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getWhiteBalanceScore(filename);
}

std::vector<int> getImageHueHistogram(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHueHistogram(filename);
}

double getImageSharpnessScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSharpnessScore(filename);
}

double compareImages(const std::string &filename1,
                     const std::string &filename2) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.compareImages(filename1, filename2);
}

std::string getImageNoiseType(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseType(filename);
}

std::pair<double, double> assessImageQualityDNN(const std::string &filename,
                                                const std::string &model_path) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.assessQualityDNN(filename, model_path);
}

ImageMetrics getImageMetrics(const std::string &filename,
                             const std::string &model_path,
                             const std::string &task, int top_k,
                             float conf_threshold, float nms_threshold) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getMetrics(filename, model_path, task, top_k, conf_threshold,
                            nms_threshold);
}

ImageMetrics getImageMetrics(const cv::Mat &mat, const std::string &model_path,
                             const std::string &task, int top_k,
                             float conf_threshold, float nms_threshold) {
  if (mat.empty()) {
    ImageMetrics m{};
    m.width = -1;
    m.height = -1;
    return m;
  }
  auto loader = std::make_unique<MemoryImageLoader>(mat);
  ImageHandler handler(std::move(loader));
  return handler.getMetrics("", model_path, task, top_k, conf_threshold,
                            nms_threshold);
}

ImageMetrics getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                                       const std::string &model_path,
                                       const std::string &task, int top_k,
                                       float conf_threshold,
                                       float nms_threshold) {
  if (!buffer || len == 0) {
    ImageMetrics m{};
    m.width = -1;
    m.height = -1;
    return m;
  }
  cv::Mat rawData(1, static_cast<int>(len), CV_8UC1,
                  const_cast<uint8_t *>(buffer));
  cv::Mat decoded = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
  if (decoded.empty()) {
    ImageMetrics m{};
    m.width = -1;
    m.height = -1;
    return m;
  }
  return getImageMetrics(decoded, model_path, task, top_k, conf_threshold,
                         nms_threshold);
}

std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::string> &filenames,
                     const std::string &model_path, const std::string &task,
                     int top_k, float conf_threshold, float nms_threshold) {
  const size_t maxConcurrency =
      std::max(1u, std::thread::hardware_concurrency());

  std::vector<ImageMetrics> results(filenames.size());

  for (size_t start = 0; start < filenames.size(); start += maxConcurrency) {
    const size_t end = std::min(start + maxConcurrency, filenames.size());

    std::vector<std::future<ImageMetrics>> futures;
    futures.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      const std::string &fn = filenames[i];
      futures.push_back(
          std::async(std::launch::async, [fn, model_path, task, top_k,
                                          conf_threshold, nms_threshold]() {
            auto loader = std::make_unique<OpenCVImageLoader>();
            ImageHandler handler(std::move(loader));
            return handler.getMetrics(fn, model_path, task, top_k,
                                      conf_threshold, nms_threshold);
          }));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      results[start + i] = futures[i].get();
    }
  }
  return results;
}

} // namespace vidicant
