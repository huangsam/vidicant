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
#include <optional>
#include <thread>

namespace vidicant {

cv::Mat OpenCVImageLoader::imread(const std::filesystem::path &filename) {
  cv::Mat img = cv::imread(filename.string(), cv::IMREAD_UNCHANGED);
  if (!img.empty() && img.depth() != CV_8U) {
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(img, &minVal, &maxVal);
    double scale = (maxVal > minVal) ? (255.0 / (maxVal - minVal)) : 1.0;
    img.convertTo(img, CV_8U, scale, -minVal * scale);
  }
  return img;
}

MemoryImageLoader::MemoryImageLoader(cv::Mat image)
    : image_(std::move(image)) {}

cv::Mat MemoryImageLoader::imread(const std::filesystem::path &) {
  return image_;
}

cv::Mat ImageHandler::loadCached(const std::filesystem::path &filename) {
  std::string key = filename.string();
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second;
  }
  cv::Mat image = loader_->imread(filename);
  if (!image.empty()) {
    cache_[key] = image;
  }
  return image;
}

ImageHandler::ImageHandler(std::unique_ptr<IImageLoader> loader)
    : loader_(std::move(loader)) {}

std::optional<std::pair<int, int>>
ImageHandler::getDimensions(const std::filesystem::path &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty()) {
    std::cerr << "Could not open or find the image: " << filename.string()
              << std::endl;
    return std::nullopt;
  }
  return std::make_pair(image.cols, image.rows);
}

bool ImageHandler::isGrayscale(const std::filesystem::path &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return false;
  return image.channels() == 1;
}

double
ImageHandler::getAverageBrightness(const std::filesystem::path &filename) {
  return core::calculateAverageBrightness(loadCached(filename));
}

std::optional<int>
ImageHandler::getNumberOfChannels(const std::filesystem::path &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return std::nullopt;
  return image.channels();
}

int ImageHandler::getEdgeCount(const std::filesystem::path &filename) {
  return core::calculateEdgeCount(loadCached(filename));
}

std::vector<std::array<double, 3>>
ImageHandler::getDominantColors(const std::filesystem::path &filename, int k) {
  return core::extractDominantColors(loadCached(filename), k);
}

double ImageHandler::getBlurScore(const std::filesystem::path &filename) {
  return core::calculateBlurScore(loadCached(filename));
}

double ImageHandler::getContrastRatio(const std::filesystem::path &filename) {
  return core::calculateContrastRatio(loadCached(filename));
}

double ImageHandler::getSaturationLevel(const std::filesystem::path &filename) {
  return core::calculateSaturationLevel(loadCached(filename));
}

std::vector<std::vector<int>>
ImageHandler::getHistogram(const std::filesystem::path &filename) {
  return core::calculateHistogram(loadCached(filename));
}

double ImageHandler::getAspectRatio(const std::filesystem::path &filename) {
  auto dims = getDimensions(filename);
  if (!dims.has_value() || dims->second <= 0)
    return 0.0;
  return static_cast<double>(dims->first) / dims->second;
}

double ImageHandler::getImageEntropy(const std::filesystem::path &filename) {
  return core::calculateEntropy(loadCached(filename));
}

double ImageHandler::getNoiseEstimate(const std::filesystem::path &filename) {
  return core::calculateNoiseEstimate(loadCached(filename));
}

double ImageHandler::getSymmetryScore(const std::filesystem::path &filename) {
  return core::calculateSymmetryScore(loadCached(filename));
}

TextureFeatures
ImageHandler::getTextureFeatures(const std::filesystem::path &filename) {
  return core::calculateTextureFeatures(loadCached(filename));
}

uint64_t
ImageHandler::getPerceptualHash(const std::filesystem::path &filename) {
  return core::calculatePerceptualHash(loadCached(filename));
}

double
ImageHandler::getWhiteBalanceScore(const std::filesystem::path &filename) {
  return core::calculateWhiteBalanceScore(loadCached(filename));
}

std::vector<int>
ImageHandler::getHueHistogram(const std::filesystem::path &filename) {
  return core::calculateHueHistogram(loadCached(filename));
}

double ImageHandler::getSharpnessScore(const std::filesystem::path &filename) {
  return core::calculateSharpnessScore(loadCached(filename));
}

double ImageHandler::compareImages(const std::filesystem::path &filename1,
                                   const std::filesystem::path &filename2) {
  return core::calculateSSIM(loadCached(filename1), loadCached(filename2));
}

std::string ImageHandler::getNoiseType(const std::filesystem::path &filename) {
  return core::classifyNoiseType(loadCached(filename));
}

void ImageHandler::runDNNInference(const std::filesystem::path &filename,
                                   const std::filesystem::path &model_path,
                                   ImageMetrics &metrics,
                                   const std::string &task, int top_k,
                                   float conf_threshold, float nms_threshold) {
  cv::Mat image = loadCached(filename);
  dnn::DnnEngine::runInference(image, model_path.string(), metrics, task, top_k,
                               conf_threshold, nms_threshold);
}

std::pair<double, double>
ImageHandler::assessQualityDNN(const std::filesystem::path &filename,
                               const std::filesystem::path &model_path) {
  cv::Mat image = loadCached(filename);
  return dnn::DnnEngine::assessQuality(image, model_path.string());
}

std::optional<ImageMetrics>
ImageHandler::getMetrics(const std::filesystem::path &filename,
                         const std::filesystem::path &model_path,
                         const std::string &task, int top_k,
                         float conf_threshold, float nms_threshold) {
  ImageAnalysisOptions options;
  options.model_path = model_path;
  options.task = task;
  options.top_k = top_k;
  options.conf_threshold = conf_threshold;
  options.nms_threshold = nms_threshold;
  return getMetrics(filename, options);
}

std::optional<ImageMetrics>
ImageHandler::getMetrics(const std::filesystem::path &filename,
                         const ImageAnalysisOptions &options) {
  cv::Mat img = loadCached(filename);
  if (img.empty())
    return std::nullopt;

  ImageMetrics m{};
  m.width = img.cols;
  m.height = img.rows;
  m.aspect_ratio =
      (m.height > 0) ? (static_cast<double>(m.width) / m.height) : 0.0;
  m.channels = img.channels();
  m.is_grayscale = (img.channels() == 1);

  // Precompute grayscale Mat once to avoid converting 10+ times in individual
  // ops
  cv::Mat gray = core::toGrayscale(img);

  m.average_brightness = core::calculateAverageBrightness(img);
  m.dominant_colors =
      core::extractDominantColors(img, options.dominant_colors_k);
  m.contrast_ratio = core::calculateContrastRatio(gray);
  m.saturation_level = core::calculateSaturationLevel(img);
  m.histogram = core::calculateHistogram(img);
  m.white_balance_score = core::calculateWhiteBalanceScore(img);
  m.hue_histogram = core::calculateHueHistogram(img);

  m.edge_count = core::calculateEdgeCount(gray);
  m.blur_score = core::calculateBlurScore(gray);
  m.entropy = core::calculateEntropy(gray);
  m.noise_estimate = core::calculateNoiseEstimate(gray);
  m.symmetry_score = core::calculateSymmetryScore(gray);
  m.texture = core::calculateTextureFeatures(gray);
  m.perceptual_hash = core::calculatePerceptualHash(gray);
  m.sharpness_score = core::calculateSharpnessScore(gray);
  m.noise_type = core::classifyNoiseType(gray);

  if (!options.model_path.empty()) {
    runDNNInference(filename, options.model_path, m, options.task,
                    options.top_k, options.conf_threshold,
                    options.nms_threshold);
  } else {
    m.aesthetic_score = -1.0;
    m.technical_quality_score = -1.0;
    m.ml_evaluated = false;
  }

  return m;
}

std::optional<std::pair<int, int>>
getImageDimensions(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDimensions(filename);
}

bool isImageGrayscale(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.isGrayscale(filename);
}

double getImageAverageBrightness(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAverageBrightness(filename);
}

std::optional<int>
getImageNumberOfChannels(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNumberOfChannels(filename);
}

int getImageEdgeCount(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getEdgeCount(filename);
}

std::vector<std::array<double, 3>>
getImageDominantColors(const std::filesystem::path &filename, int k) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDominantColors(filename, k);
}

double getImageBlurScore(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getBlurScore(filename);
}

double getImageContrastRatio(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getContrastRatio(filename);
}

double getImageSaturationLevel(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSaturationLevel(filename);
}

std::vector<std::vector<int>>
getImageHistogram(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHistogram(filename);
}

double getImageAspectRatio(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAspectRatio(filename);
}

double getImageEntropy(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getImageEntropy(filename);
}

double getImageNoiseEstimate(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseEstimate(filename);
}

double getImageSymmetryScore(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSymmetryScore(filename);
}

TextureFeatures getImageTextureFeatures(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getTextureFeatures(filename);
}

uint64_t getImagePerceptualHash(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getPerceptualHash(filename);
}

double getImageWhiteBalanceScore(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getWhiteBalanceScore(filename);
}

std::vector<int> getImageHueHistogram(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHueHistogram(filename);
}

double getImageSharpnessScore(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSharpnessScore(filename);
}

double compareImages(const std::filesystem::path &filename1,
                     const std::filesystem::path &filename2) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.compareImages(filename1, filename2);
}

std::string getImageNoiseType(const std::filesystem::path &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseType(filename);
}

std::pair<double, double>
assessImageQualityDNN(const std::filesystem::path &filename,
                      const std::filesystem::path &model_path) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.assessQualityDNN(filename, model_path);
}

std::optional<ImageMetrics>
getImageMetrics(const std::filesystem::path &filename,
                const std::filesystem::path &model_path,
                const std::string &task, int top_k, float conf_threshold,
                float nms_threshold) {
  ImageAnalysisOptions options;
  options.model_path = model_path;
  options.task = task;
  options.top_k = top_k;
  options.conf_threshold = conf_threshold;
  options.nms_threshold = nms_threshold;
  return getImageMetrics(filename, options);
}

std::optional<ImageMetrics>
getImageMetrics(const std::filesystem::path &filename,
                const ImageAnalysisOptions &options) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getMetrics(filename, options);
}

ImageMetrics getImageMetrics(const cv::Mat &mat,
                             const std::filesystem::path &model_path,
                             const std::string &task, int top_k,
                             float conf_threshold, float nms_threshold) {
  ImageAnalysisOptions options;
  options.model_path = model_path;
  options.task = task;
  options.top_k = top_k;
  options.conf_threshold = conf_threshold;
  options.nms_threshold = nms_threshold;
  return getImageMetrics(mat, options);
}

ImageMetrics getImageMetrics(const cv::Mat &mat,
                             const ImageAnalysisOptions &options) {
  if (mat.empty()) {
    ImageMetrics m{};
    m.width = -1;
    m.height = -1;
    return m;
  }
  auto loader = std::make_unique<MemoryImageLoader>(mat);
  ImageHandler handler(std::move(loader));
  return handler.getMetrics("", options).value_or(ImageMetrics{});
}

ImageMetrics getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                                       const std::filesystem::path &model_path,
                                       const std::string &task, int top_k,
                                       float conf_threshold,
                                       float nms_threshold) {
  ImageAnalysisOptions options;
  options.model_path = model_path;
  options.task = task;
  options.top_k = top_k;
  options.conf_threshold = conf_threshold;
  options.nms_threshold = nms_threshold;
  return getImageMetricsFromBuffer(buffer, len, options);
}

ImageMetrics getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                                       const ImageAnalysisOptions &options) {
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
  return getImageMetrics(decoded, options);
}

std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::filesystem::path> &filenames,
                     const std::filesystem::path &model_path,
                     const std::string &task, int top_k, float conf_threshold,
                     float nms_threshold) {
  ImageAnalysisOptions options;
  options.model_path = model_path;
  options.task = task;
  options.top_k = top_k;
  options.conf_threshold = conf_threshold;
  options.nms_threshold = nms_threshold;
  return getBatchImageMetrics(filenames, options);
}

std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::filesystem::path> &filenames,
                     const ImageAnalysisOptions &options) {
  const size_t maxConcurrency =
      std::max(1u, std::thread::hardware_concurrency());

  std::vector<ImageMetrics> results(filenames.size());

  for (size_t start = 0; start < filenames.size(); start += maxConcurrency) {
    const size_t end = std::min(start + maxConcurrency, filenames.size());

    std::vector<std::future<ImageMetrics>> futures;
    futures.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      const std::filesystem::path fn = filenames[i];
      futures.push_back(std::async(std::launch::async, [fn, options]() {
        auto loader = std::make_unique<OpenCVImageLoader>();
        ImageHandler handler(std::move(loader));
        auto opt = handler.getMetrics(fn, options);
        if (opt.has_value()) {
          return *opt;
        }
        ImageMetrics m{};
        m.width = -1;
        m.height = -1;
        return m;
      }));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      results[start + i] = futures[i].get();
    }
  }
  return results;
}

} // namespace vidicant
