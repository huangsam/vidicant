// File: image.hpp
// Header file for image processing functionalities in the Vidicant library.
//
// This file defines interfaces and classes for loading and analyzing images.
// It provides an abstraction layer over image loading mechanisms and offers
// various image analysis functions such as dimension retrieval, color analysis,
// edge detection, and blur scoring.

#ifndef VIDICANT_IMAGE_HPP
#define VIDICANT_IMAGE_HPP

#include "vidicant/types.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vidicant {

// Class: IImageLoader
// Abstract interface for image loading operations.
//
// This interface defines the contract for loading images from files.
// Implementations can use different libraries or methods for image loading.
class IImageLoader {
public:
  // Loads an image from the specified file.
  virtual cv::Mat imread(const std::filesystem::path &filename) = 0;

  // Virtual destructor for proper cleanup of derived classes.
  virtual ~IImageLoader() = default;
};

// Class: OpenCVImageLoader
// Concrete implementation of IImageLoader using OpenCV.
//
// This class uses OpenCV's imread function to load images from files.
class OpenCVImageLoader : public IImageLoader {
public:
  // Loads an image using OpenCV's imread function.
  cv::Mat imread(const std::filesystem::path &filename) override;
};

// Class: MemoryImageLoader
// Concrete implementation of IImageLoader for in-memory images.
class MemoryImageLoader : public IImageLoader {
private:
  cv::Mat image_;

public:
  explicit MemoryImageLoader(cv::Mat image);
  cv::Mat imread(const std::filesystem::path &filename) override;
};

// Class: ImageHandler
// High-level handler for image analysis operations.
//
// This class provides various methods to analyze images, including
// dimension retrieval, color analysis, edge detection, and blur scoring.
// It uses a loader object to load images internally and caches loaded images
// to avoid redundant I/O when multiple analyses are run on the same file.
class ImageHandler {
private:
  std::unique_ptr<IImageLoader>
      loader_; // Pointer to the image loader implementation.
  std::unordered_map<std::string, cv::Mat> cache_; // Per-instance image cache.

  // Loads an image via the loader and caches the result for reuse.
  cv::Mat loadCached(const std::filesystem::path &filename);

public:
  // Constructs an ImageHandler with the specified loader.
  explicit ImageHandler(std::unique_ptr<IImageLoader> loader);

  // Retrieves the dimensions of the image. Returns nullopt on load failure.
  std::optional<std::pair<int, int>>
  getDimensions(const std::filesystem::path &filename);

  // Checks if the image is grayscale.
  bool isGrayscale(const std::filesystem::path &filename);

  // Calculates the average brightness of the image.
  double getAverageBrightness(const std::filesystem::path &filename);

  // Gets the number of color channels in the image. Returns nullopt on load
  // failure.
  std::optional<int> getNumberOfChannels(const std::filesystem::path &filename);

  // Counts the number of edges in the image using Canny edge detection.
  int getEdgeCount(const std::filesystem::path &filename);

  // Extracts the dominant colors from the image using k-means clustering.
  std::vector<std::array<double, 3>>
  getDominantColors(const std::filesystem::path &filename, int k = 3);

  // Calculates a blur score for the image using Laplacian variance.
  double getBlurScore(const std::filesystem::path &filename);

  // Calculates the contrast ratio of the image.
  double getContrastRatio(const std::filesystem::path &filename);

  // Calculates the average saturation of the image.
  double getSaturationLevel(const std::filesystem::path &filename);

  // Gets the RGB histogram data for the image.
  std::vector<std::vector<int>>
  getHistogram(const std::filesystem::path &filename);

  // Calculates the aspect ratio (width/height) of the image.
  double getAspectRatio(const std::filesystem::path &filename);

  // Calculates the entropy (information content) of the image.
  double getImageEntropy(const std::filesystem::path &filename);

  // Estimates noise level using the Immerkær Laplacian-based formula.
  double getNoiseEstimate(const std::filesystem::path &filename);

  // Measures bilateral symmetry via histogram correlation of image halves.
  double getSymmetryScore(const std::filesystem::path &filename);

  // Computes GLCM-based texture features (contrast, energy, homogeneity,
  // correlation).
  TextureFeatures getTextureFeatures(const std::filesystem::path &filename);

  // Computes a 64-bit difference hash (dHash) for near-duplicate detection.
  uint64_t getPerceptualHash(const std::filesystem::path &filename);

  // Estimates white balance quality: returns max channel deviation from the
  // scene mean (0 = perfect balance, higher = stronger color cast).
  double getWhiteBalanceScore(const std::filesystem::path &filename);

  // Returns a 36-bin hue histogram (5-degree bins over the HSV hue channel).
  std::vector<int> getHueHistogram(const std::filesystem::path &filename);

  // Calculates mean Sobel gradient magnitude as a sharpness measure.
  double getSharpnessScore(const std::filesystem::path &filename);

  // Computes the Structural Similarity Index (SSIM) between two images.
  // Returns a value in [-1, 1] where 1 means identical.
  double compareImages(const std::filesystem::path &filename1,
                       const std::filesystem::path &filename2);

  // Classifies the dominant noise type as "gaussian" or "salt_and_pepper".
  std::string getNoiseType(const std::filesystem::path &filename);

  // Evaluates aesthetic and technical quality using an ONNX model via OpenCV
  // DNN. Returns pair of {aesthetic_score (1.0-10.0), technical_quality_score
  // (0.0-1.0)}.
  std::pair<double, double>
  assessQualityDNN(const std::filesystem::path &filename,
                   const std::filesystem::path &model_path);

  // Runs DNN inference for a specified task ("quality", "classify", "detect",
  // "embed", "auto") and populates neural fields in ImageMetrics.
  void runDNNInference(const std::filesystem::path &filename,
                       const std::filesystem::path &model_path,
                       ImageMetrics &metrics,
                       const std::string &task = "quality", int top_k = 5,
                       float conf_threshold = 0.5f, float nms_threshold = 0.4f);

  // Returns an ImageMetrics struct populated with all analyses for the file.
  // Returns nullopt on load failure.
  std::optional<ImageMetrics>
  getMetrics(const std::filesystem::path &filename,
             const std::filesystem::path &model_path = "",
             const std::string &task = "quality", int top_k = 5,
             float conf_threshold = 0.5f, float nms_threshold = 0.4f);

  // Returns an ImageMetrics struct populated with all analyses for the file
  // using options.
  std::optional<ImageMetrics> getMetrics(const std::filesystem::path &filename,
                                         const ImageAnalysisOptions &options);
};

// Convenience functions for image analysis.
//
// These standalone functions mirror the ImageHandler methods, allowing for
// direct use without manually constructing handler objects.

// Convenience function to get image dimensions. Returns nullopt on failure.
std::optional<std::pair<int, int>>
getImageDimensions(const std::filesystem::path &filename);

// Convenience function to check if image is grayscale.
bool isImageGrayscale(const std::filesystem::path &filename);

// Convenience function to get average brightness.
double getImageAverageBrightness(const std::filesystem::path &filename);

// Convenience function to get number of channels. Returns nullopt on failure.
std::optional<int>
getImageNumberOfChannels(const std::filesystem::path &filename);

// Convenience function to get edge count.
int getImageEdgeCount(const std::filesystem::path &filename);

// Convenience function to get dominant colors.
std::vector<std::array<double, 3>>
getImageDominantColors(const std::filesystem::path &filename, int k = 3);

// Convenience function to get blur score.
double getImageBlurScore(const std::filesystem::path &filename);

// Convenience function to get contrast ratio.
double getImageContrastRatio(const std::filesystem::path &filename);

// Convenience function to get saturation level.
double getImageSaturationLevel(const std::filesystem::path &filename);

// Convenience function to get histogram.
std::vector<std::vector<int>>
getImageHistogram(const std::filesystem::path &filename);

// Convenience function to get aspect ratio.
double getImageAspectRatio(const std::filesystem::path &filename);

// Convenience function to get image entropy.
double getImageEntropy(const std::filesystem::path &filename);

// Convenience function to get noise estimate.
double getImageNoiseEstimate(const std::filesystem::path &filename);

// Convenience function to get symmetry score.
double getImageSymmetryScore(const std::filesystem::path &filename);

// Convenience function to get GLCM texture features.
TextureFeatures getImageTextureFeatures(const std::filesystem::path &filename);

// Convenience function to get 64-bit difference hash (dHash).
uint64_t getImagePerceptualHash(const std::filesystem::path &filename);
uint64_t getImagePerceptualHash(const cv::Mat &mat);

// Convenience function to estimate white balance quality score.
double getImageWhiteBalanceScore(const std::filesystem::path &filename);

// Convenience function to get 36-bin hue histogram.
std::vector<int> getImageHueHistogram(const std::filesystem::path &filename);

// Convenience function to get sharpness score (mean Sobel magnitude).
double getImageSharpnessScore(const std::filesystem::path &filename);

// Convenience function to compute SSIM between two images.
double compareImages(const std::filesystem::path &filename1,
                     const std::filesystem::path &filename2);

// Convenience function to classify noise type.
std::string getImageNoiseType(const std::filesystem::path &filename);

// Convenience function to evaluate quality via ONNX DNN model.
std::pair<double, double>
assessImageQualityDNN(const std::filesystem::path &filename,
                      const std::filesystem::path &model_path);

// Convenience function to get all image metrics. Returns nullopt on load
// failure.
std::optional<ImageMetrics>
getImageMetrics(const std::filesystem::path &filename,
                const std::filesystem::path &model_path = "",
                const std::string &task = "quality", int top_k = 5,
                float conf_threshold = 0.5f, float nms_threshold = 0.4f);

// Convenience function to get all image metrics using options.
std::optional<ImageMetrics>
getImageMetrics(const std::filesystem::path &filename,
                const ImageAnalysisOptions &options);

// Convenience function to get all image metrics from an in-memory decoded
// image.
ImageMetrics getImageMetrics(const cv::Mat &mat,
                             const std::filesystem::path &model_path = "",
                             const std::string &task = "quality", int top_k = 5,
                             float conf_threshold = 0.5f,
                             float nms_threshold = 0.4f);

ImageMetrics getImageMetrics(const cv::Mat &mat,
                             const ImageAnalysisOptions &options);

// Convenience function to decode an image buffer and retrieve its metrics.
ImageMetrics
getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                          const std::filesystem::path &model_path = "",
                          const std::string &task = "quality", int top_k = 5,
                          float conf_threshold = 0.5f,
                          float nms_threshold = 0.4f);

ImageMetrics getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                                       const ImageAnalysisOptions &options);

// Processes a batch of image files in parallel and returns their metrics.
std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::filesystem::path> &filenames,
                     const std::filesystem::path &model_path = "",
                     const std::string &task = "quality", int top_k = 5,
                     float conf_threshold = 0.5f, float nms_threshold = 0.4f);

std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::filesystem::path> &filenames,
                     const ImageAnalysisOptions &options);

} // namespace vidicant

#endif // VIDICANT_IMAGE_HPP
