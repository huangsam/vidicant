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
#include <memory>
#include <opencv2/core.hpp>
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
  virtual cv::Mat imread(const std::string &filename) = 0;

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
  cv::Mat imread(const std::string &filename) override;
};

// Class: MemoryImageLoader
// Concrete implementation of IImageLoader for in-memory images.
class MemoryImageLoader : public IImageLoader {
private:
  cv::Mat image_;

public:
  explicit MemoryImageLoader(cv::Mat image);
  cv::Mat imread(const std::string &filename) override;
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
  cv::Mat loadCached(const std::string &filename);

public:
  // Constructs an ImageHandler with the specified loader.
  explicit ImageHandler(std::unique_ptr<IImageLoader> loader);

  // Retrieves the dimensions of the image.
  std::pair<int, int> getDimensions(const std::string &filename);

  // Checks if the image is grayscale.
  bool isGrayscale(const std::string &filename);

  // Calculates the average brightness of the image.
  double getAverageBrightness(const std::string &filename);

  // Gets the number of color channels in the image.
  int getNumberOfChannels(const std::string &filename);

  // Counts the number of edges in the image using Canny edge detection.
  int getEdgeCount(const std::string &filename);

  // Extracts the dominant colors from the image using k-means clustering.
  std::vector<std::array<double, 3>>
  getDominantColors(const std::string &filename, int k = 3);

  // Calculates a blur score for the image using Laplacian variance.
  double getBlurScore(const std::string &filename);

  // Calculates the contrast ratio of the image.
  double getContrastRatio(const std::string &filename);

  // Calculates the average saturation of the image.
  double getSaturationLevel(const std::string &filename);

  // Gets the RGB histogram data for the image.
  std::vector<std::vector<int>> getHistogram(const std::string &filename);

  // Calculates the aspect ratio (width/height) of the image.
  double getAspectRatio(const std::string &filename);

  // Calculates the entropy (information content) of the image.
  double getImageEntropy(const std::string &filename);

  // Estimates noise level using the Immerkær Laplacian-based formula.
  double getNoiseEstimate(const std::string &filename);

  // Measures bilateral symmetry via histogram correlation of image halves.
  double getSymmetryScore(const std::string &filename);

  // Computes GLCM-based texture features (contrast, energy, homogeneity,
  // correlation).
  TextureFeatures getTextureFeatures(const std::string &filename);

  // Computes a 64-bit difference hash (dHash) for near-duplicate detection.
  uint64_t getPerceptualHash(const std::string &filename);

  // Estimates white balance quality: returns max channel deviation from the
  // scene mean (0 = perfect balance, higher = stronger color cast).
  double getWhiteBalanceScore(const std::string &filename);

  // Returns a 36-bin hue histogram (5-degree bins over the HSV hue channel).
  std::vector<int> getHueHistogram(const std::string &filename);

  // Calculates mean Sobel gradient magnitude as a sharpness measure.
  double getSharpnessScore(const std::string &filename);

  // Computes the Structural Similarity Index (SSIM) between two images.
  // Returns a value in [-1, 1] where 1 means identical.
  double compareImages(const std::string &filename1,
                       const std::string &filename2);

  // Classifies the dominant noise type as "gaussian" or "salt_and_pepper".
  std::string getNoiseType(const std::string &filename);

  // Evaluates aesthetic and technical quality using an ONNX model via OpenCV
  // DNN. Returns pair of {aesthetic_score (1.0-10.0), technical_quality_score
  // (0.0-1.0)}.
  std::pair<double, double> assessQualityDNN(const std::string &filename,
                                             const std::string &model_path);

  // Runs DNN inference for a specified task ("quality", "classify", "detect",
  // "embed", "auto") and populates neural fields in ImageMetrics.
  void runDNNInference(const std::string &filename,
                       const std::string &model_path, ImageMetrics &metrics,
                       const std::string &task = "quality", int top_k = 5,
                       float conf_threshold = 0.5f, float nms_threshold = 0.4f);

  // Returns an ImageMetrics struct populated with all analyses for the file.
  ImageMetrics getMetrics(const std::string &filename,
                          const std::string &model_path = "",
                          const std::string &task = "quality", int top_k = 5,
                          float conf_threshold = 0.5f,
                          float nms_threshold = 0.4f);
};

// Convenience functions for image analysis.
//
// These standalone functions mirror the ImageHandler methods, allowing for
// direct use without manually constructing handler objects.

// Convenience function to get image dimensions.
std::pair<int, int> getImageDimensions(const std::string &filename);

// Convenience function to check if image is grayscale.
bool isImageGrayscale(const std::string &filename);

// Convenience function to get average brightness.
double getImageAverageBrightness(const std::string &filename);

// Convenience function to get number of channels.
int getImageNumberOfChannels(const std::string &filename);

// Convenience function to get edge count.
int getImageEdgeCount(const std::string &filename);

// Convenience function to get dominant colors.
std::vector<std::array<double, 3>>
getImageDominantColors(const std::string &filename, int k = 3);

// Convenience function to get blur score.
double getImageBlurScore(const std::string &filename);

// Convenience function to get contrast ratio.
double getImageContrastRatio(const std::string &filename);

// Convenience function to get saturation level.
double getImageSaturationLevel(const std::string &filename);

// Convenience function to get histogram.
std::vector<std::vector<int>> getImageHistogram(const std::string &filename);

// Convenience function to get aspect ratio.
double getImageAspectRatio(const std::string &filename);

// Convenience function to get image entropy.
double getImageEntropy(const std::string &filename);

// Convenience function to get noise estimate.
double getImageNoiseEstimate(const std::string &filename);

// Convenience function to get symmetry score.
double getImageSymmetryScore(const std::string &filename);

// Convenience function to get GLCM texture features.
TextureFeatures getImageTextureFeatures(const std::string &filename);

// Convenience function to get perceptual hash.
uint64_t getImagePerceptualHash(const std::string &filename);

// Convenience function to get white balance score.
double getImageWhiteBalanceScore(const std::string &filename);

// Convenience function to get hue histogram.
std::vector<int> getImageHueHistogram(const std::string &filename);

// Convenience function to get sharpness score.
double getImageSharpnessScore(const std::string &filename);

// Convenience function to compare two images via SSIM.
double compareImages(const std::string &filename1,
                     const std::string &filename2);

// Convenience function to classify noise type.
std::string getImageNoiseType(const std::string &filename);

// Convenience function to assess aesthetic & technical quality via ONNX model.
std::pair<double, double> assessImageQualityDNN(const std::string &filename,
                                                const std::string &model_path);

// Convenience function to get all image metrics at once.
ImageMetrics getImageMetrics(const std::string &filename,
                             const std::string &model_path = "",
                             const std::string &task = "quality", int top_k = 5,
                             float conf_threshold = 0.5f,
                             float nms_threshold = 0.4f);

// Convenience function to get all image metrics from an in-memory decoded
// image.
ImageMetrics getImageMetrics(const cv::Mat &mat,
                             const std::string &model_path = "",
                             const std::string &task = "quality", int top_k = 5,
                             float conf_threshold = 0.5f,
                             float nms_threshold = 0.4f);

// Convenience function to decode an image buffer and retrieve its metrics.
ImageMetrics getImageMetricsFromBuffer(const uint8_t *buffer, size_t len,
                                       const std::string &model_path = "",
                                       const std::string &task = "quality",
                                       int top_k = 5,
                                       float conf_threshold = 0.5f,
                                       float nms_threshold = 0.4f);

// Processes a batch of image files in parallel and returns their metrics.
std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::string> &filenames,
                     const std::string &model_path = "",
                     const std::string &task = "quality", int top_k = 5,
                     float conf_threshold = 0.5f, float nms_threshold = 0.4f);

} // namespace vidicant

#endif // VIDICANT_IMAGE_HPP
