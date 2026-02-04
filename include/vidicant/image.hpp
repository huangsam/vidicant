// File: image.hpp
// Header file for image processing functionalities in the Vidicant library.
//
// This file defines interfaces and classes for loading and analyzing images.
// It provides an abstraction layer over image loading mechanisms and offers
// various image analysis functions such as dimension retrieval, color analysis,
// edge detection, and blur scoring.

#ifndef VIDICANT_IMAGE_HPP
#define VIDICANT_IMAGE_HPP

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

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

// Class: ImageHandler
// High-level handler for image analysis operations.
//
// This class provides various methods to analyze images, including
// dimension retrieval, color analysis, edge detection, and blur scoring.
// It uses a loader object to load images internally.
class ImageHandler {
private:
  std::unique_ptr<IImageLoader>
      loader_; // Pointer to the image loader implementation.

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
};

// Namespace: vidicant
// Namespace containing convenience functions for image analysis.
//
// This namespace provides standalone functions that mirror the ImageHandler
// methods, allowing for easier use without creating handler objects.
namespace vidicant {

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

} // namespace vidicant

#endif // VIDICANT_IMAGE_HPP
