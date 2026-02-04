#include "vidicant/image.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat OpenCVImageLoader::imread(const std::string &filename) {
  return cv::imread(filename);
}

ImageHandler::ImageHandler(std::unique_ptr<IImageLoader> loader)
    : loader_(std::move(loader)) {}

std::pair<int, int> ImageHandler::getDimensions(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) {
    std::cerr << "Could not open or find the image: " << filename << std::endl;
    return {-1, -1};
  }
  return {image.cols, image.rows};
}

namespace vidicant {

std::pair<int, int> getImageDimensions(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDimensions(filename);
}

} // namespace vidicant
