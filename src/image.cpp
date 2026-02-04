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

bool ImageHandler::isGrayscale(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return false;
  return image.channels() == 1;
}

double ImageHandler::getAverageBrightness(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return -1.0;
  cv::Scalar mean = cv::mean(image);
  if (image.channels() == 1) return mean[0];
  return (mean[0] + mean[1] + mean[2]) / 3.0;
}

int ImageHandler::getNumberOfChannels(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return -1;
  return image.channels();
}

namespace vidicant {

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

} // namespace vidicant
