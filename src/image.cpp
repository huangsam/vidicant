#include "vidicant/image.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

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

int ImageHandler::getEdgeCount(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return -1;
  cv::Mat gray, edges;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Canny(gray, edges, 100, 200);
  return cv::countNonZero(edges);
}

std::vector<std::array<double, 3>> ImageHandler::getDominantColors(const std::string &filename, int k) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return {};
  cv::Mat data;
  image.convertTo(data, CV_32F);
  data = data.reshape(1, data.total());

  std::vector<int> labels;
  cv::Mat centers;
  cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

  std::vector<std::array<double, 3>> dominantColors;
  for (int i = 0; i < k; ++i) {
    dominantColors.push_back({centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2)});
  }
  return dominantColors;
}

double ImageHandler::getBlurScore(const std::string &filename) {
  cv::Mat image = loader_->imread(filename);
  if (image.empty()) return -1.0;
  cv::Mat gray, laplacian;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0] * stddev[0];  // Variance
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

int getImageEdgeCount(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getEdgeCount(filename);
}

std::vector<std::array<double, 3>> getImageDominantColors(const std::string &filename, int k) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDominantColors(filename, k);
}

double getImageBlurScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getBlurScore(filename);
}

} // namespace vidicant
