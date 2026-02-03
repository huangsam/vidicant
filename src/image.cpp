#include "vidicant/image.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace vidicant {

std::pair<int, int> getImageDimensions(const std::string &filename) {
  cv::Mat image = cv::imread(filename);
  if (image.empty()) {
    std::cerr << "Could not open or find the image: " << filename << std::endl;
    return {-1, -1};
  }
  return {image.cols, image.rows};
}

} // namespace vidicant
