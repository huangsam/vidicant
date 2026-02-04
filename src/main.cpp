#include "vidicant/image.hpp"
#include <iostream>

int main() {
  // Get image dimensions
  auto [width, height] = vidicant::getImageDimensions("../examples/sample.jpg");
  if (width == -1) {
    return -1;
  }

  std::cout << "Image width: " << width << std::endl;
  std::cout << "Image height: " << height << std::endl;

  // Additional analyses
  bool grayscale = vidicant::isImageGrayscale("../examples/sample.jpg");
  std::cout << "Is grayscale: " << (grayscale ? "Yes" : "No") << std::endl;

  double brightness = vidicant::getImageAverageBrightness("../examples/sample.jpg");
  std::cout << "Average brightness: " << brightness << std::endl;

  int channels = vidicant::getImageNumberOfChannels("../examples/sample.jpg");
  std::cout << "Number of channels: " << channels << std::endl;

  int edgeCount = vidicant::getImageEdgeCount("../examples/sample.jpg");
  std::cout << "Edge count: " << edgeCount << std::endl;

  auto dominantColors = vidicant::getImageDominantColors("../examples/sample.jpg", 3);
  std::cout << "Dominant colors (RGB):" << std::endl;
  for (size_t i = 0; i < dominantColors.size(); ++i) {
    std::cout << "  Color " << (i + 1) << ": (" 
              << dominantColors[i][0] << ", " 
              << dominantColors[i][1] << ", " 
              << dominantColors[i][2] << ")" << std::endl;
  }

  double blurScore = vidicant::getImageBlurScore("../examples/sample.jpg");
  std::cout << "Blur score (variance): " << blurScore << std::endl;

  return 0;
}
