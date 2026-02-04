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

class IImageLoader {
public:
  virtual cv::Mat imread(const std::string &filename) = 0;
  virtual ~IImageLoader() = default;
};

class OpenCVImageLoader : public IImageLoader {
public:
  cv::Mat imread(const std::string &filename) override;
};

class ImageHandler {
private:
  std::unique_ptr<IImageLoader> loader_;

public:
  explicit ImageHandler(std::unique_ptr<IImageLoader> loader);
  std::pair<int, int> getDimensions(const std::string &filename);
  bool isGrayscale(const std::string &filename);
  double getAverageBrightness(const std::string &filename);
  int getNumberOfChannels(const std::string &filename);
  int getEdgeCount(const std::string &filename);
  std::vector<std::array<double, 3>> getDominantColors(const std::string &filename, int k = 3);
  double getBlurScore(const std::string &filename);
};

namespace vidicant {

std::pair<int, int> getImageDimensions(const std::string &filename);
bool isImageGrayscale(const std::string &filename);
double getImageAverageBrightness(const std::string &filename);
int getImageNumberOfChannels(const std::string &filename);
int getImageEdgeCount(const std::string &filename);
std::vector<std::array<double, 3>> getImageDominantColors(const std::string &filename, int k = 3);
double getImageBlurScore(const std::string &filename);

} // namespace vidicant

#endif // VIDICANT_IMAGE_HPP
