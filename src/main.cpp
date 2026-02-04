#include "vidicant/image.hpp"
#include "vidicant/video.hpp"
#include <opencv2/core.hpp>
#include <iostream>

int main() {
  // Get image dimensions
  auto [width, height] = vidicant::getImageDimensions("examples/sample.jpg");
  if (width == -1) {
    return -1;
  }

  std::cout << "Image width: " << width << std::endl;
  std::cout << "Image height: " << height << std::endl;

  // Additional analyses
  bool grayscale = vidicant::isImageGrayscale("examples/sample.jpg");
  std::cout << "Is grayscale: " << (grayscale ? "Yes" : "No") << std::endl;

  double brightness = vidicant::getImageAverageBrightness("examples/sample.jpg");
  std::cout << "Average brightness: " << brightness << std::endl;

  int channels = vidicant::getImageNumberOfChannels("examples/sample.jpg");
  std::cout << "Number of channels: " << channels << std::endl;

  int edgeCount = vidicant::getImageEdgeCount("examples/sample.jpg");
  std::cout << "Edge count: " << edgeCount << std::endl;

  auto dominantColors = vidicant::getImageDominantColors("examples/sample.jpg", 3);
  std::cout << "Dominant colors (RGB):" << std::endl;
  for (size_t i = 0; i < dominantColors.size(); ++i) {
    std::cout << "  Color " << (i + 1) << ": (" 
              << dominantColors[i][0] << ", " 
              << dominantColors[i][1] << ", " 
              << dominantColors[i][2] << ")" << std::endl;
  }

  double blurScore = vidicant::getImageBlurScore("examples/sample.jpg");
  std::cout << "Blur score (variance): " << blurScore << std::endl;

  // Video analysis
  std::cout << "\n--- Video Analysis ---\n";
  int frameCount = vidicant::getVideoFrameCount("examples/sample.mp4");
  std::cout << "Video frame count: " << frameCount << std::endl;

  double fps = vidicant::getVideoFPS("examples/sample.mp4");
  std::cout << "Video FPS: " << fps << std::endl;

  auto [vWidth, vHeight] = vidicant::getVideoResolution("examples/sample.mp4");
  std::cout << "Video resolution: " << vWidth << "x" << vHeight << std::endl;

  double duration = vidicant::getVideoDuration("examples/sample.mp4");
  std::cout << "Video duration: " << duration << " seconds" << std::endl;

  // Advanced video processing
  cv::Mat firstFrame = vidicant::extractFirstFrame("examples/sample.mp4");
  if (!firstFrame.empty()) {
    std::cout << "First frame extracted: " << firstFrame.cols << "x" << firstFrame.rows << ", channels: " << firstFrame.channels() << std::endl;
  } else {
    std::cout << "Failed to extract first frame" << std::endl;
  }

  double videoBrightness = vidicant::getVideoAverageBrightness("examples/sample.mp4");
  std::cout << "Video average brightness: " << videoBrightness << std::endl;

  bool videoGrayscale = vidicant::isVideoGrayscale("examples/sample.mp4");
  std::cout << "Is video grayscale: " << (videoGrayscale ? "Yes" : "No") << std::endl;

  // Save first frame as image
  bool saved = vidicant::saveFirstFrameAsImage("examples/sample.mp4", "examples/first_frame.jpg");
  std::cout << "Saved first frame as image: " << (saved ? "Yes" : "No") << std::endl;

  // Motion score
  double motionScore = vidicant::getVideoMotionScore("examples/sample.mp4");
  std::cout << "Video motion score: " << motionScore << std::endl;

  // Dominant colors from video
  auto videoColors = vidicant::getVideoDominantColors("examples/sample.mp4");
  std::cout << "Video dominant colors (RGB):" << std::endl;
  for (size_t i = 0; i < videoColors.size(); ++i) {
    std::cout << "  Color " << (i + 1) << ": (" 
              << videoColors[i][0] << ", " 
              << videoColors[i][1] << ", " 
              << videoColors[i][2] << ")" << std::endl;
  }

  return 0;
}
