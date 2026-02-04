// controller.cpp
// Implementation file for the media processing controller.
//
// This file contains the implementation of functions for
// media file processing, including file type detection
// and analysis of images and videos.

#include "controller.hpp"
#include "vidicant/image.hpp"
#include "vidicant/video.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

bool isImageFile(const std::string &filename) {
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::vector<std::string> imageExtensions = {
      ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"};
  return std::find(imageExtensions.begin(), imageExtensions.end(), ext) !=
         imageExtensions.end();
}

// Function to determine if a file is a video based on extension
bool isVideoFile(const std::string &filename) {
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::vector<std::string> videoExtensions = {".mp4", ".avi", ".mov",  ".mkv",
                                              ".wmv", ".flv", ".webm", ".m4v"};
  return std::find(videoExtensions.begin(), videoExtensions.end(), ext) !=
         videoExtensions.end();
}

// Function to process an image file
void processImage(const std::string &filename) {
  std::cout << "\n--- Processing Image: " << filename << " ---\n";

  // Get image dimensions
  auto [width, height] = vidicant::getImageDimensions(filename);
  if (width == -1) {
    std::cout << "Failed to load image: " << filename << std::endl;
    return;
  }

  std::cout << "Image width: " << width << std::endl;
  std::cout << "Image height: " << height << std::endl;

  // Additional analyses
  bool grayscale = vidicant::isImageGrayscale(filename);
  std::cout << "Is grayscale: " << (grayscale ? "Yes" : "No") << std::endl;

  double brightness = vidicant::getImageAverageBrightness(filename);
  std::cout << "Average brightness: " << brightness << std::endl;

  int channels = vidicant::getImageNumberOfChannels(filename);
  std::cout << "Number of channels: " << channels << std::endl;

  int edgeCount = vidicant::getImageEdgeCount(filename);
  std::cout << "Edge count: " << edgeCount << std::endl;

  auto dominantColors = vidicant::getImageDominantColors(filename, 3);
  std::cout << "Dominant colors (RGB):" << std::endl;
  for (size_t i = 0; i < dominantColors.size(); ++i) {
    std::cout << "  Color " << (i + 1) << ": (" << dominantColors[i][0] << ", "
              << dominantColors[i][1] << ", " << dominantColors[i][2] << ")"
              << std::endl;
  }

  double blurScore = vidicant::getImageBlurScore(filename);
  std::cout << "Blur score (variance): " << blurScore << std::endl;
}

// Function to process a video file
void processVideo(const std::string &filename) {
  std::cout << "\n--- Processing Video: " << filename << " ---\n";

  int frameCount = vidicant::getVideoFrameCount(filename);
  if (frameCount == -1) {
    std::cout << "Failed to load video: " << filename << std::endl;
    return;
  }

  std::cout << "Video frame count: " << frameCount << std::endl;

  double fps = vidicant::getVideoFPS(filename);
  std::cout << "Video FPS: " << fps << std::endl;

  auto [vWidth, vHeight] = vidicant::getVideoResolution(filename);
  std::cout << "Video resolution: " << vWidth << "x" << vHeight << std::endl;

  double duration = vidicant::getVideoDuration(filename);
  std::cout << "Video duration: " << duration << " seconds" << std::endl;

  // Advanced video processing
  cv::Mat firstFrame = vidicant::extractFirstFrame(filename);
  if (!firstFrame.empty()) {
    std::cout << "First frame extracted: " << firstFrame.cols << "x"
              << firstFrame.rows << ", channels: " << firstFrame.channels()
              << std::endl;
  } else {
    std::cout << "Failed to extract first frame" << std::endl;
  }

  double videoBrightness = vidicant::getVideoAverageBrightness(filename);
  std::cout << "Video average brightness: " << videoBrightness << std::endl;

  bool videoGrayscale = vidicant::isVideoGrayscale(filename);
  std::cout << "Is video grayscale: " << (videoGrayscale ? "Yes" : "No")
            << std::endl;

  // Save first frame as image
  std::filesystem::path videoPath(filename);
  std::string imageOutput = videoPath.stem().string() + "_first_frame.jpg";
  bool saved = vidicant::saveFirstFrameAsImage(filename, imageOutput);
  std::cout << "Saved first frame as image: " << (saved ? "Yes" : "No")
            << std::endl;

  // Motion score
  double motionScore = vidicant::getVideoMotionScore(filename);
  std::cout << "Video motion score: " << motionScore << std::endl;

  // Dominant colors from video
  auto videoColors = vidicant::getVideoDominantColors(filename);
  std::cout << "Video dominant colors (RGB):" << std::endl;
  for (size_t i = 0; i < videoColors.size(); ++i) {
    std::cout << "  Color " << (i + 1) << ": (" << videoColors[i][0] << ", "
              << videoColors[i][1] << ", " << videoColors[i][2] << ")"
              << std::endl;
  }
}
