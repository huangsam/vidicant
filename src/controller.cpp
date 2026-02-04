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
#include <nlohmann/json.hpp>

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
nlohmann::json processImage(const std::string &filename) {
  nlohmann::json result;
  result["filename"] = filename;

  // Get image dimensions
  auto [width, height] = vidicant::getImageDimensions(filename);
  if (width == -1) {
    result["error"] = "Failed to load image";
    return result;
  }

  result["width"] = width;
  result["height"] = height;

  // Additional analyses
  bool grayscale = vidicant::isImageGrayscale(filename);
  result["is_grayscale"] = grayscale;

  double brightness = vidicant::getImageAverageBrightness(filename);
  result["average_brightness"] = brightness;

  int channels = vidicant::getImageNumberOfChannels(filename);
  result["channels"] = channels;

  int edgeCount = vidicant::getImageEdgeCount(filename);
  result["edge_count"] = edgeCount;

  auto dominantColors = vidicant::getImageDominantColors(filename, 3);
  result["dominant_colors"] = nlohmann::json::array();
  for (size_t i = 0; i < dominantColors.size(); ++i) {
    result["dominant_colors"].push_back({
        dominantColors[i][0],
        dominantColors[i][1],
        dominantColors[i][2]
    });
  }

  double blurScore = vidicant::getImageBlurScore(filename);
  result["blur_score"] = blurScore;

  return result;
}

// Function to process a video file
nlohmann::json processVideo(const std::string &filename) {
  nlohmann::json result;
  result["filename"] = filename;

  int frameCount = vidicant::getVideoFrameCount(filename);
  if (frameCount == -1) {
    result["error"] = "Failed to load video";
    return result;
  }

  result["frame_count"] = frameCount;

  double fps = vidicant::getVideoFPS(filename);
  result["fps"] = fps;

  auto [vWidth, vHeight] = vidicant::getVideoResolution(filename);
  result["width"] = vWidth;
  result["height"] = vHeight;

  double duration = vidicant::getVideoDuration(filename);
  result["duration_seconds"] = duration;

  // Advanced video processing
  cv::Mat firstFrame = vidicant::extractFirstFrame(filename);
  if (!firstFrame.empty()) {
    result["first_frame_extracted"] = true;
    result["first_frame_info"] = {
        {"width", firstFrame.cols},
        {"height", firstFrame.rows},
        {"channels", firstFrame.channels()}
    };
  } else {
    result["first_frame_extracted"] = false;
  }

  double videoBrightness = vidicant::getVideoAverageBrightness(filename);
  result["average_brightness"] = videoBrightness;

  bool videoGrayscale = vidicant::isVideoGrayscale(filename);
  result["is_grayscale"] = videoGrayscale;

  // Save first frame as image
  std::filesystem::path videoPath(filename);
  std::string imageOutput = videoPath.stem().string() + "_first_frame.jpg";
  bool saved = vidicant::saveFirstFrameAsImage(filename, imageOutput);
  result["first_frame_saved"] = saved;
  if (saved) {
    result["first_frame_path"] = imageOutput;
  }

  // Motion score
  double motionScore = vidicant::getVideoMotionScore(filename);
  result["motion_score"] = motionScore;

  // Dominant colors from video
  auto videoColors = vidicant::getVideoDominantColors(filename);
  result["dominant_colors"] = nlohmann::json::array();
  for (size_t i = 0; i < videoColors.size(); ++i) {
    result["dominant_colors"].push_back({
        videoColors[i][0],
        videoColors[i][1],
        videoColors[i][2]
    });
  }

  return result;
}
