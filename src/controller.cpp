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
#include <nlohmann/json.hpp>
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
nlohmann::json processImage(const std::string &filename) {
  nlohmann::json result;
  result["filename"] = filename;

  // Load and analyse the image once via a single ImageHandler.
  ImageMetrics m = vidicant::getImageMetrics(filename);
  if (m.width == -1) {
    result["error"] = "Failed to load image";
    return result;
  }

  result["width"] = m.width;
  result["height"] = m.height;
  result["is_grayscale"] = m.is_grayscale;
  result["average_brightness"] = m.average_brightness;
  result["channels"] = m.channels;
  result["edge_count"] = m.edge_count;

  result["dominant_colors"] = nlohmann::json::array();
  for (const auto &color : m.dominant_colors) {
    result["dominant_colors"].push_back({color[0], color[1], color[2]});
  }

  result["blur_score"] = m.blur_score;
  result["contrast_ratio"] = m.contrast_ratio;
  result["saturation_level"] = m.saturation_level;
  result["histogram"] = m.histogram;
  result["aspect_ratio"] = m.aspect_ratio;
  result["entropy"] = m.entropy;
  result["noise_estimate"] = m.noise_estimate;
  result["symmetry_score"] = m.symmetry_score;

  result["texture_features"] = {{"contrast", m.texture.contrast},
                                {"energy", m.texture.energy},
                                {"homogeneity", m.texture.homogeneity},
                                {"correlation", m.texture.correlation}};

  result["perceptual_hash"] = m.perceptual_hash;
  result["white_balance_score"] = m.white_balance_score;
  result["hue_histogram"] = m.hue_histogram;
  result["sharpness_score"] = m.sharpness_score;
  result["noise_type"] = m.noise_type;

  return result;
}

// Function to process a video file
nlohmann::json processVideo(const std::string &filename) {
  nlohmann::json result;
  result["filename"] = filename;

  // Validate the file can be opened before the expensive full analysis.
  int frameCount = vidicant::getVideoFrameCount(filename);
  if (frameCount == -1) {
    result["error"] = "Failed to load video";
    return result;
  }

  // Extract and save the first frame (not part of VideoMetrics).
  cv::Mat firstFrame = vidicant::extractFirstFrame(filename);
  if (!firstFrame.empty()) {
    result["first_frame_extracted"] = true;
    result["first_frame_info"] = {{"width", firstFrame.cols},
                                  {"height", firstFrame.rows},
                                  {"channels", firstFrame.channels()}};
  } else {
    result["first_frame_extracted"] = false;
  }

  std::filesystem::path videoPath(filename);
  std::string imageOutput = videoPath.stem().string() + "_first_frame.jpg";
  bool saved = vidicant::saveFirstFrameAsImage(filename, imageOutput);
  result["first_frame_saved"] = saved;
  if (saved) {
    result["first_frame_path"] = imageOutput;
  }

  // Compute all remaining metrics in a single pass via VideoMetrics.
  VideoMetrics m = vidicant::getVideoMetrics(filename);

  result["frame_count"] = m.frame_count;
  result["fps"] = m.fps;
  result["width"] = m.width;
  result["height"] = m.height;
  result["duration_seconds"] = m.duration;
  result["average_brightness"] = m.average_brightness;
  result["is_grayscale"] = m.is_grayscale;
  result["motion_score"] = m.motion_score;

  result["dominant_colors"] = nlohmann::json::array();
  for (const auto &color : m.dominant_colors) {
    result["dominant_colors"].push_back({color[0], color[1], color[2]});
  }

  auto sceneChanges = vidicant::detectVideoSceneChanges(filename);
  result["scene_changes"] = sceneChanges;

  result["frame_rate_stability"] = m.frame_rate_stability;
  result["color_consistency"] = m.color_consistency;
  result["optical_flow_magnitude"] = m.optical_flow_magnitude;
  result["has_audio_track"] = m.has_audio_track;

  result["shot_length_stats"] = {{"mean", m.shot_length_stats.mean},
                                 {"stddev", m.shot_length_stats.stddev},
                                 {"min", m.shot_length_stats.min},
                                 {"max", m.shot_length_stats.max},
                                 {"count", m.shot_length_stats.count}};

  result["flicker_score"] = m.flicker_score;
  result["best_thumbnail_frame"] = m.best_thumbnail_frame;
  result["temporal_brightness_curve"] = m.temporal_brightness_curve;
  result["codec_fourcc"] = m.codec_fourcc;

  return result;
}
