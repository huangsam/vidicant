// controller.cpp
// Implementation file for the media processing controller.
//
// This file contains the implementation of functions for
// media file processing, including file type detection
// and analysis of images and videos.

#include "vidicant/controller.hpp"
#include "vidicant/image.hpp"
#include "vidicant/io/file_detector.hpp"
#include "vidicant/video.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vidicant {

bool isImageFile(const std::string &filename) {
  return io::isImageFile(filename);
}

bool isVideoFile(const std::string &filename) {
  return io::isVideoFile(filename);
}

static nlohmann::json formatImageMetricsJson(const ImageMetrics &m,
                                             const std::string &identifier) {
  nlohmann::json result;
  if (!identifier.empty()) {
    result["filename"] = identifier;
  }

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

  result["top_labels"] = nlohmann::json::array();
  for (const auto &lbl : m.top_labels) {
    result["top_labels"].push_back({
        {"class_id", lbl.class_id},
        {"label", lbl.label},
        {"confidence", lbl.confidence},
    });
  }

  result["detected_objects"] = nlohmann::json::array();
  for (const auto &obj : m.detected_objects) {
    result["detected_objects"].push_back({
        {"box", {obj.box.x, obj.box.y, obj.box.width, obj.box.height}},
        {"class_name", obj.class_name},
        {"confidence", obj.confidence},
    });
  }

  result["embedding"] = m.embedding;

  if (m.ml_evaluated) {
    result["aesthetic_score"] = (m.aesthetic_score >= 0.0)
                                    ? nlohmann::json(m.aesthetic_score)
                                    : nlohmann::json(nullptr);
    result["technical_quality_score"] =
        (m.technical_quality_score >= 0.0)
            ? nlohmann::json(m.technical_quality_score)
            : nlohmann::json(nullptr);
    result["ml_evaluated"] = true;
  } else {
    result["aesthetic_score"] = nullptr;
    result["technical_quality_score"] = nullptr;
    result["ml_evaluated"] = false;
  }

  return result;
}

// Function to process an image file
nlohmann::json processImage(const std::string &filename,
                            const std::string &model_path,
                            const std::string &task, int top_k,
                            float conf_threshold, float nms_threshold) {
  // Load and analyse the image once via a single ImageHandler.
  ImageMetrics m = vidicant::getImageMetrics(filename, model_path, task, top_k,
                                             conf_threshold, nms_threshold);
  return formatImageMetricsJson(m, filename);
}

// Function to process an in-memory image buffer
nlohmann::json processImageBytes(const uint8_t *buffer, size_t len,
                                 const std::string &model_path,
                                 const std::string &task, int top_k,
                                 float conf_threshold, float nms_threshold) {
  ImageMetrics m = vidicant::getImageMetricsFromBuffer(
      buffer, len, model_path, task, top_k, conf_threshold, nms_threshold);
  return formatImageMetricsJson(m, "");
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

} // namespace vidicant
