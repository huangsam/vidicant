// File: pipeline.hpp
// Header file for media processing pipelines and JSON serialization.
//
// This file contains declarations for end-to-end media analysis pipelines,
// format validation, and deduplication orchestration.

#ifndef VIDICANT_PIPELINE_HPP
#define VIDICANT_PIPELINE_HPP

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace vidicant {

// Function to determine if a file is an image based on extension or magic
// bytes.
bool isImageFile(const std::filesystem::path &filename);

// Function to determine if a file is a video based on extension, magic bytes,
// or probe.
bool isVideoFile(const std::filesystem::path &filename);

// Function to process an image file and return JSON result.
nlohmann::json processImage(const std::filesystem::path &filename,
                            const std::filesystem::path &model_path = "",
                            const std::string &task = "quality", int top_k = 5,
                            float conf_threshold = 0.5f,
                            float nms_threshold = 0.4f);

// Function to process an in-memory image buffer and return JSON result.
nlohmann::json processImageBytes(const uint8_t *buffer, size_t len,
                                 const std::filesystem::path &model_path = "",
                                 const std::string &task = "quality",
                                 int top_k = 5, float conf_threshold = 0.5f,
                                 float nms_threshold = 0.4f);

// Function to process a video file and return JSON result.
nlohmann::json processVideo(const std::filesystem::path &filename);

// Function to deduplicate a directory of images and return JSON result.
nlohmann::json dedupeDirectory(const std::filesystem::path &dir,
                               int threshold = 5, bool recursive = true);

} // namespace vidicant

#endif // VIDICANT_PIPELINE_HPP
