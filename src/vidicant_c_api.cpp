// vidicant_c_api.cpp
// C-ABI wrapper for Vidicant media processor

#include "vidicant/c_api.h"
#include "vidicant/pipeline.hpp"
#include <cstdlib>
#include <cstring>

extern "C" {

bool vidicant_is_image_file(const char *filename) {
  if (!filename)
    return false;
  return vidicant::isImageFile(std::string(filename));
}

bool vidicant_is_video_file(const char *filename) {
  if (!filename)
    return false;
  return vidicant::isVideoFile(std::string(filename));
}

const char *vidicant_process_image(const char *filename) {
  if (!filename)
    return nullptr;
  try {
    nlohmann::json res = vidicant::processImage(std::string(filename));
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

const char *vidicant_process_image_ml(const char *filename,
                                      const char *model_path) {
  if (!filename)
    return nullptr;
  try {
    std::string mp = model_path ? std::string(model_path) : "";
    nlohmann::json res = vidicant::processImage(std::string(filename), mp);
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

const char *vidicant_process_image_dnn(const char *filename,
                                       const char *model_path, const char *task,
                                       int top_k, float conf_threshold,
                                       float nms_threshold) {
  if (!filename)
    return nullptr;
  try {
    std::string mp = model_path ? std::string(model_path) : "";
    std::string tk = task ? std::string(task) : "quality";
    nlohmann::json res = vidicant::processImage(
        std::string(filename), mp, tk, top_k, conf_threshold, nms_threshold);
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

const char *vidicant_process_image_bytes(const uint8_t *buffer, size_t len,
                                         const char *model_path,
                                         const char *task, int top_k,
                                         float conf_threshold,
                                         float nms_threshold) {
  if (!buffer || len == 0)
    return nullptr;
  try {
    std::string mp = model_path ? std::string(model_path) : "";
    std::string tk = task ? std::string(task) : "quality";
    nlohmann::json res = vidicant::processImageBytes(
        buffer, len, mp, tk, top_k, conf_threshold, nms_threshold);
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

const char *vidicant_process_video(const char *filename) {
  if (!filename)
    return nullptr;
  try {
    nlohmann::json res = vidicant::processVideo(std::string(filename));
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

const char *vidicant_dedupe_directory(const char *dir, int threshold,
                                      bool recursive) {
  if (!dir)
    return nullptr;
  try {
    nlohmann::json res =
        vidicant::dedupeDirectory(std::string(dir), threshold, recursive);
    std::string s = res.dump();
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  } catch (...) {
    return nullptr;
  }
}

void vidicant_free_string(const char *str) {
  if (str) {
    std::free(const_cast<char *>(str));
  }
}
}
