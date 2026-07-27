// vidicant_c_api.cpp
// C-ABI wrapper for Vidicant media processor

#include "controller.hpp"
#include <cstdlib>
#include <cstring>

extern "C" {

bool vidicant_is_image_file(const char *filename) {
  if (!filename)
    return false;
  return isImageFile(std::string(filename));
}

bool vidicant_is_video_file(const char *filename) {
  if (!filename)
    return false;
  return isVideoFile(std::string(filename));
}

const char *vidicant_process_image(const char *filename) {
  if (!filename)
    return nullptr;
  try {
    nlohmann::json res = processImage(std::string(filename));
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
    nlohmann::json res = processVideo(std::string(filename));
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
