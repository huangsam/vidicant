// File: file_detector.cpp
// Implementation of media file format detection.

#include "file_detector.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

namespace vidicant::io {

bool isImageFile(const std::string &filename) {
  if (!std::filesystem::exists(filename) ||
      std::filesystem::is_directory(filename)) {
    return false;
  }

  // 1. Fast extension check
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  const std::vector<std::string> imageExtensions = {
      ".jpg", ".jpeg", ".png",  ".bmp",  ".tiff", ".tif",
      ".gif", ".webp", ".heic", ".avif", ".ppm",  ".pgm"};
  if (std::find(imageExtensions.begin(), imageExtensions.end(), ext) !=
      imageExtensions.end()) {
    return true;
  }

  // 2. Magic bytes header inspection
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    unsigned char magic[12] = {0};
    file.read(reinterpret_cast<char *>(magic), sizeof(magic));
    std::streamsize bytesRead = file.gcount();

    if (bytesRead >= 3 && magic[0] == 0xFF && magic[1] == 0xD8 &&
        magic[2] == 0xFF)
      return true; // JPEG
    if (bytesRead >= 4 && magic[0] == 0x89 && magic[1] == 0x50 &&
        magic[2] == 0x4E && magic[3] == 0x47)
      return true; // PNG
    if (bytesRead >= 3 && magic[0] == 'G' && magic[1] == 'I' && magic[2] == 'F')
      return true; // GIF
    if (bytesRead >= 12 && magic[0] == 'R' && magic[1] == 'I' &&
        magic[2] == 'F' && magic[3] == 'F' && magic[8] == 'W' &&
        magic[9] == 'E' && magic[10] == 'B' && magic[11] == 'P')
      return true; // WebP
    if (bytesRead >= 2 && magic[0] == 'B' && magic[1] == 'M')
      return true; // BMP

    // If MP4/MOV ftyp or MKV magic bytes are detected, it's a video, not an
    // image
    if (bytesRead >= 8 && magic[4] == 'f' && magic[5] == 't' &&
        magic[6] == 'y' && magic[7] == 'p')
      return false;
    if (bytesRead >= 4 && magic[0] == 0x1A && magic[1] == 0x45 &&
        magic[2] == 0xDF && magic[3] == 0xA3)
      return false;
  }

  // 3. Fallback OpenCV reader check (only if not a video format)
  const std::vector<std::string> videoExtensions = {
      ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"};
  if (std::find(videoExtensions.begin(), videoExtensions.end(), ext) !=
      videoExtensions.end()) {
    return false;
  }

  return cv::haveImageReader(filename);
}

bool isVideoFile(const std::string &filename) {
  if (!std::filesystem::exists(filename) ||
      std::filesystem::is_directory(filename)) {
    return false;
  }

  // 1. Fast extension check
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  const std::vector<std::string> videoExtensions = {
      ".mp4", ".avi",  ".mov", ".mkv", ".wmv",
      ".flv", ".webm", ".m4v", ".ts",  ".mts"};
  if (std::find(videoExtensions.begin(), videoExtensions.end(), ext) !=
      videoExtensions.end()) {
    return true;
  }

  // 2. Magic bytes check (MP4/MOV ftyp, MKV/WebM 0x1A45DFA3)
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    unsigned char magic[12] = {0};
    file.read(reinterpret_cast<char *>(magic), sizeof(magic));
    std::streamsize bytesRead = file.gcount();

    if (bytesRead >= 8 && magic[4] == 'f' && magic[5] == 't' &&
        magic[6] == 'y' && magic[7] == 'p')
      return true;
    if (bytesRead >= 4 && magic[0] == 0x1A && magic[1] == 0x45 &&
        magic[2] == 0xDF && magic[3] == 0xA3)
      return true;
  }

  // 3. Probe with VideoCapture (fast open & check)
  const std::vector<std::string> imageExtensions = {
      ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"};
  if (std::find(imageExtensions.begin(), imageExtensions.end(), ext) !=
      imageExtensions.end()) {
    return false;
  }

  cv::VideoCapture cap(filename);
  return cap.isOpened() && cap.get(cv::CAP_PROP_FRAME_COUNT) > 0;
}

} // namespace vidicant::io
