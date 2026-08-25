// File: file_detector.hpp
// Media file format detection via extension, magic bytes, and probing.

#ifndef VIDICANT_IO_FILE_DETECTOR_HPP
#define VIDICANT_IO_FILE_DETECTOR_HPP

#include <string>

namespace vidicant::io {

// Checks whether a given path points to an image file.
bool isImageFile(const std::string &filename);

// Checks whether a given path points to a video file.
bool isVideoFile(const std::string &filename);

} // namespace vidicant::io

#endif // VIDICANT_IO_FILE_DETECTOR_HPP
