// controller.hpp
// Header file for the media processing controller.
//
// This file contains declarations for functions that handle
// media file processing logic, including file type detection
// and analysis of images and videos.

#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <string>

// Function to determine if a file is an image based on extension
bool isImageFile(const std::string &filename);

// Function to determine if a file is a video based on extension
bool isVideoFile(const std::string &filename);

// Function to process an image file
void processImage(const std::string &filename);

// Function to process a video file
void processVideo(const std::string &filename);

#endif // CONTROLLER_HPP
