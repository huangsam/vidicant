#include "controller.hpp"
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <file1> [file2] [file3] ..."
              << std::endl;
    std::cout
        << "Supported image formats: jpg, jpeg, png, bmp, tiff, tif, gif, webp"
        << std::endl;
    std::cout
        << "Supported video formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v"
        << std::endl;
    return 1;
  }

  for (int i = 1; i < argc; ++i) {
    std::string filename = argv[i];

    if (!std::filesystem::exists(filename)) {
      std::cout << "File does not exist: " << filename << std::endl;
      continue;
    }

    if (isImageFile(filename)) {
      processImage(filename);
    } else if (isVideoFile(filename)) {
      processVideo(filename);
    } else {
      std::cout << "Unsupported file type: " << filename << std::endl;
    }
  }

  return 0;
}
