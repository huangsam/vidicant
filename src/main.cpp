#include "controller.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <file1> [file2] [file3] ... [--output <output.json>] "
                 "[--enable-ml] [--model <path.onnx>]"
              << std::endl;
    std::cout
        << "Supported image formats: jpg, jpeg, png, bmp, tiff, tif, gif, webp"
        << std::endl;
    std::cout
        << "Supported video formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v"
        << std::endl;
    std::cout
        << "Use --output to specify output JSON file (default: results.json)"
        << std::endl;
    std::cout << "Use --enable-ml to enable ONNX aesthetic & quality assessment"
              << std::endl;
    return 1;
  }

  std::string outputFile = "results.json";
  std::string modelPath = "";
  bool enableMl = false;
  std::vector<std::string> inputFiles;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      outputFile = argv[++i];
    } else if ((arg == "--model" || arg == "--model-path" || arg == "-m") &&
               i + 1 < argc) {
      modelPath = argv[++i];
      enableMl = true;
    } else if (arg == "--enable-ml") {
      enableMl = true;
    } else if ((arg == "--image" || arg == "-i" || arg == "--video" ||
                arg == "-v") &&
               i + 1 < argc) {
      inputFiles.push_back(argv[++i]);
    } else if (arg.rfind("-", 0) != 0) {
      inputFiles.push_back(arg);
    }
  }

  // If enable_ml is set but no custom model path was provided, check standard
  // cache
  if (enableMl && modelPath.empty()) {
    const char *home = std::getenv("HOME");
    if (home) {
      std::filesystem::path defaultModel = std::filesystem::path(home) /
                                           ".cache" / "vidicant" / "models" /
                                           "aesthetic_mobilenetv2.onnx";
      if (std::filesystem::exists(defaultModel)) {
        modelPath = defaultModel.string();
      }
    }
  }

  nlohmann::json results;
  results["images"] = nlohmann::json::array();
  results["videos"] = nlohmann::json::array();

  for (const auto &filename : inputFiles) {
    if (!std::filesystem::exists(filename)) {
      std::cout << "File does not exist: " << filename << std::endl;
      continue;
    }

    if (isImageFile(filename)) {
      std::cout << "Processing image: " << filename << std::endl;
      auto imageResult = processImage(filename, modelPath);
      results["images"].push_back(imageResult);
    } else if (isVideoFile(filename)) {
      std::cout << "Processing video: " << filename << std::endl;
      auto videoResult = processVideo(filename);
      results["videos"].push_back(videoResult);
    } else {
      std::cout << "Unsupported file type: " << filename << std::endl;
    }
  }

  // Write results to JSON file
  std::ofstream output(outputFile);
  if (output.is_open()) {
    output << results.dump(2); // Pretty print with 2-space indentation
    output.close();
    std::cout << "Results written to: " << outputFile << std::endl;
  } else {
    std::cerr << "Error: Could not open output file: " << outputFile
              << std::endl;
    return 1;
  }

  return 0;
}
