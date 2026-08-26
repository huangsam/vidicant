// File: main.cpp
// Entry point for the Vidicant CLI utility.

#include "vidicant/cli/dedupe_cmd.hpp"
#include "vidicant/cli/filter.hpp"
#include "vidicant/cli/formatters.hpp"
#include "vidicant/pipeline.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace vidicant;
using namespace vidicant::cli;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout
        << "Usage: " << argv[0]
        << " <file1|dir> [file2] ... [--output <output.json>] "
           "[--format <json|jsonl|csv>] [--stride <int>] [--sample-rate <fps>] "
           "[--export-scenes <dir>] [--filter <expression>] [--enable-ml] "
           "[--model <path.onnx>] [--task <quality|classify|detect|embed>]"
        << std::endl;
    std::cout << "Subcommands:" << std::endl;
    std::cout << "  dedupe <dir> [--threshold <int>] [--output <file>] "
                 "[--format <text|json|jsonl|csv>]"
              << std::endl;
    std::cout
        << "Supported image formats: jpg, jpeg, png, bmp, tiff, tif, gif, webp"
        << std::endl;
    std::cout
        << "Supported video formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v"
        << std::endl;
    return 1;
  }

  // Handle dedupe subcommand
  if (std::string(argv[1]) == "dedupe") {
    return runDedupe(argc - 2, argv + 2);
  }

  std::string outputFile = "";
  std::string format = "json"; // json, jsonl, csv
  std::string modelPath = "";
  std::string task = "quality";
  std::string filterExpr = "";
  std::string exportScenesDir = "";
  int sampleStride = 1;
  double sampleFps = 0.0;
  int topK = 5;
  float confThreshold = 0.5f;
  float nmsThreshold = 0.4f;
  bool enableMl = false;
  std::vector<std::string> inputPaths;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      outputFile = argv[++i];
    } else if ((arg == "--format" || arg == "-f") && i + 1 < argc) {
      format = argv[++i];
      std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    } else if ((arg == "--filter" || arg == "-F") && i + 1 < argc) {
      filterExpr = argv[++i];
    } else if ((arg == "--stride" || arg == "-s") && i + 1 < argc) {
      sampleStride = std::stoi(argv[++i]);
    } else if ((arg == "--sample-rate" || arg == "--sample-fps") &&
               i + 1 < argc) {
      sampleFps = std::stod(argv[++i]);
    } else if (arg == "--export-scenes" && i + 1 < argc) {
      exportScenesDir = argv[++i];
    } else if ((arg == "--model" || arg == "--model-path" || arg == "-m") &&
               i + 1 < argc) {
      modelPath = argv[++i];
      enableMl = true;
    } else if (arg == "--task" && i + 1 < argc) {
      task = argv[++i];
      enableMl = true;
    } else if (arg == "--top-k" && i + 1 < argc) {
      topK = std::stoi(argv[++i]);
    } else if (arg == "--conf-threshold" && i + 1 < argc) {
      confThreshold = std::stof(argv[++i]);
    } else if (arg == "--nms-threshold" && i + 1 < argc) {
      nmsThreshold = std::stof(argv[++i]);
    } else if (arg == "--enable-ml") {
      enableMl = true;
    } else if ((arg == "--image" || arg == "-i" || arg == "--video" ||
                arg == "-v") &&
               i + 1 < argc) {
      inputPaths.push_back(argv[++i]);
    } else if (arg.rfind("-", 0) != 0) {
      inputPaths.push_back(arg);
    }
  }

  // If enable_ml is set but no custom model path was provided, check standard
  // cache
  if (enableMl && modelPath.empty()) {
    const char *home = std::getenv("HOME");
    if (home) {
      std::string defaultModelName = "aesthetic_mobilenetv2.onnx";
      if (task == "classify") {
        defaultModelName = "mobilenetv2_imagenet.onnx";
      } else if (task == "detect") {
        defaultModelName = "yunet_face_detection.onnx";
      }
      fs::path defaultModel =
          fs::path(home) / ".cache" / "vidicant" / "models" / defaultModelName;
      if (fs::exists(defaultModel)) {
        modelPath = defaultModel.string();
      }
    }
  }

  // Expand any directories in input paths
  std::vector<std::string> inputFiles;
  for (const auto &p : inputPaths) {
    if (fs::is_directory(p)) {
      for (const auto &entry : fs::recursive_directory_iterator(
               p, fs::directory_options::skip_permission_denied)) {
        if (entry.is_regular_file()) {
          inputFiles.push_back(entry.path().string());
        }
      }
    } else {
      inputFiles.push_back(p);
    }
  }

  if (outputFile.empty() && format == "json") {
    outputFile = "results.json";
  }

  std::ostream *outStream = &std::cout;
  std::ofstream fileStream;
  if (!outputFile.empty() && outputFile != "-") {
    fileStream.open(outputFile);
    if (!fileStream.is_open()) {
      std::cerr << "Error: Could not open output file: " << outputFile
                << std::endl;
      return 1;
    }
    outStream = &fileStream;
  }

  if (format == "csv") {
    writeCsvHeader(*outStream);
  }

  nlohmann::json results;
  if (format == "json") {
    results["images"] = nlohmann::json::array();
    results["videos"] = nlohmann::json::array();
  }

  VideoAnalysisOptions videoOpts;
  videoOpts.sample_stride = sampleStride;
  videoOpts.sample_fps = sampleFps;
  videoOpts.export_scenes_dir = exportScenesDir;

  for (const auto &filename : inputFiles) {
    if (!fs::exists(filename)) {
      std::cerr << "File does not exist: " << filename << std::endl;
      continue;
    }

    if (isImageFile(filename)) {
      auto imageResult = processImage(filename, modelPath, task, topK,
                                      confThreshold, nmsThreshold);
      if (!filterExpr.empty() &&
          !filter::evaluateFilter(imageResult, filterExpr)) {
        continue;
      }
      if (format == "json") {
        results["images"].push_back(imageResult);
      } else if (format == "jsonl") {
        writeJsonlRecord(*outStream, "image", imageResult);
      } else if (format == "csv") {
        writeCsvRecord(*outStream, "image", imageResult);
      }
    } else if (isVideoFile(filename)) {
      auto videoResult = processVideo(filename, videoOpts);
      if (!filterExpr.empty() &&
          !filter::evaluateFilter(videoResult, filterExpr)) {
        continue;
      }
      if (format == "json") {
        results["videos"].push_back(videoResult);
      } else if (format == "jsonl") {
        writeJsonlRecord(*outStream, "video", videoResult);
      } else if (format == "csv") {
        writeCsvRecord(*outStream, "video", videoResult);
      }
    } else {
      std::cerr << "Unsupported file type: " << filename << std::endl;
    }
  }

  if (format == "json") {
    *outStream << results.dump(2) << std::endl;
  }

  if (fileStream.is_open()) {
    fileStream.close();
    std::cout << "Results written to: " << outputFile << std::endl;
  }

  return 0;
}
