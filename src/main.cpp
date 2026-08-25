#include "controller.hpp"
#include "vidicant/image.hpp"
#include "vidicant/video.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int hammingDistance(uint64_t a, uint64_t b) {
  return static_cast<int>(std::bitset<64>(a ^ b).count());
}

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DisjointSet(int n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int i) {
    if (parent[i] == i)
      return i;
    return parent[i] = find(parent[i]);
  }

  void unite(int i, int j) {
    int root_i = find(i);
    int root_j = find(j);
    if (root_i != root_j) {
      if (rank[root_i] < rank[root_j])
        std::swap(root_i, root_j);
      parent[root_j] = root_i;
      if (rank[root_i] == rank[root_j])
        rank[root_i]++;
    }
  }
};

static std::string escapeCsv(const std::string &str) {
  if (str.find(',') != std::string::npos ||
      str.find('"') != std::string::npos ||
      str.find('\n') != std::string::npos) {
    std::string res = "\"";
    for (char c : str) {
      if (c == '"')
        res += "\"\"";
      else
        res += c;
    }
    res += "\"";
    return res;
  }
  return str;
}

static int runDedupe(int argc, char *argv[]) {
  std::string targetDir = "";
  int threshold = 5;
  std::string outputFile = "";
  std::string format = "text"; // text, json, jsonl, csv

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--threshold" || arg == "-t") && i + 1 < argc) {
      threshold = std::stoi(argv[++i]);
    } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      outputFile = argv[++i];
    } else if ((arg == "--format" || arg == "-f") && i + 1 < argc) {
      format = argv[++i];
      std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    } else if (arg.rfind("-", 0) != 0 && targetDir.empty()) {
      targetDir = arg;
    }
  }

  if (targetDir.empty() || !fs::exists(targetDir)) {
    std::cerr << "Error: Valid target directory required for dedupe."
              << std::endl;
    std::cerr << "Usage: vidicant_cli dedupe <dir> [--threshold <int>] "
                 "[--output <file>] [--format <text|json|jsonl|csv>]"
              << std::endl;
    return 1;
  }

  std::vector<std::string> imageFiles;
  try {
    for (const auto &entry : fs::recursive_directory_iterator(
             targetDir, fs::directory_options::skip_permission_denied)) {
      if (entry.is_regular_file() && isImageFile(entry.path().string())) {
        imageFiles.push_back(entry.path().string());
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error scanning directory: " << e.what() << std::endl;
    return 1;
  }

  std::sort(imageFiles.begin(), imageFiles.end());

  struct ImageHash {
    std::string path;
    uint64_t hash;
  };

  std::vector<ImageHash> hashes;
  hashes.reserve(imageFiles.size());

  for (const auto &img : imageFiles) {
    uint64_t h = vidicant::getImagePerceptualHash(img);
    hashes.push_back({img, h});
  }

  int n = static_cast<int>(hashes.size());
  DisjointSet dsu(n);

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      int dist = hammingDistance(hashes[i].hash, hashes[j].hash);
      if (dist <= threshold) {
        dsu.unite(i, j);
      }
    }
  }

  std::map<int, std::vector<int>> clustersMap;
  for (int i = 0; i < n; ++i) {
    clustersMap[dsu.find(i)].push_back(i);
  }

  struct DuplicateCluster {
    int lead_index;
    std::vector<int> members;
  };

  std::vector<DuplicateCluster> duplicateClusters;
  for (auto const &[root, members] : clustersMap) {
    if (members.size() >= 2) {
      duplicateClusters.push_back({root, members});
    }
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

  if (format == "json") {
    nlohmann::json j;
    j["threshold"] = threshold;
    j["total_images"] = n;
    j["clusters_count"] = duplicateClusters.size();
    j["duplicate_clusters"] = nlohmann::json::array();

    int clusterId = 1;
    for (const auto &cl : duplicateClusters) {
      nlohmann::json clJson;
      clJson["cluster_id"] = clusterId++;
      clJson["count"] = cl.members.size();
      clJson["lead_image"] = hashes[cl.lead_index].path;
      clJson["files"] = nlohmann::json::array();

      for (int idx : cl.members) {
        int dist =
            hammingDistance(hashes[cl.lead_index].hash, hashes[idx].hash);
        clJson["files"].push_back({{"path", hashes[idx].path},
                                   {"perceptual_hash", hashes[idx].hash},
                                   {"distance_to_lead", dist}});
      }
      j["duplicate_clusters"].push_back(clJson);
    }
    *outStream << j.dump(2) << std::endl;
  } else if (format == "jsonl") {
    int clusterId = 1;
    for (const auto &cl : duplicateClusters) {
      nlohmann::json clJson;
      clJson["cluster_id"] = clusterId++;
      clJson["count"] = cl.members.size();
      clJson["lead_image"] = hashes[cl.lead_index].path;
      clJson["files"] = nlohmann::json::array();

      for (int idx : cl.members) {
        int dist =
            hammingDistance(hashes[cl.lead_index].hash, hashes[idx].hash);
        clJson["files"].push_back({{"path", hashes[idx].path},
                                   {"perceptual_hash", hashes[idx].hash},
                                   {"distance_to_lead", dist}});
      }
      *outStream << clJson.dump() << std::endl;
    }
  } else if (format == "csv") {
    *outStream << "cluster_id,path,perceptual_hash,distance_to_lead,is_lead\n";
    int clusterId = 1;
    for (const auto &cl : duplicateClusters) {
      for (int idx : cl.members) {
        int dist =
            hammingDistance(hashes[cl.lead_index].hash, hashes[idx].hash);
        bool isLead = (idx == cl.lead_index);
        *outStream << clusterId << "," << escapeCsv(hashes[idx].path) << ","
                   << hashes[idx].hash << "," << dist << ","
                   << (isLead ? "true" : "false") << "\n";
      }
      clusterId++;
    }
  } else {
    // Human readable text
    *outStream << "Scanned " << n << " images in '" << targetDir << "'."
               << std::endl;
    *outStream << "Found " << duplicateClusters.size()
               << " duplicate/near-duplicate cluster(s) (threshold <= "
               << threshold << "):" << std::endl
               << std::endl;

    int clusterId = 1;
    for (const auto &cl : duplicateClusters) {
      *outStream << "Cluster #" << clusterId++ << " (" << cl.members.size()
                 << " files):" << std::endl;
      for (int idx : cl.members) {
        int dist =
            hammingDistance(hashes[cl.lead_index].hash, hashes[idx].hash);
        if (idx == cl.lead_index) {
          *outStream << "  - [LEAD] " << hashes[idx].path
                     << " (hash: " << hashes[idx].hash << ")" << std::endl;
        } else {
          *outStream << "  - " << hashes[idx].path
                     << " (hash: " << hashes[idx].hash << ", dist: " << dist
                     << ")" << std::endl;
        }
      }
      *outStream << std::endl;
    }
  }

  if (fileStream.is_open()) {
    fileStream.close();
    std::cout << "Deduplication results written to: " << outputFile
              << std::endl;
  }

  return 0;
}

static void writeJsonlRecord(std::ostream &os, const std::string &type,
                             const nlohmann::json &res) {
  nlohmann::json line = res;
  line["media_type"] = type;
  os << line.dump() << std::endl;
}

static void writeCsvHeader(std::ostream &os) {
  os << "filename,media_type,width,height,channels,is_grayscale,average_"
        "brightness,blur_score,contrast_ratio,saturation_level,aspect_ratio,"
        "entropy,noise_estimate,perceptual_hash,ml_evaluated,aesthetic_score,"
        "technical_quality_score,duration_seconds,frame_count,fps,motion_"
        "score\n";
}

static void writeCsvRecord(std::ostream &os, const std::string &type,
                           const nlohmann::json &res) {
  std::string fn = res.value("filename", "");
  os << escapeCsv(fn) << "," << type << ",";

  if (type == "image") {
    os << res.value("width", 0) << "," << res.value("height", 0) << ","
       << res.value("channels", 0) << ","
       << (res.value("is_grayscale", false) ? "true" : "false") << ","
       << res.value("average_brightness", 0.0) << ","
       << res.value("blur_score", 0.0) << ","
       << res.value("contrast_ratio", 0.0) << ","
       << res.value("saturation_level", 0.0) << ","
       << res.value("aspect_ratio", 0.0) << "," << res.value("entropy", 0.0)
       << "," << res.value("noise_estimate", 0.0) << ","
       << res.value("perceptual_hash", 0ULL) << ","
       << (res.value("ml_evaluated", false) ? "true" : "false") << ",";

    if (res.contains("aesthetic_score") && !res["aesthetic_score"].is_null()) {
      os << res["aesthetic_score"].get<double>() << ",";
    } else {
      os << ",";
    }

    if (res.contains("technical_quality_score") &&
        !res["technical_quality_score"].is_null()) {
      os << res["technical_quality_score"].get<double>() << ",";
    } else {
      os << ",";
    }

    os << ",,,"; // Video fields empty
  } else {
    // Video
    os << res.value("width", 0) << "," << res.value("height", 0) << ",,"
       << (res.value("is_grayscale", false) ? "true" : "false") << ","
       << res.value("average_brightness", 0.0) << ",,,,,,,,false,,,";
    os << res.value("duration_seconds", 0.0) << ","
       << res.value("frame_count", 0) << "," << res.value("fps", 0.0) << ","
       << res.value("motion_score", 0.0);
  }
  os << "\n";
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <file1|dir> [file2] ... [--output <output.json>] "
                 "[--format <json|jsonl|csv>] [--enable-ml] [--model "
                 "<path.onnx>] [--task <quality|classify|detect|embed>]"
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

  for (const auto &filename : inputFiles) {
    if (!fs::exists(filename)) {
      std::cerr << "File does not exist: " << filename << std::endl;
      continue;
    }

    if (isImageFile(filename)) {
      auto imageResult = processImage(filename, modelPath, task, topK,
                                      confThreshold, nmsThreshold);
      if (format == "json") {
        results["images"].push_back(imageResult);
      } else if (format == "jsonl") {
        writeJsonlRecord(*outStream, "image", imageResult);
      } else if (format == "csv") {
        writeCsvRecord(*outStream, "image", imageResult);
      }
    } else if (isVideoFile(filename)) {
      auto videoResult = processVideo(filename);
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
