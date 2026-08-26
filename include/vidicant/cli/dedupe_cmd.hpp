// File: dedupe_cmd.hpp
// CLI handler for the 'dedupe' subcommand.

#ifndef VIDICANT_CLI_DEDUPE_CMD_HPP
#define VIDICANT_CLI_DEDUPE_CMD_HPP

#include "vidicant/cli/formatters.hpp"
#include "vidicant/core/dedupe.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace vidicant::cli {

namespace fs = std::filesystem;

inline int runDedupe(int argc, char *argv[]) {
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

  core::DedupeResult dedupeRes =
      core::dedupeDirectory(targetDir, threshold, true);

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
    nlohmann::json j = core::formatDedupeJson(dedupeRes);
    *outStream << j.dump(2) << std::endl;
  } else if (format == "jsonl") {
    for (const auto &cl : dedupeRes.clusters) {
      nlohmann::json clJson;
      clJson["cluster_id"] = cl.cluster_id;
      clJson["count"] = cl.members.size();
      clJson["lead_image"] = cl.lead_path;
      clJson["files"] = nlohmann::json::array();

      for (const auto &m : cl.members) {
        clJson["files"].push_back({{"path", m.path},
                                   {"perceptual_hash", m.hash},
                                   {"distance_to_lead", m.distance_to_lead}});
      }
      *outStream << clJson.dump() << std::endl;
    }
  } else if (format == "csv") {
    *outStream << "cluster_id,path,perceptual_hash,distance_to_lead,is_lead\n";
    for (const auto &cl : dedupeRes.clusters) {
      for (const auto &m : cl.members) {
        *outStream << cl.cluster_id << "," << escapeCsv(m.path) << "," << m.hash
                   << "," << m.distance_to_lead << ","
                   << (m.is_lead ? "true" : "false") << "\n";
      }
    }
  } else {
    // Human readable text
    *outStream << "Scanned " << dedupeRes.total_images << " images in '"
               << targetDir << "'." << std::endl;
    *outStream << "Found " << dedupeRes.clusters.size()
               << " duplicate/near-duplicate cluster(s) (threshold <= "
               << threshold << "):" << std::endl
               << std::endl;

    for (const auto &cl : dedupeRes.clusters) {
      *outStream << "Cluster #" << cl.cluster_id << " (" << cl.members.size()
                 << " files):" << std::endl;
      for (const auto &m : cl.members) {
        if (m.is_lead) {
          *outStream << "  - [LEAD] " << m.path << " (hash: " << m.hash << ")"
                     << std::endl;
        } else {
          *outStream << "  - " << m.path << " (hash: " << m.hash
                     << ", dist: " << m.distance_to_lead << ")" << std::endl;
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

} // namespace vidicant::cli

#endif // VIDICANT_CLI_DEDUPE_CMD_HPP
