// File: dedupe.hpp
// Core perceptual hash deduplication and clustering algorithms.

#ifndef VIDICANT_CORE_DEDUPE_HPP
#define VIDICANT_CORE_DEDUPE_HPP

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace vidicant::core {

// Computes the Hamming distance between two 64-bit perceptual hashes.
int hammingDistance(uint64_t a, uint64_t b);

// Disjoint-set data structure with path compression and union by rank.
class DisjointSet {
public:
  explicit DisjointSet(int n);
  int find(int i);
  void unite(int i, int j);

private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

// Represents an image path and its computed perceptual hash.
struct ImageHashItem {
  std::string path;
  uint64_t hash{0};
};

// Represents an individual member in a duplicate cluster.
struct DuplicateMember {
  std::string path;
  uint64_t hash{0};
  int distance_to_lead{0};
  bool is_lead{false};
};

// Represents a cluster of near-duplicate images.
struct DuplicateCluster {
  int cluster_id{0};
  int lead_index{0};
  std::string lead_path;
  uint64_t lead_hash{0};
  std::vector<DuplicateMember> members;
};

// Aggregated results of duplicate clustering.
struct DedupeResult {
  int threshold{5};
  int total_images{0};
  std::vector<DuplicateCluster> clusters;
};

// Clusters an existing list of image hashes within the specified Hamming
// distance threshold.
DedupeResult clusterDuplicateHashes(const std::vector<ImageHashItem> &hashes,
                                    int threshold = 5);

// Scans a directory for images, calculates perceptual hashes, and clusters
// near-duplicates.
DedupeResult dedupeDirectory(const std::filesystem::path &dir,
                             int threshold = 5, bool recursive = true);

// Formats DedupeResult into standard nlohmann::json structure.
nlohmann::json formatDedupeJson(const DedupeResult &result);

} // namespace vidicant::core

#endif // VIDICANT_CORE_DEDUPE_HPP
