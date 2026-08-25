// File: dedupe.cpp
// Implementation of core perceptual hash deduplication and clustering
// algorithms.

#include "vidicant/core/dedupe.hpp"
#include "vidicant/image.hpp"
#include "vidicant/io/file_detector.hpp"
#include <algorithm>
#include <bitset>
#include <map>
#include <numeric>
#include <utility>

namespace vidicant::core {

int hammingDistance(uint64_t a, uint64_t b) {
  return static_cast<int>(std::bitset<64>(a ^ b).count());
}

DisjointSet::DisjointSet(int n) : parent_(n), rank_(n, 0) {
  std::iota(parent_.begin(), parent_.end(), 0);
}

int DisjointSet::find(int i) {
  if (parent_[i] == i)
    return i;
  return parent_[i] = find(parent_[i]);
}

void DisjointSet::unite(int i, int j) {
  int root_i = find(i);
  int root_j = find(j);
  if (root_i != root_j) {
    if (rank_[root_i] < rank_[root_j])
      std::swap(root_i, root_j);
    parent_[root_j] = root_i;
    if (rank_[root_i] == rank_[root_j])
      rank_[root_i]++;
  }
}

DedupeResult clusterDuplicateHashes(const std::vector<ImageHashItem> &hashes,
                                    int threshold) {
  DedupeResult result;
  result.threshold = threshold;
  result.total_images = static_cast<int>(hashes.size());

  int n = result.total_images;
  if (n <= 1) {
    return result;
  }

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

  int clusterId = 1;
  for (const auto &[leadIdx, members] : clustersMap) {
    if (members.size() >= 2) {
      DuplicateCluster cluster;
      cluster.cluster_id = clusterId++;
      cluster.lead_index = leadIdx;
      cluster.lead_path = hashes[leadIdx].path;
      cluster.lead_hash = hashes[leadIdx].hash;

      cluster.members.reserve(members.size());
      for (int idx : members) {
        DuplicateMember member;
        member.path = hashes[idx].path;
        member.hash = hashes[idx].hash;
        member.distance_to_lead =
            hammingDistance(hashes[leadIdx].hash, hashes[idx].hash);
        member.is_lead = (idx == leadIdx);
        cluster.members.push_back(member);
      }
      result.clusters.push_back(cluster);
    }
  }

  return result;
}

DedupeResult dedupeDirectory(const std::filesystem::path &dir, int threshold,
                             bool recursive) {
  std::vector<std::string> imageFiles;
  if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
    if (recursive) {
      for (const auto &entry : std::filesystem::recursive_directory_iterator(
               dir,
               std::filesystem::directory_options::skip_permission_denied)) {
        if (entry.is_regular_file() && io::isImageFile(entry.path())) {
          imageFiles.push_back(entry.path().string());
        }
      }
    } else {
      for (const auto &entry : std::filesystem::directory_iterator(
               dir,
               std::filesystem::directory_options::skip_permission_denied)) {
        if (entry.is_regular_file() && io::isImageFile(entry.path())) {
          imageFiles.push_back(entry.path().string());
        }
      }
    }
  }

  std::sort(imageFiles.begin(), imageFiles.end());

  std::vector<ImageHashItem> hashes;
  hashes.reserve(imageFiles.size());
  for (const auto &img : imageFiles) {
    uint64_t h = vidicant::getImagePerceptualHash(img);
    hashes.push_back({img, h});
  }

  return clusterDuplicateHashes(hashes, threshold);
}

nlohmann::json formatDedupeJson(const DedupeResult &result) {
  nlohmann::json j;
  j["threshold"] = result.threshold;
  j["total_images"] = result.total_images;
  j["clusters_count"] = result.clusters.size();
  j["duplicate_clusters"] = nlohmann::json::array();

  for (const auto &cl : result.clusters) {
    nlohmann::json clJson;
    clJson["cluster_id"] = cl.cluster_id;
    clJson["count"] = cl.members.size();
    clJson["lead_image"] = cl.lead_path;
    clJson["files"] = nlohmann::json::array();

    for (const auto &m : cl.members) {
      clJson["files"].push_back({
          {"path", m.path},
          {"perceptual_hash", m.hash},
          {"distance_to_lead", m.distance_to_lead},
      });
    }
    j["duplicate_clusters"].push_back(clJson);
  }

  return j;
}

} // namespace vidicant::core
