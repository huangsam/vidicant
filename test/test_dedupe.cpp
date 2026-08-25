#include "vidicant/core/dedupe.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace vidicant::core;

TEST(DedupeTest, HammingDistanceCalculations) {
  EXPECT_EQ(hammingDistance(0x0ULL, 0x0ULL), 0);
  EXPECT_EQ(hammingDistance(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL), 0);
  EXPECT_EQ(hammingDistance(0x0ULL, 0x1ULL), 1);
  EXPECT_EQ(hammingDistance(0x0ULL, 0x3ULL), 2);
  EXPECT_EQ(hammingDistance(0x0ULL, 0xFFFFFFFFFFFFFFFFULL), 64);
  EXPECT_EQ(hammingDistance(0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL), 64);
}

TEST(DedupeTest, DisjointSetOperations) {
  DisjointSet dsu(5);
  // Initially all are separate
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(dsu.find(i), i);
  }

  // Unite 0 and 1
  dsu.unite(0, 1);
  EXPECT_EQ(dsu.find(0), dsu.find(1));

  // Unite 2 and 3
  dsu.unite(2, 3);
  EXPECT_EQ(dsu.find(2), dsu.find(3));
  EXPECT_NE(dsu.find(0), dsu.find(2));

  // Unite 1 and 3 (merges {0,1} and {2,3})
  dsu.unite(1, 3);
  EXPECT_EQ(dsu.find(0), dsu.find(3));
  EXPECT_EQ(dsu.find(1), dsu.find(2));

  // 4 remains distinct
  EXPECT_NE(dsu.find(4), dsu.find(0));
}

TEST(DedupeTest, ClusterDuplicateHashesEmptyAndSingle) {
  EXPECT_EQ(clusterDuplicateHashes({}, 5).clusters.size(), 0);

  std::vector<ImageHashItem> single = {{"img1.jpg", 0x123456789ABCDEF0ULL}};
  auto resSingle = clusterDuplicateHashes(single, 5);
  EXPECT_EQ(resSingle.total_images, 1);
  EXPECT_EQ(resSingle.clusters.size(), 0);
}

TEST(DedupeTest, ClusterDuplicateHashesGroupings) {
  std::vector<ImageHashItem> items = {
      {"img_a1.jpg", 0x0000000000000000ULL}, // Lead of cluster 1
      {"img_a2.jpg", 0x0000000000000001ULL}, // dist = 1 to a1 -> in cluster 1
      {"img_a3.jpg", 0x0000000000000003ULL}, // dist = 2 to a1 -> in cluster 1
      {"img_b1.jpg", 0x0000FFFF00000000ULL}, // Unique / far from cluster 1
      {"img_c1.jpg", 0xFFFFFFFFFFFFFFFFULL}, // Lead of cluster 2
      {"img_c2.jpg", 0xFFFFFFFFFFFFFFFEULL}, // dist = 1 to c1 -> in cluster 2
  };

  auto res = clusterDuplicateHashes(items, 3);
  EXPECT_EQ(res.total_images, 6);
  ASSERT_EQ(res.clusters.size(), 2);

  // Cluster 1 has 3 items
  const auto &cl1 = res.clusters[0];
  EXPECT_EQ(cl1.members.size(), 3);
  EXPECT_EQ(cl1.lead_path, "img_a1.jpg");

  // Cluster 2 has 2 items
  const auto &cl2 = res.clusters[1];
  EXPECT_EQ(cl2.members.size(), 2);
  EXPECT_EQ(cl2.lead_path, "img_c1.jpg");

  // JSON formatting check
  nlohmann::json j = formatDedupeJson(res);
  EXPECT_EQ(j["total_images"], 6);
  EXPECT_EQ(j["clusters_count"], 2);
  ASSERT_TRUE(j["duplicate_clusters"].is_array());
  EXPECT_EQ(j["duplicate_clusters"].size(), 2);
}
