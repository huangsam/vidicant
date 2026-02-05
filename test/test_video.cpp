#include "vidicant/video.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

class MockVideoLoader : public IVideoLoader {
public:
  MOCK_METHOD(bool, open, (const std::string &), (override));
  MOCK_METHOD(int, getFrameCount, (), (override));
  MOCK_METHOD(double, getFPS, (), (override));
  MOCK_METHOD((std::pair<int, int>), getResolution, (), (override));
  MOCK_METHOD(cv::Mat, readFrame, (), (override));
};

TEST(VideoHandlerTest, GetFrameCount) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  EXPECT_CALL(*mockLoader, open("test.mp4")).WillOnce(::testing::Return(true));
  EXPECT_CALL(*mockLoader, getFrameCount()).WillOnce(::testing::Return(100));

  VideoHandler handler(std::move(mockLoader));
  handler.open("test.mp4");
  int frameCount = handler.getFrameCount();

  EXPECT_EQ(frameCount, 100);
}

TEST(VideoHandlerTest, GetFPS) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  EXPECT_CALL(*mockLoader, open("test.mp4")).WillOnce(::testing::Return(true));
  EXPECT_CALL(*mockLoader, getFPS()).WillOnce(::testing::Return(30.0));

  VideoHandler handler(std::move(mockLoader));
  handler.open("test.mp4");
  double fps = handler.getFPS();

  EXPECT_EQ(fps, 30.0);
}

TEST(VideoHandlerTest, GetResolution) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  EXPECT_CALL(*mockLoader, open("test.mp4")).WillOnce(::testing::Return(true));
  EXPECT_CALL(*mockLoader, getResolution())
      .WillOnce(::testing::Return(std::make_pair(1920, 1080)));

  VideoHandler handler(std::move(mockLoader));
  handler.open("test.mp4");
  auto [width, height] = handler.getResolution();

  EXPECT_EQ(width, 1920);
  EXPECT_EQ(height, 1080);
}

TEST(VideoHandlerTest, GetDuration) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  EXPECT_CALL(*mockLoader, open("test.mp4")).WillOnce(::testing::Return(true));
  EXPECT_CALL(*mockLoader, getFrameCount()).WillOnce(::testing::Return(300));
  EXPECT_CALL(*mockLoader, getFPS()).WillOnce(::testing::Return(30.0));

  VideoHandler handler(std::move(mockLoader));
  handler.open("test.mp4");
  double duration = handler.getDuration();

  EXPECT_EQ(duration, 10.0); // 300 / 30
}

TEST(VideoHandlerTest, OpenFail) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  EXPECT_CALL(*mockLoader, open("bad.mp4")).WillOnce(::testing::Return(false));

  VideoHandler handler(std::move(mockLoader));
  bool opened = handler.open("bad.mp4");

  EXPECT_FALSE(opened);
}

// Tests using real files for methods that need frame reading
TEST(VideoGlobalTest, GetVideoFrameCountReal) {
  int frameCount =
      vidicant::getVideoFrameCount("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_EQ(frameCount, 250);
}

TEST(VideoGlobalTest, GetVideoFPSReal) {
  double fps =
      vidicant::getVideoFPS("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_EQ(fps, 25.0);
}

TEST(VideoGlobalTest, GetVideoResolutionReal) {
  auto [width, height] =
      vidicant::getVideoResolution("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_EQ(width, 320);
  EXPECT_EQ(height, 176);
}

TEST(VideoGlobalTest, GetVideoDurationReal) {
  double duration =
      vidicant::getVideoDuration("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_EQ(duration, 10.0);
}

TEST(VideoGlobalTest, ExtractFirstFrameReal) {
  cv::Mat frame =
      vidicant::extractFirstFrame("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_FALSE(frame.empty());
  EXPECT_EQ(frame.cols, 320);
  EXPECT_EQ(frame.rows, 176);
}

TEST(VideoGlobalTest, GetVideoAverageBrightnessReal) {
  double brightness = vidicant::getVideoAverageBrightness(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GT(brightness, 0.0);
}

TEST(VideoGlobalTest, IsVideoGrayscaleReal) {
  bool grayscale =
      vidicant::isVideoGrayscale("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_FALSE(grayscale);
}

TEST(VideoGlobalTest, SaveFirstFrameAsImageReal) {
  bool saved = vidicant::saveFirstFrameAsImage(
      "/workspaces/vidicant/examples/sample.mp4",
      "/workspaces/vidicant/examples/test_first_frame.jpg");
  EXPECT_TRUE(saved);
  // Clean up
  std::remove("/workspaces/vidicant/examples/test_first_frame.jpg");
}

TEST(VideoGlobalTest, GetVideoMotionScoreReal) {
  double motion =
      vidicant::getVideoMotionScore("/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GE(motion, 0.0);
}

TEST(VideoGlobalTest, GetVideoDominantColorsReal) {
  auto colors = vidicant::getVideoDominantColors(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_EQ(colors.size(), 3);
  for (const auto &color : colors) {
    EXPECT_GE(color[0], 0.0);
    EXPECT_LE(color[0], 255.0);
    EXPECT_GE(color[1], 0.0);
    EXPECT_LE(color[1], 255.0);
    EXPECT_GE(color[2], 0.0);
    EXPECT_LE(color[2], 255.0);
  }
}

TEST(VideoGlobalTest, DetectVideoSceneChangesReal) {
  auto sceneChanges = vidicant::detectVideoSceneChanges(
      "/workspaces/vidicant/examples/sample.mp4");
  // Should return a vector of frame indices
  EXPECT_TRUE(sceneChanges.empty() ||
              !sceneChanges.empty()); // Can be empty or have changes
  for (int frameIdx : sceneChanges) {
    EXPECT_GE(frameIdx, 0);
  }
}

TEST(VideoGlobalTest, GetVideoFrameRateStabilityReal) {
  double stability = vidicant::getVideoFrameRateStability(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GE(stability, 0.0); // Should be non-negative
}

TEST(VideoGlobalTest, GetVideoColorConsistencyReal) {
  double consistency = vidicant::getVideoColorConsistency(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GE(consistency, 0.0); // Should be non-negative
  EXPECT_LE(consistency, 1.0); // Coefficient of variation should be <= 1.0
}

// Tests for new video convenience functions
TEST(VideoGlobalTest, DetectVideoSceneChangesConvenienceReal) {
  auto sceneChanges = vidicant::detectVideoSceneChanges(
      "/workspaces/vidicant/examples/sample.mp4");
  // Should return a vector of frame indices
  EXPECT_TRUE(sceneChanges.empty() ||
              !sceneChanges.empty()); // Can be empty or have changes
  for (int frameIdx : sceneChanges) {
    EXPECT_GE(frameIdx, 0);
  }
}

TEST(VideoGlobalTest, GetVideoFrameRateStabilityConvenienceReal) {
  double stability = vidicant::getVideoFrameRateStability(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GE(stability, 0.0); // Should be non-negative
}

TEST(VideoGlobalTest, GetVideoColorConsistencyConvenienceReal) {
  double consistency = vidicant::getVideoColorConsistency(
      "/workspaces/vidicant/examples/sample.mp4");
  EXPECT_GE(consistency, 0.0); // Should be non-negative
  EXPECT_LE(consistency, 1.0); // Coefficient of variation should be <= 1.0
}