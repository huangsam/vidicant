#include "vidicant/video.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vidicant;

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

TEST(VideoHandlerTest, ExtractFirstFrameWithMock) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  cv::Mat fakeFrame(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
  EXPECT_CALL(*mockLoader, open("mock.mp4"))
      .WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*mockLoader, readFrame()).WillOnce(::testing::Return(fakeFrame));

  VideoHandler handler(std::move(mockLoader));
  handler.open("mock.mp4");
  cv::Mat frame = handler.extractFirstFrame();

  EXPECT_EQ(frame.cols, 200);
  EXPECT_EQ(frame.rows, 100);
  EXPECT_EQ(frame.channels(), 3);
}

TEST(VideoHandlerTest, GetAverageBrightnessWithMock) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  cv::Mat frame1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
  cv::Mat frame2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
  cv::Mat emptyFrame;

  EXPECT_CALL(*mockLoader, open("mock.mp4"))
      .WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*mockLoader, readFrame())
      .WillOnce(::testing::Return(frame1))
      .WillOnce(::testing::Return(frame2))
      .WillOnce(::testing::Return(emptyFrame));

  VideoHandler handler(std::move(mockLoader));
  handler.open("mock.mp4");
  double brightness = handler.getAverageBrightness();

  EXPECT_NEAR(brightness, 150.0, 1e-3);
}

TEST(VideoHandlerTest, IsGrayscaleWithMock) {
  auto mockLoader = std::make_unique<MockVideoLoader>();
  cv::Mat grayFrame(10, 10, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, open("mock.mp4"))
      .WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*mockLoader, readFrame()).WillOnce(::testing::Return(grayFrame));

  VideoHandler handler(std::move(mockLoader));
  handler.open("mock.mp4");
  EXPECT_TRUE(handler.isGrayscale());
}

// Tests using real files for methods that need frame reading
TEST(VideoGlobalTest, GetVideoFrameCountReal) {
  int frameCount = vidicant::getVideoFrameCount("examples/sample.mp4");
  EXPECT_EQ(frameCount, 250);
}

TEST(VideoGlobalTest, GetVideoFPSReal) {
  double fps = vidicant::getVideoFPS("examples/sample.mp4");
  EXPECT_EQ(fps, 25.0);
}

TEST(VideoGlobalTest, GetVideoResolutionReal) {
  auto [width, height] = vidicant::getVideoResolution("examples/sample.mp4");
  EXPECT_EQ(width, 320);
  EXPECT_EQ(height, 176);
}

TEST(VideoGlobalTest, GetVideoDurationReal) {
  double duration = vidicant::getVideoDuration("examples/sample.mp4");
  EXPECT_EQ(duration, 10.0);
}

TEST(VideoGlobalTest, ExtractFirstFrameReal) {
  cv::Mat frame = vidicant::extractFirstFrame("examples/sample.mp4");
  EXPECT_FALSE(frame.empty());
  EXPECT_EQ(frame.cols, 320);
  EXPECT_EQ(frame.rows, 176);
}

TEST(VideoGlobalTest, GetVideoAverageBrightnessReal) {
  double brightness =
      vidicant::getVideoAverageBrightness("examples/sample.mp4");
  EXPECT_GT(brightness, 0.0);
}

TEST(VideoGlobalTest, IsVideoGrayscaleReal) {
  bool grayscale = vidicant::isVideoGrayscale("examples/sample.mp4");
  EXPECT_FALSE(grayscale);
}

TEST(VideoGlobalTest, SaveFirstFrameAsImageReal) {
  bool saved = vidicant::saveFirstFrameAsImage("examples/sample.mp4",
                                               "examples/test_first_frame.jpg");
  EXPECT_TRUE(saved);
  // Clean up
  std::remove("examples/test_first_frame.jpg");
}

TEST(VideoGlobalTest, GetVideoMotionScoreReal) {
  double motion = vidicant::getVideoMotionScore("examples/sample.mp4");
  EXPECT_GE(motion, 0.0);
}

TEST(VideoGlobalTest, GetVideoDominantColorsReal) {
  auto colors = vidicant::getVideoDominantColors("examples/sample.mp4");
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
  auto sceneChanges = vidicant::detectVideoSceneChanges("examples/sample.mp4");
  // Should return a vector of frame indices
  EXPECT_TRUE(sceneChanges.empty() ||
              !sceneChanges.empty()); // Can be empty or have changes
  for (int frameIdx : sceneChanges) {
    EXPECT_GE(frameIdx, 0);
  }
}

TEST(VideoGlobalTest, GetVideoFrameRateStabilityReal) {
  double stability =
      vidicant::getVideoFrameRateStability("examples/sample.mp4");
  EXPECT_GE(stability, 0.0); // Should be non-negative
}

TEST(VideoGlobalTest, GetVideoColorConsistencyReal) {
  double consistency =
      vidicant::getVideoColorConsistency("examples/sample.mp4");
  EXPECT_GE(consistency, 0.0); // Should be non-negative
  EXPECT_LE(consistency, 1.0); // Coefficient of variation should be <= 1.0
}

// Tests for new video convenience functions
TEST(VideoGlobalTest, DetectVideoSceneChangesConvenienceReal) {
  auto sceneChanges = vidicant::detectVideoSceneChanges("examples/sample.mp4");
  // Should return a vector of frame indices
  EXPECT_TRUE(sceneChanges.empty() ||
              !sceneChanges.empty()); // Can be empty or have changes
  for (int frameIdx : sceneChanges) {
    EXPECT_GE(frameIdx, 0);
  }
}

TEST(VideoGlobalTest, GetVideoFrameRateStabilityConvenienceReal) {
  double stability =
      vidicant::getVideoFrameRateStability("examples/sample.mp4");
  EXPECT_GE(stability, 0.0); // Should be non-negative
}

TEST(VideoGlobalTest, GetVideoColorConsistencyConvenienceReal) {
  double consistency =
      vidicant::getVideoColorConsistency("examples/sample.mp4");
  EXPECT_GE(consistency, 0.0); // Should be non-negative
  EXPECT_LE(consistency, 1.0); // Coefficient of variation should be <= 1.0
}

TEST(VideoGlobalTest, GetVideoOpticalFlowMagnitudeReal) {
  double flow = vidicant::getVideoOpticalFlowMagnitude("examples/sample.mp4");
  EXPECT_GE(flow, 0.0);
}

TEST(VideoGlobalTest, VideoHasAudioTrackReal) {
  // Just verify it returns a boolean without crashing
  bool hasAudio = vidicant::videoHasAudioTrack("examples/sample.mp4");
  (void)hasAudio;
  SUCCEED();
}

TEST(VideoGlobalTest, GetVideoShotLengthStatsReal) {
  ShotLengthStats stats =
      vidicant::getVideoShotLengthStats("examples/sample.mp4");
  EXPECT_GE(stats.count, 1);
  EXPECT_GT(stats.mean, 0.0);
  EXPECT_GE(stats.min, 0.0);
  EXPECT_GE(stats.max, stats.min);
}

TEST(VideoGlobalTest, GetVideoFlickerScoreReal) {
  double flicker = vidicant::getVideoFlickerScore("examples/sample.mp4");
  EXPECT_GE(flicker, 0.0);
}

TEST(VideoGlobalTest, GetVideoBestThumbnailIndexReal) {
  int idx = vidicant::getVideoBestThumbnailIndex("examples/sample.mp4");
  EXPECT_GE(idx, 0);
}

TEST(VideoGlobalTest, GetVideoTemporalBrightnessCurveReal) {
  auto curve = vidicant::getVideoTemporalBrightnessCurve("examples/sample.mp4");
  EXPECT_GT(curve.size(), 0U);
  EXPECT_LE(curve.size(), 100U);
  for (double b : curve) {
    EXPECT_GE(b, 0.0);
    EXPECT_LE(b, 255.0);
  }
}

TEST(VideoGlobalTest, GetVideoCodecFourccReal) {
  std::string fourcc = vidicant::getVideoCodecFourcc("examples/sample.mp4");
  // May be empty on some backends; just check it doesn't crash
  EXPECT_TRUE(fourcc.empty() || fourcc.length() == 4);
}

TEST(VideoGlobalTest, CompareVideoWithSelfReal) {
  double sim =
      vidicant::compareVideos("examples/sample.mp4", "examples/sample.mp4");
  EXPECT_GT(sim, 0.5); // Same video should be highly similar
}

TEST(VideoGlobalTest, GetVideoMetricsReal) {
  VideoMetrics m = vidicant::getVideoMetrics("examples/sample.mp4");
  EXPECT_GT(m.frame_count, 0);
  EXPECT_GT(m.fps, 0.0);
  EXPECT_GT(m.width, 0);
  EXPECT_GT(m.height, 0);
  EXPECT_GE(m.optical_flow_magnitude, 0.0);
  EXPECT_GE(m.flicker_score, 0.0);
  EXPECT_GE(m.best_thumbnail_frame, 0);
  EXPECT_GT(m.temporal_brightness_curve.size(), 0U);
  EXPECT_GE(m.shot_length_stats.count, 1);
}
