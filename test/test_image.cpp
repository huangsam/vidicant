#include "vidicant/image.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

class MockImageLoader : public IImageLoader {
public:
  MOCK_METHOD(cv::Mat, imread, (const std::string &), (override));
};

TEST(ImageHandlerTest, GetDimensionsSuccess) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat fakeImage(100, 200, CV_8UC3,
                    cv::Scalar(0, 0, 0)); // height 100, width 200
  EXPECT_CALL(*mockLoader, imread("test.jpg"))
      .WillOnce(::testing::Return(fakeImage));

  ImageHandler handler(std::move(mockLoader));
  auto [width, height] = handler.getDimensions("test.jpg");

  EXPECT_EQ(width, 200);
  EXPECT_EQ(height, 100);
}

TEST(ImageHandlerTest, GetDimensionsFail) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat emptyImage;
  EXPECT_CALL(*mockLoader, imread("bad.jpg"))
      .WillOnce(::testing::Return(emptyImage));

  ImageHandler handler(std::move(mockLoader));
  auto [width, height] = handler.getDimensions("bad.jpg");

  EXPECT_EQ(width, -1);
  EXPECT_EQ(height, -1);
}

TEST(ImageHandlerTest, IsGrayscaleTrue) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat grayImage(10, 10, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("gray.jpg"))
      .WillOnce(::testing::Return(grayImage));

  ImageHandler handler(std::move(mockLoader));
  bool isGray = handler.isGrayscale("gray.jpg");

  EXPECT_TRUE(isGray);
}

TEST(ImageHandlerTest, IsGrayscaleFalse) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat colorImage(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
  EXPECT_CALL(*mockLoader, imread("color.jpg"))
      .WillOnce(::testing::Return(colorImage));

  ImageHandler handler(std::move(mockLoader));
  bool isGray = handler.isGrayscale("color.jpg");

  EXPECT_FALSE(isGray);
}

TEST(ImageHandlerTest, GetAverageBrightness) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(2, 2, CV_8UC3, cv::Scalar(100, 150, 200));
  EXPECT_CALL(*mockLoader, imread("bright.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double brightness = handler.getAverageBrightness("bright.jpg");

  EXPECT_NEAR(brightness, 150.0, 1.0); // (100+150+200)/3 = 150
}

TEST(ImageHandlerTest, GetNumberOfChannels) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC3);
  EXPECT_CALL(*mockLoader, imread("channels.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  int channels = handler.getNumberOfChannels("channels.jpg");

  EXPECT_EQ(channels, 3);
}

TEST(ImageHandlerTest, GetEdgeCount) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
  // Add some edges by setting some pixels to 255
  image.at<uchar>(5, 5) = 255;
  EXPECT_CALL(*mockLoader, imread("edges.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  int edgeCount = handler.getEdgeCount("edges.jpg");

  EXPECT_GT(edgeCount, 0); // Should detect some edges
}

TEST(ImageHandlerTest, GetDominantColors) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC3, cv::Scalar(255, 0, 0)); // Red image
  EXPECT_CALL(*mockLoader, imread("colors.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  auto colors = handler.getDominantColors("colors.jpg", 1);

  EXPECT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0][0], 255.0, 10.0); // Red channel
  EXPECT_NEAR(colors[0][1], 0.0, 10.0);   // Green
  EXPECT_NEAR(colors[0][2], 0.0, 10.0);   // Blue
}

TEST(ImageHandlerTest, GetBlurScore) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat sharpImage(10, 10, CV_8UC1, cv::Scalar(128));
  // Make it sharp by adding high frequency
  sharpImage.at<uchar>(5, 5) = 255;
  EXPECT_CALL(*mockLoader, imread("sharp.jpg"))
      .WillOnce(::testing::Return(sharpImage));

  ImageHandler handler(std::move(mockLoader));
  double blurScore = handler.getBlurScore("sharp.jpg");

  EXPECT_GT(blurScore, 0.0); // Should have some variance
}

TEST(ImageHandlerTest, GetContrastRatio) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(100));
  image.at<uchar>(5, 5) = 200; // Add some contrast
  EXPECT_CALL(*mockLoader, imread("contrast.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double contrastRatio = handler.getContrastRatio("contrast.jpg");

  EXPECT_GT(contrastRatio, 1.0); // Should have contrast
}

TEST(ImageHandlerTest, GetSaturationLevel) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC3, cv::Scalar(100, 150, 200)); // High saturation
  EXPECT_CALL(*mockLoader, imread("saturated.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double saturation = handler.getSaturationLevel("saturated.jpg");

  EXPECT_GT(saturation, 0.0);
  EXPECT_LE(saturation, 255.0);
}

TEST(ImageHandlerTest, GetHistogram) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 128, 255));
  EXPECT_CALL(*mockLoader, imread("histogram.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  auto histogram = handler.getHistogram("histogram.jpg");

  EXPECT_EQ(histogram.size(), 3);      // RGB channels
  EXPECT_EQ(histogram[0].size(), 256); // 256 bins
  EXPECT_EQ(histogram[1].size(), 256);
  EXPECT_EQ(histogram[2].size(), 256);
}

TEST(ImageHandlerTest, GetAspectRatio) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(100, 200, CV_8UC3); // width 200, height 100
  EXPECT_CALL(*mockLoader, imread("aspect.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double aspectRatio = handler.getAspectRatio("aspect.jpg");

  EXPECT_EQ(aspectRatio, 2.0); // 200/100 = 2.0
}

TEST(ImageHandlerTest, GetImageEntropy) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("entropy.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double entropy = handler.getImageEntropy("entropy.jpg");

  EXPECT_GE(entropy, 0.0);
  EXPECT_LE(entropy, 8.0); // Max entropy for 8-bit image
}

// Tests using real files for convenience functions
TEST(ImageGlobalTest, GetImageContrastRatioReal) {
  double contrast = vidicant::getImageContrastRatio(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GT(contrast, 1.0); // Should have some contrast
}

TEST(ImageGlobalTest, GetImageSaturationLevelReal) {
  double saturation = vidicant::getImageSaturationLevel(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(saturation, 0.0);
  EXPECT_LE(saturation, 255.0);
}

TEST(ImageGlobalTest, GetImageHistogramReal) {
  auto histogram =
      vidicant::getImageHistogram("/workspaces/vidicant/examples/sample.jpg");
  EXPECT_EQ(histogram.size(), 3);      // RGB channels
  EXPECT_EQ(histogram[0].size(), 256); // 256 bins per channel
  EXPECT_EQ(histogram[1].size(), 256);
  EXPECT_EQ(histogram[2].size(), 256);
}

TEST(ImageGlobalTest, GetImageAspectRatioReal) {
  double aspectRatio =
      vidicant::getImageAspectRatio("/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GT(aspectRatio, 0.0);
}

TEST(ImageGlobalTest, GetImageEntropyReal) {
  double entropy =
      vidicant::getImageEntropy("/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(entropy, 0.0);
  EXPECT_LE(entropy, 8.0);
}