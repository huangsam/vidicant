#include "vidicant/image.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace vidicant;

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

TEST(ImageHandlerTest, GetNoiseEstimate) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(20, 20, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("noise.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double noise = handler.getNoiseEstimate("noise.jpg");

  EXPECT_GE(noise, 0.0);
}

TEST(ImageHandlerTest, GetNoiseEstimateTooSmall) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(2, 2, CV_8UC1, cv::Scalar(100));
  EXPECT_CALL(*mockLoader, imread("tiny.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double noise = handler.getNoiseEstimate("tiny.jpg");

  EXPECT_EQ(noise, 0.0); // Too small, returns 0
}

TEST(ImageHandlerTest, GetSymmetryScore) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  // A uniform image should be perfectly symmetric
  cv::Mat image(20, 20, CV_8UC1, cv::Scalar(100));
  EXPECT_CALL(*mockLoader, imread("sym.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double sym = handler.getSymmetryScore("sym.jpg");

  EXPECT_GE(sym, -1.0);
  EXPECT_LE(sym, 1.0);
}

TEST(ImageHandlerTest, GetTextureFeatures) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(20, 20, CV_8UC1, cv::Scalar(100));
  EXPECT_CALL(*mockLoader, imread("texture.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  TextureFeatures tf = handler.getTextureFeatures("texture.jpg");

  EXPECT_GE(tf.contrast, 0.0);
  EXPECT_GE(tf.energy, 0.0);
  EXPECT_LE(tf.energy, 1.0);
  EXPECT_GE(tf.homogeneity, 0.0);
  EXPECT_LE(tf.homogeneity, 1.0);
}

TEST(ImageHandlerTest, GetPerceptualHash) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
  EXPECT_CALL(*mockLoader, imread("hash.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  uint64_t hash = handler.getPerceptualHash("hash.jpg");

  // Hash is a 64-bit integer; uniform image gives all-zero hash
  EXPECT_EQ(hash, 0ULL);
}

TEST(ImageHandlerTest, GetPerceptualHashEmpty) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat empty;
  EXPECT_CALL(*mockLoader, imread("bad.jpg"))
      .WillOnce(::testing::Return(empty));

  ImageHandler handler(std::move(mockLoader));
  uint64_t hash = handler.getPerceptualHash("bad.jpg");

  EXPECT_EQ(hash, 0ULL);
}

TEST(ImageHandlerTest, GetWhiteBalanceScore) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  // Neutral grey: perfect white balance
  cv::Mat image(10, 10, CV_8UC3, cv::Scalar(128, 128, 128));
  EXPECT_CALL(*mockLoader, imread("wb.jpg")).WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double score = handler.getWhiteBalanceScore("wb.jpg");

  EXPECT_NEAR(score, 0.0, 1e-9);
}

TEST(ImageHandlerTest, GetWhiteBalanceScoreGrayscale) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("gray_wb.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double score = handler.getWhiteBalanceScore("gray_wb.jpg");

  EXPECT_EQ(score, -1.0); // Grayscale not applicable
}

TEST(ImageHandlerTest, GetHueHistogram) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue in BGR
  EXPECT_CALL(*mockLoader, imread("hue.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  auto hueHist = handler.getHueHistogram("hue.jpg");

  EXPECT_EQ(hueHist.size(), 36U);
}

TEST(ImageHandlerTest, GetHueHistogramGrayscale) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("gray_hue.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  auto hueHist = handler.getHueHistogram("gray_hue.jpg");

  EXPECT_TRUE(hueHist.empty());
}

TEST(ImageHandlerTest, GetSharpnessScore) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(10, 10, CV_8UC1, cv::Scalar(128));
  image.at<uchar>(5, 5) = 255; // Sharp edge
  EXPECT_CALL(*mockLoader, imread("sharp.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double sharpness = handler.getSharpnessScore("sharp.jpg");

  EXPECT_GT(sharpness, 0.0);
}

TEST(ImageHandlerTest, CompareImagesSelf) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(20, 20, CV_8UC3, cv::Scalar(100, 150, 200));
  // Cache returns image after first load; mock called once for "img.jpg"
  EXPECT_CALL(*mockLoader, imread("img.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  double ssim = handler.compareImages("img.jpg", "img.jpg");

  EXPECT_NEAR(ssim, 1.0, 0.01); // Identical images
}

TEST(ImageHandlerTest, CompareImagesDifferent) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat white(20, 20, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat black(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
  EXPECT_CALL(*mockLoader, imread("white.jpg"))
      .WillOnce(::testing::Return(white));
  EXPECT_CALL(*mockLoader, imread("black.jpg"))
      .WillOnce(::testing::Return(black));

  ImageHandler handler(std::move(mockLoader));
  double ssim = handler.compareImages("white.jpg", "black.jpg");

  EXPECT_LT(ssim, 0.5); // Very different images
}

TEST(ImageHandlerTest, GetNoiseTypeGaussian) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  // Smooth uniform image → gaussian classification
  cv::Mat image(20, 20, CV_8UC1, cv::Scalar(128));
  EXPECT_CALL(*mockLoader, imread("gaussian_noise.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  std::string noiseType = handler.getNoiseType("gaussian_noise.jpg");

  EXPECT_EQ(noiseType, "gaussian");
}

TEST(ImageHandlerTest, GetNoiseTypeSaltPepper) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(20, 20, CV_8UC1, cv::Scalar(128));
  // Salt-and-pepper: lots of 0 and 255 pixels
  for (int i = 0; i < 20; ++i) {
    image.at<uchar>(i, i) = 0;
    image.at<uchar>(i, 19 - i) = 255;
  }
  EXPECT_CALL(*mockLoader, imread("sp_noise.jpg"))
      .WillOnce(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  std::string noiseType = handler.getNoiseType("sp_noise.jpg");

  EXPECT_EQ(noiseType, "salt_and_pepper");
}

TEST(ImageHandlerTest, GetMetrics) {
  auto mockLoader = std::make_unique<MockImageLoader>();
  cv::Mat image(20, 20, CV_8UC3, cv::Scalar(100, 150, 200));
  EXPECT_CALL(*mockLoader, imread(::testing::_))
      .WillRepeatedly(::testing::Return(image));

  ImageHandler handler(std::move(mockLoader));
  ImageMetrics m = handler.getMetrics("test.jpg");

  EXPECT_EQ(m.width, 20);
  EXPECT_EQ(m.height, 20);
  EXPECT_FALSE(m.is_grayscale);
  EXPECT_GE(m.sharpness_score, 0.0);
  EXPECT_EQ(m.hue_histogram.size(), 36U);
}

// Tests using real files for new convenience functions
TEST(ImageGlobalTest, GetImageNoiseEstimateReal) {
  double noise = vidicant::getImageNoiseEstimate(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(noise, 0.0);
}

TEST(ImageGlobalTest, GetImageSymmetryScoreReal) {
  double sym = vidicant::getImageSymmetryScore(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(sym, -1.0);
  EXPECT_LE(sym, 1.0);
}

TEST(ImageGlobalTest, GetImageTextureFeaturesReal) {
  TextureFeatures tf = vidicant::getImageTextureFeatures(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(tf.contrast, 0.0);
  EXPECT_GE(tf.energy, 0.0);
  EXPECT_LE(tf.energy, 1.0);
  EXPECT_GE(tf.homogeneity, 0.0);
}

TEST(ImageGlobalTest, GetImagePerceptualHashReal) {
  uint64_t hash = vidicant::getImagePerceptualHash(
      "/workspaces/vidicant/examples/sample.jpg");
  // Hash of same image called twice must be identical
  uint64_t hash2 = vidicant::getImagePerceptualHash(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_EQ(hash, hash2);
}

TEST(ImageGlobalTest, GetImageWhiteBalanceScoreReal) {
  double wb = vidicant::getImageWhiteBalanceScore(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GE(wb, 0.0);
}

TEST(ImageGlobalTest, GetImageHueHistogramReal) {
  auto hueHist = vidicant::getImageHueHistogram(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_EQ(hueHist.size(), 36U);
  int total = 0;
  for (int bin : hueHist)
    total += bin;
  EXPECT_GT(total, 0);
}

TEST(ImageGlobalTest, GetImageSharpnessScoreReal) {
  double sharpness = vidicant::getImageSharpnessScore(
      "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GT(sharpness, 0.0);
}

TEST(ImageGlobalTest, CompareImagesWithSelfReal) {
  double ssim =
      vidicant::compareImages("/workspaces/vidicant/examples/sample.jpg",
                              "/workspaces/vidicant/examples/sample.jpg");
  EXPECT_NEAR(ssim, 1.0, 0.01);
}

TEST(ImageGlobalTest, GetImageNoiseTypeReal) {
  std::string noiseType =
      vidicant::getImageNoiseType("/workspaces/vidicant/examples/sample.jpg");
  EXPECT_TRUE(noiseType == "gaussian" || noiseType == "salt_and_pepper");
}

TEST(ImageGlobalTest, GetImageMetricsReal) {
  ImageMetrics m =
      vidicant::getImageMetrics("/workspaces/vidicant/examples/sample.jpg");
  EXPECT_GT(m.width, 0);
  EXPECT_GT(m.height, 0);
  EXPECT_GE(m.blur_score, 0.0);
  EXPECT_GE(m.noise_estimate, 0.0);
  EXPECT_EQ(m.hue_histogram.size(), 36U);
  EXPECT_GE(m.sharpness_score, 0.0);
  EXPECT_FALSE(m.noise_type.empty());
}
