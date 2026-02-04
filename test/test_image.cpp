#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/opencv.hpp>
#include "vidicant/image.hpp"

class MockImageLoader : public IImageLoader {
public:
    MOCK_METHOD(cv::Mat, imread, (const std::string&), (override));
};

TEST(ImageHandlerTest, GetDimensionsSuccess) {
    auto mockLoader = std::make_unique<MockImageLoader>();
    cv::Mat fakeImage(100, 200, CV_8UC3, cv::Scalar(0, 0, 0)); // height 100, width 200
    EXPECT_CALL(*mockLoader, imread("test.jpg")).WillOnce(::testing::Return(fakeImage));

    ImageHandler handler(std::move(mockLoader));
    auto [width, height] = handler.getDimensions("test.jpg");

    EXPECT_EQ(width, 200);
    EXPECT_EQ(height, 100);
}

TEST(ImageHandlerTest, GetDimensionsFail) {
    auto mockLoader = std::make_unique<MockImageLoader>();
    cv::Mat emptyImage;
    EXPECT_CALL(*mockLoader, imread("bad.jpg")).WillOnce(::testing::Return(emptyImage));

    ImageHandler handler(std::move(mockLoader));
    auto [width, height] = handler.getDimensions("bad.jpg");

    EXPECT_EQ(width, -1);
    EXPECT_EQ(height, -1);
}