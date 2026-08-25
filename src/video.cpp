// File: video.cpp
// Implementation of video loader, VideoHandler coordinator, and public video
// functions.

#include "vidicant/video.hpp"
#include "core/video_ops.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace vidicant {

bool OpenCVVideoLoader::open(const std::string &filename) {
  cap_.open(filename, cv::CAP_FFMPEG);
  return cap_.isOpened();
}

int OpenCVVideoLoader::getFrameCount() {
  return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
}

double OpenCVVideoLoader::getFPS() { return cap_.get(cv::CAP_PROP_FPS); }

std::pair<int, int> OpenCVVideoLoader::getResolution() {
  int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
  return {width, height};
}

cv::Mat OpenCVVideoLoader::readFrame() {
  cv::Mat frame;
  cap_ >> frame;
  return frame;
}

double OpenCVVideoLoader::getProperty(int propId) { return cap_.get(propId); }

VideoHandler::VideoHandler(std::unique_ptr<IVideoLoader> loader)
    : loader_(std::move(loader)) {}

bool VideoHandler::open(const std::string &filename) {
  filename_ = filename;
  return loader_ && loader_->open(filename);
}

int VideoHandler::getFrameCount() {
  return loader_ ? loader_->getFrameCount() : -1;
}

double VideoHandler::getFPS() { return loader_ ? loader_->getFPS() : -1.0; }

std::pair<int, int> VideoHandler::getResolution() {
  return loader_ ? loader_->getResolution() : std::make_pair(-1, -1);
}

double VideoHandler::getDuration() {
  if (!loader_)
    return -1.0;
  int frameCount = loader_->getFrameCount();
  double fps = loader_->getFPS();
  if (fps <= 0.0)
    return -1.0;
  return static_cast<double>(frameCount) / fps;
}

cv::Mat VideoHandler::extractFirstFrame() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return cv::Mat();
  return loader_->readFrame();
}

double VideoHandler::getAverageBrightness() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  double totalBrightness = 0.0;
  int frameCount = 0;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty()) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    totalBrightness += brightness;
    frameCount++;
    if (frameCount > 100)
      break;
    frame = loader_->readFrame();
  }
  return frameCount > 0 ? (totalBrightness / frameCount) : -1.0;
}

bool VideoHandler::isGrayscale() {
  cv::Mat frame = extractFirstFrame();
  return !frame.empty() && frame.channels() == 1;
}

bool VideoHandler::saveFirstFrameAsImage(const std::string &imagePath) {
  cv::Mat frame = extractFirstFrame();
  if (frame.empty())
    return false;
  return cv::imwrite(imagePath, frame);
}

double VideoHandler::getMotionScore() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  cv::Mat prevFrame = loader_->readFrame();
  if (prevFrame.empty())
    return 0.0;

  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

  double totalMotion = 0.0;
  int frameCount = 1;
  cv::Mat currFrame = loader_->readFrame();
  while (!currFrame.empty() && frameCount < 50) {
    cv::Mat grayCurr;
    if (currFrame.channels() == 1)
      grayCurr = currFrame;
    else
      cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);

    totalMotion += core::calculateFrameMotion(prevGray, grayCurr);
    prevGray = grayCurr;
    frameCount++;
    currFrame = loader_->readFrame();
  }
  return frameCount > 1 ? (totalMotion / (frameCount - 1)) : 0.0;
}

std::vector<std::array<double, 3>> VideoHandler::getDominantColors() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return {};

  std::vector<cv::Mat> frames;
  int count = 0;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty() && count < 10) {
    frames.push_back(frame.clone());
    count++;
    frame = loader_->readFrame();
  }

  return core::extractVideoDominantColors(frames, 3);
}

std::vector<int> VideoHandler::detectSceneChanges(double threshold) {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return {};

  cv::Mat prevFrame = loader_->readFrame();
  if (prevFrame.empty())
    return {};

  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

  std::vector<int> sceneChanges;
  int frameIndex = 1;
  cv::Mat currFrame = loader_->readFrame();
  while (!currFrame.empty()) {
    cv::Mat grayCurr;
    if (currFrame.channels() == 1)
      grayCurr = currFrame;
    else
      cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);

    double motion = core::calculateFrameMotion(prevGray, grayCurr);
    if (motion > threshold) {
      sceneChanges.push_back(frameIndex);
    }
    prevGray = grayCurr;
    frameIndex++;
    currFrame = loader_->readFrame();
  }
  return sceneChanges;
}

double VideoHandler::getFrameRateStability() {
  double fps = getFPS();
  if (fps <= 0.0)
    return -1.0;
  return 0.0; // Simplified placeholder
}

double VideoHandler::getColorConsistency() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  std::vector<double> brightnesses;
  int count = 0;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty() && count < 50) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    brightnesses.push_back(brightness);
    count++;
    frame = loader_->readFrame();
  }

  return core::calculateColorConsistency(brightnesses);
}

double VideoHandler::getOpticalFlowMagnitude() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  std::vector<cv::Mat> frames;
  int count = 0;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty() && count <= core::kOpticalFlowMaxPairs) {
    frames.push_back(frame.clone());
    count++;
    frame = loader_->readFrame();
  }

  return core::calculateOpticalFlowMagnitude(frames,
                                             core::kOpticalFlowMaxPairs);
}

bool VideoHandler::hasAudioTrack() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return false;
  return loader_->getProperty(cv::CAP_PROP_AUDIO_BASE_INDEX) >= 0.0;
}

ShotLengthStats VideoHandler::getShotLengthStats(double threshold) {
  std::vector<int> changes = detectSceneChanges(threshold);
  int totalFrames = getFrameCount();
  return core::calculateShotLengthStats(changes, totalFrames);
}

double VideoHandler::getFlickerScore() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  std::vector<double> brightnesses;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty() && static_cast<int>(brightnesses.size()) <
                               core::kMaxBrightnessCurveFrames) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    brightnesses.push_back(brightness);
    frame = loader_->readFrame();
  }

  return core::calculateFlickerScore(brightnesses);
}

int VideoHandler::getBestThumbnailIndex() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1;

  int totalFrames = loader_->getFrameCount();
  int stepSize = std::max(1, totalFrames / 20);

  int bestIndex = 0;
  double bestScore = -1.0;
  int frameIndex = 0;

  cv::Mat frame = loader_->readFrame();
  while (!frame.empty()) {
    if (frameIndex % stepSize == 0) {
      core::evaluateThumbnailFrame(frame, frameIndex, bestScore, bestIndex);
    }
    frameIndex++;
    frame = loader_->readFrame();
  }
  return bestIndex;
}

std::vector<double> VideoHandler::getTemporalBrightnessCurve() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return {};

  std::vector<double> curve;
  cv::Mat frame = loader_->readFrame();
  while (!frame.empty() &&
         static_cast<int>(curve.size()) < core::kMaxBrightnessCurveFrames) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    curve.push_back(brightness);
    frame = loader_->readFrame();
  }
  return curve;
}

std::string VideoHandler::getCodecFourcc() {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return "";
  double fourccCode = loader_->getProperty(cv::CAP_PROP_FOURCC);
  return core::decodeFourcc(fourccCode);
}

double VideoHandler::compareVideos(const std::string &otherFilename) {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  auto loader2 = std::make_unique<OpenCVVideoLoader>();
  if (!loader2->open(otherFilename))
    return -1.0;

  std::vector<cv::Mat> frames1, frames2;
  for (int i = 0; i < core::kVideoCompareSampleCount; ++i) {
    cv::Mat f1 = loader_->readFrame();
    cv::Mat f2 = loader2->readFrame();
    if (f1.empty() || f2.empty())
      break;
    frames1.push_back(f1);
    frames2.push_back(f2);
  }

  return core::compareVideoHistograms(frames1, frames2,
                                      core::kVideoCompareSampleCount);
}

VideoMetrics VideoHandler::getMetrics() {
  VideoMetrics m{};
  m.frame_count = getFrameCount();
  m.fps = getFPS();
  auto [w, h] = getResolution();
  m.width = w;
  m.height = h;
  m.duration = getDuration();
  m.is_grayscale = isGrayscale();
  m.average_brightness = getAverageBrightness();
  if (!m.is_grayscale) {
    m.motion_score = getMotionScore();
  }
  m.dominant_colors = getDominantColors();
  m.frame_rate_stability = getFrameRateStability();
  m.color_consistency = getColorConsistency();
  m.optical_flow_magnitude = getOpticalFlowMagnitude();
  m.has_audio_track = hasAudioTrack();
  if (!m.is_grayscale) {
    m.shot_length_stats = getShotLengthStats();
  }
  m.flicker_score = getFlickerScore();
  m.best_thumbnail_frame = getBestThumbnailIndex();
  m.temporal_brightness_curve = getTemporalBrightnessCurve();
  m.codec_fourcc = getCodecFourcc();
  return m;
}

int getVideoFrameCount(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1;
  return handler.getFrameCount();
}

double getVideoFPS(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getFPS();
}

std::pair<int, int> getVideoResolution(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {-1, -1};
  return handler.getResolution();
}

double getVideoDuration(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getDuration();
}

cv::Mat extractFirstFrame(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return cv::Mat();
  return handler.extractFirstFrame();
}

double getVideoAverageBrightness(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getAverageBrightness();
}

bool isVideoGrayscale(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return false;
  return handler.isGrayscale();
}

bool saveFirstFrameAsImage(const std::string &videoPath,
                           const std::string &imagePath) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(videoPath))
    return false;
  return handler.saveFirstFrameAsImage(imagePath);
}

double getVideoMotionScore(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getMotionScore();
}

std::vector<std::array<double, 3>>
getVideoDominantColors(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.getDominantColors();
}

std::vector<int> detectVideoSceneChanges(const std::string &filename,
                                         double threshold) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.detectSceneChanges(threshold);
}

double getVideoFrameRateStability(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getFrameRateStability();
}

double getVideoColorConsistency(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getColorConsistency();
}

double getVideoOpticalFlowMagnitude(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getOpticalFlowMagnitude();
}

bool videoHasAudioTrack(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return false;
  return handler.hasAudioTrack();
}

ShotLengthStats getVideoShotLengthStats(const std::string &filename,
                                        double threshold) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {-1.0, -1.0, -1.0, -1.0, -1};
  return handler.getShotLengthStats(threshold);
}

double getVideoFlickerScore(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getFlickerScore();
}

int getVideoBestThumbnailIndex(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1;
  return handler.getBestThumbnailIndex();
}

std::vector<double>
getVideoTemporalBrightnessCurve(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.getTemporalBrightnessCurve();
}

std::string getVideoCodecFourcc(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return "";
  return handler.getCodecFourcc();
}

double compareVideos(const std::string &filename1,
                     const std::string &filename2) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename1))
    return -1.0;
  return handler.compareVideos(filename2);
}

VideoMetrics getVideoMetrics(const std::string &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return VideoMetrics{};
  return handler.getMetrics();
}

} // namespace vidicant
