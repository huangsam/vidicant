// File: video.cpp
// Implementation of video loader, VideoHandler coordinator, and public video
// functions.

#include "vidicant/video.hpp"
#include "vidicant/core/video_ops.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace vidicant {

bool OpenCVVideoLoader::open(const std::filesystem::path &filename) {
  cap_.open(filename.string());
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

bool OpenCVVideoLoader::seekFrame(int frameIndex) {
  return cap_.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
}

bool OpenCVVideoLoader::grabFrame() { return cap_.grab(); }

double OpenCVVideoLoader::getProperty(int propId) { return cap_.get(propId); }

VideoHandler::VideoHandler(std::unique_ptr<IVideoLoader> loader)
    : loader_(std::move(loader)) {}

bool VideoHandler::open(const std::filesystem::path &filename) {
  filename_ = filename;
  return loader_ && loader_->open(filename);
}

std::optional<int> VideoHandler::getFrameCount() {
  if (!loader_)
    return std::nullopt;
  int count = loader_->getFrameCount();
  if (count < 0)
    return std::nullopt;
  return count;
}

std::optional<double> VideoHandler::getFPS() {
  if (!loader_)
    return std::nullopt;
  double fps = loader_->getFPS();
  if (fps < 0.0)
    return std::nullopt;
  return fps;
}

std::optional<std::pair<int, int>> VideoHandler::getResolution() {
  if (!loader_)
    return std::nullopt;
  auto res = loader_->getResolution();
  if (res.first < 0 || res.second < 0)
    return std::nullopt;
  return res;
}

std::optional<double> VideoHandler::getDuration() {
  if (!loader_)
    return std::nullopt;
  auto countOpt = getFrameCount();
  auto fpsOpt = getFPS();
  if (!countOpt.has_value() || !fpsOpt.has_value() || *fpsOpt <= 0.0)
    return std::nullopt;
  return static_cast<double>(*countOpt) / *fpsOpt;
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

bool VideoHandler::saveFirstFrameAsImage(
    const std::filesystem::path &imagePath) {
  cv::Mat frame = extractFirstFrame();
  if (frame.empty())
    return false;
  return cv::imwrite(imagePath.string(), frame);
}

double VideoHandler::getMotionScore(int stride) {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return -1.0;

  stride = std::max(1, stride);
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
  int frameIndex = stride;
  while (frameCount < 50) {
    if (stride > 1) {
      if (stride <= 10) {
        for (int i = 0; i < stride - 1; ++i) {
          if (!loader_->grabFrame())
            break;
        }
      } else {
        loader_->seekFrame(frameIndex);
      }
    }

    cv::Mat currFrame = loader_->readFrame();
    if (currFrame.empty())
      break;

    cv::Mat grayCurr;
    if (currFrame.channels() == 1)
      grayCurr = currFrame;
    else
      cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);

    totalMotion += core::calculateFrameMotion(prevGray, grayCurr);
    prevGray = grayCurr;
    frameCount++;
    frameIndex += stride;
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

std::vector<int> VideoHandler::detectSceneChanges(double threshold,
                                                  int stride) {
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return {};

  stride = std::max(1, stride);
  cv::Mat prevFrame = loader_->readFrame();
  if (prevFrame.empty())
    return {};

  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

  std::vector<int> sceneChanges;
  int frameIndex = stride;
  while (true) {
    if (stride > 1) {
      if (stride <= 10) {
        for (int i = 0; i < stride - 1; ++i) {
          if (!loader_->grabFrame())
            break;
        }
      } else {
        loader_->seekFrame(frameIndex);
      }
    }

    cv::Mat currFrame = loader_->readFrame();
    if (currFrame.empty())
      break;

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
    frameIndex += stride;
  }
  return sceneChanges;
}

double VideoHandler::getFrameRateStability() {
  auto fps = getFPS();
  if (!fps.has_value() || *fps <= 0.0)
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

ShotLengthStats VideoHandler::getShotLengthStats(double threshold, int stride) {
  std::vector<int> changes = detectSceneChanges(threshold, stride);
  int totalFrames = getFrameCount().value_or(0);
  return core::calculateShotLengthStats(changes, totalFrames);
}

std::vector<SceneThumbnail>
VideoHandler::exportSceneThumbnails(const std::vector<int> &sceneChanges,
                                    const std::filesystem::path &outputDir) {
  if (outputDir.empty() || sceneChanges.empty())
    return {};

  std::error_code ec;
  std::filesystem::create_directories(outputDir, ec);
  if (ec)
    return {};

  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return {};

  double fps = getFPS().value_or(30.0);
  if (fps <= 0.0)
    fps = 30.0;

  std::string stem = filename_.stem().string();
  if (stem.empty())
    stem = "video";

  std::vector<SceneThumbnail> exported;
  for (size_t i = 0; i < sceneChanges.size(); ++i) {
    int targetFrame = sceneChanges[i];
    if (!loader_->seekFrame(targetFrame)) {
      continue;
    }
    cv::Mat frame = loader_->readFrame();
    if (frame.empty())
      continue;

    cv::Mat gray;
    if (frame.channels() == 1)
      gray = frame;
    else
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double sharpness = stddev[0] * stddev[0];

    std::ostringstream filenameStream;
    filenameStream << stem << "_scene_" << (i + 1) << "_frame_" << targetFrame
                   << ".jpg";
    std::filesystem::path thumbPath = outputDir / filenameStream.str();

    if (cv::imwrite(thumbPath.string(), frame)) {
      SceneThumbnail st;
      st.scene_index = static_cast<int>(i + 1);
      st.frame_index = targetFrame;
      st.timestamp_seconds = static_cast<double>(targetFrame) / fps;
      st.thumbnail_path = thumbPath.string();
      st.sharpness_score = sharpness;
      exported.push_back(st);
    }
  }

  return exported;
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

double VideoHandler::compareVideos(const std::filesystem::path &otherFilename) {
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

std::optional<VideoMetrics> VideoHandler::getMetrics() {
  return getMetrics(VideoAnalysisOptions{});
}

std::optional<VideoMetrics>
VideoHandler::getMetrics(const VideoAnalysisOptions &options) {
  auto frameCountOpt = getFrameCount();
  if (!frameCountOpt.has_value() || *frameCountOpt <= 0)
    return std::nullopt;

  VideoMetrics m{};
  m.frame_count = *frameCountOpt;
  m.fps = getFPS().value_or(0.0);
  auto res = getResolution();
  if (res.has_value()) {
    m.width = res->first;
    m.height = res->second;
  }
  m.duration = getDuration().value_or(0.0);
  m.is_grayscale = isGrayscale();
  m.average_brightness = getAverageBrightness();

  int effective_stride = std::max(1, options.sample_stride);
  if (options.sample_fps > 0.0 && m.fps > 0.0) {
    effective_stride =
        std::max(1, static_cast<int>(std::round(m.fps / options.sample_fps)));
  }

  if (!m.is_grayscale) {
    m.motion_score = getMotionScore(effective_stride);
  }
  m.dominant_colors = getDominantColors();
  m.frame_rate_stability = getFrameRateStability();
  m.color_consistency = getColorConsistency();
  m.optical_flow_magnitude = getOpticalFlowMagnitude();
  m.has_audio_track = hasAudioTrack();

  std::vector<int> sceneChanges;
  if (!m.is_grayscale) {
    sceneChanges =
        detectSceneChanges(options.scene_change_threshold, effective_stride);
    m.shot_length_stats =
        core::calculateShotLengthStats(sceneChanges, m.frame_count);
  }

  if (!options.export_scenes_dir.empty() && !sceneChanges.empty()) {
    m.scene_thumbnails =
        exportSceneThumbnails(sceneChanges, options.export_scenes_dir);
  }

  m.flicker_score = getFlickerScore();
  m.best_thumbnail_frame = getBestThumbnailIndex();
  m.temporal_brightness_curve = getTemporalBrightnessCurve();
  m.codec_fourcc = getCodecFourcc();
  return m;
}

std::optional<int> getVideoFrameCount(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return std::nullopt;
  return handler.getFrameCount();
}

std::optional<double> getVideoFPS(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return std::nullopt;
  return handler.getFPS();
}

std::optional<std::pair<int, int>>
getVideoResolution(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return std::nullopt;
  return handler.getResolution();
}

std::optional<double> getVideoDuration(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return std::nullopt;
  return handler.getDuration();
}

cv::Mat extractFirstFrame(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return cv::Mat();
  return handler.extractFirstFrame();
}

double getVideoAverageBrightness(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getAverageBrightness();
}

bool isVideoGrayscale(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return false;
  return handler.isGrayscale();
}

bool saveFirstFrameAsImage(const std::filesystem::path &videoPath,
                           const std::filesystem::path &imagePath) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(videoPath))
    return false;
  return handler.saveFirstFrameAsImage(imagePath);
}

double getVideoMotionScore(const std::filesystem::path &filename, int stride) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getMotionScore(stride);
}

std::vector<std::array<double, 3>>
getVideoDominantColors(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.getDominantColors();
}

std::vector<int> detectVideoSceneChanges(const std::filesystem::path &filename,
                                         double threshold, int stride) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.detectSceneChanges(threshold, stride);
}

std::vector<SceneThumbnail>
exportVideoSceneThumbnails(const std::filesystem::path &filename,
                           const std::vector<int> &sceneChanges,
                           const std::filesystem::path &outputDir) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.exportSceneThumbnails(sceneChanges, outputDir);
}

double getVideoFrameRateStability(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getFrameRateStability();
}

double getVideoColorConsistency(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getColorConsistency();
}

double getVideoOpticalFlowMagnitude(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getOpticalFlowMagnitude();
}

bool videoHasAudioTrack(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return false;
  return handler.hasAudioTrack();
}

ShotLengthStats getVideoShotLengthStats(const std::filesystem::path &filename,
                                        double threshold, int stride) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {-1.0, -1.0, -1.0, -1.0, -1};
  return handler.getShotLengthStats(threshold, stride);
}

double getVideoFlickerScore(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1.0;
  return handler.getFlickerScore();
}

int getVideoBestThumbnailIndex(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return -1;
  return handler.getBestThumbnailIndex();
}

std::vector<double>
getVideoTemporalBrightnessCurve(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return {};
  return handler.getTemporalBrightnessCurve();
}

std::string getVideoCodecFourcc(const std::filesystem::path &filename) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return "";
  return handler.getCodecFourcc();
}

double compareVideos(const std::filesystem::path &filename1,
                     const std::filesystem::path &filename2) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename1))
    return -1.0;
  return handler.compareVideos(filename2);
}

std::optional<VideoMetrics>
getVideoMetrics(const std::filesystem::path &filename) {
  return getVideoMetrics(filename, VideoAnalysisOptions{});
}

std::optional<VideoMetrics>
getVideoMetrics(const std::filesystem::path &filename,
                const VideoAnalysisOptions &options) {
  VideoHandler handler(std::make_unique<OpenCVVideoLoader>());
  if (!handler.open(filename))
    return std::nullopt;
  return handler.getMetrics(options);
}

} // namespace vidicant
