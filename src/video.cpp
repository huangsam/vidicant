// File: video.cpp
// Implementation of video loader, VideoHandler coordinator, and public video
// functions.

#include "vidicant/video.hpp"
#include "vidicant/core/video_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
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
  int lastSceneFrame = -100;
  constexpr int kSceneDebounce = 5;

  while (true) {
    if (stride > 1) {
      if (stride <= 10 || !loader_->seekFrame(frameIndex)) {
        for (int i = 0; i < stride - 1; ++i) {
          if (!loader_->grabFrame())
            break;
        }
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
    if (motion > threshold && (frameIndex - lastSceneFrame >= kSceneDebounce)) {
      sceneChanges.push_back(frameIndex);
      lastSceneFrame = frameIndex;
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
    core::evaluateThumbnailFrame(frame, frameIndex, bestScore, bestIndex);
    frameIndex += stepSize;
    if (stepSize > 1) {
      if (!loader_->seekFrame(frameIndex)) {
        for (int i = 0; i < stepSize - 1; ++i) {
          if (!loader_->grabFrame())
            break;
        }
      }
    }
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
  if (!loader_ || (!filename_.empty() && !loader_->open(filename_)))
    return std::nullopt;

  auto frameCountOpt = getFrameCount();
  int totalFrames = frameCountOpt.value_or(0);

  VideoMetrics m{};
  m.frame_count = totalFrames;
  m.fps = getFPS().value_or(0.0);
  auto res = getResolution();
  if (res.has_value()) {
    m.width = res->first;
    m.height = res->second;
  }
  m.duration = getDuration().value_or(0.0);
  m.has_audio_track = hasAudioTrack();
  m.codec_fourcc = getCodecFourcc();
  m.frame_rate_stability = getFrameRateStability();

  int effective_stride = std::max(1, options.sample_stride);
  if (options.sample_fps > 0.0 && m.fps > 0.0) {
    effective_stride =
        std::max(1, static_cast<int>(std::round(m.fps / options.sample_fps)));
  }

  // Open loader to start streaming frames
  if (!filename_.empty() && !loader_->open(filename_))
    return std::nullopt;

  cv::Mat firstFrame = loader_->readFrame();
  if (firstFrame.empty())
    return std::nullopt;

  if (m.width <= 0)
    m.width = firstFrame.cols;
  if (m.height <= 0)
    m.height = firstFrame.rows;
  m.is_grayscale = (firstFrame.channels() == 1);

  auto makeOptFlowFrame = [](const cv::Mat &frame) -> cv::Mat {
    cv::Mat gray;
    if (frame.channels() == 1)
      gray = frame;
    else
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    if (gray.cols > 320 || gray.rows > 320) {
      double scale = 320.0 / std::max(gray.cols, gray.rows);
      cv::Mat small;
      cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
      return small;
    }
    return gray.clone();
  };

  auto makeDominantColorFrame = [](const cv::Mat &frame) -> cv::Mat {
    if (frame.cols > 64 || frame.rows > 64) {
      cv::Mat small;
      cv::resize(frame, small, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
      return small;
    }
    return frame.clone();
  };

  // Optical flow collection (first few frames)
  std::vector<cv::Mat> optFlowFrames;
  optFlowFrames.push_back(makeOptFlowFrame(firstFrame));

  // Dominant color sampling: sample up to 5 frames throughout video
  std::vector<cv::Mat> dominantColorFrames;
  dominantColorFrames.push_back(makeDominantColorFrame(firstFrame));
  int dominantColorInterval =
      (totalFrames > 0) ? std::max(1, totalFrames / 5) : 30;

  // Brightness curve & color consistency
  std::vector<double> brightnesses;
  auto getFrameBrightness = [](const cv::Mat &f) -> double {
    cv::Scalar mean = cv::mean(f);
    return (f.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
  };
  brightnesses.push_back(getFrameBrightness(firstFrame));
  int maxBrightness = options.max_brightness_frames > 0
                          ? options.max_brightness_frames
                          : core::kMaxBrightnessCurveFrames;

  // Best thumbnail
  int bestThumbnailIndex = 0;
  double bestThumbnailScore = -1.0;
  int thumbnailStepSize =
      (totalFrames > 0) ? std::max(1, totalFrames / 20) : 10;
  core::evaluateThumbnailFrame(firstFrame, 0, bestThumbnailScore,
                               bestThumbnailIndex);

  // Scene changes & motion tracking
  std::vector<int> sceneChanges;
  double totalMotion = 0.0;
  int motionPairs = 0;
  cv::Mat prevGray;
  if (firstFrame.channels() == 1)
    prevGray = firstFrame;
  else
    cv::cvtColor(firstFrame, prevGray, cv::COLOR_BGR2GRAY);

  // Scene thumbnail export setup
  bool exportScenes = !options.export_scenes_dir.empty();
  if (exportScenes) {
    std::error_code ec;
    std::filesystem::create_directories(options.export_scenes_dir, ec);
  }
  std::string stem = filename_.stem().string();
  if (stem.empty())
    stem = "video";

  int frameIndex = effective_stride;
  int lastSceneFrame = -100;
  constexpr int kSceneDebounce = 5;

  while (true) {
    if (effective_stride > 1) {
      bool ok = true;
      for (int i = 0; i < effective_stride - 1; ++i) {
        if (!loader_->grabFrame()) {
          ok = false;
          break;
        }
      }
      if (!ok)
        break;
    }

    cv::Mat currFrame = loader_->readFrame();
    if (currFrame.empty())
      break;

    // Optical flow: first kOpticalFlowMaxPairs frames
    if (static_cast<int>(optFlowFrames.size()) <= core::kOpticalFlowMaxPairs) {
      optFlowFrames.push_back(makeOptFlowFrame(currFrame));
    }

    // Dominant colors: sample up to 5 frames
    if (dominantColorFrames.size() < 5 &&
        frameIndex % dominantColorInterval == 0) {
      dominantColorFrames.push_back(makeDominantColorFrame(currFrame));
    }

    // Brightness curve: up to max_brightness_frames
    if (static_cast<int>(brightnesses.size()) < maxBrightness) {
      brightnesses.push_back(getFrameBrightness(currFrame));
    }

    // Thumbnail evaluation
    if (frameIndex % thumbnailStepSize == 0) {
      core::evaluateThumbnailFrame(currFrame, frameIndex, bestThumbnailScore,
                                   bestThumbnailIndex);
    }

    // Motion & scene detection
    if (!m.is_grayscale) {
      cv::Mat currGray;
      if (currFrame.channels() == 1)
        currGray = currFrame;
      else
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

      double motion = core::calculateFrameMotion(prevGray, currGray);
      if (options.max_motion_frames <= 0 ||
          motionPairs < options.max_motion_frames) {
        totalMotion += motion;
        motionPairs++;
      }

      if (motion > options.scene_change_threshold &&
          (frameIndex - lastSceneFrame >= kSceneDebounce)) {
        sceneChanges.push_back(frameIndex);
        lastSceneFrame = frameIndex;

        if (exportScenes) {
          cv::Mat laplacian;
          cv::Laplacian(currGray, laplacian, CV_64F);
          cv::Scalar meanVal, stddevVal;
          cv::meanStdDev(laplacian, meanVal, stddevVal);
          double sharpness = stddevVal[0] * stddevVal[0];

          std::ostringstream filenameStream;
          filenameStream << stem << "_scene_" << sceneChanges.size()
                         << "_frame_" << frameIndex << ".jpg";
          std::filesystem::path thumbPath =
              options.export_scenes_dir / filenameStream.str();
          if (cv::imwrite(thumbPath.string(), currFrame)) {
            SceneThumbnail st;
            st.scene_index = static_cast<int>(sceneChanges.size());
            st.frame_index = frameIndex;
            st.timestamp_seconds =
                (m.fps > 0.0) ? (static_cast<double>(frameIndex) / m.fps) : 0.0;
            st.thumbnail_path = thumbPath.string();
            st.sharpness_score = sharpness;
            m.scene_thumbnails.push_back(st);
          }
        }
      }
      prevGray = currGray;
    }

    frameIndex += effective_stride;
  }

  int actualFrames = frameIndex;
  if (m.frame_count <= 0) {
    m.frame_count = actualFrames;
  }
  if (m.duration <= 0.0 && m.fps > 0.0) {
    m.duration = static_cast<double>(m.frame_count) / m.fps;
  }

  // Populate aggregated metrics
  m.temporal_brightness_curve = brightnesses;
  if (!brightnesses.empty()) {
    double sum = std::accumulate(brightnesses.begin(), brightnesses.end(), 0.0);
    m.average_brightness = sum / brightnesses.size();
  }
  m.flicker_score = core::calculateFlickerScore(brightnesses);
  m.color_consistency = core::calculateColorConsistency(brightnesses);
  m.best_thumbnail_frame = bestThumbnailIndex;
  m.motion_score = (motionPairs > 0) ? (totalMotion / motionPairs) : 0.0;
  m.dominant_colors = core::extractVideoDominantColors(
      dominantColorFrames, options.dominant_colors_k);
  m.optical_flow_magnitude = core::calculateOpticalFlowMagnitude(
      optFlowFrames, core::kOpticalFlowMaxPairs);

  m.scene_changes = sceneChanges;
  if (!m.is_grayscale) {
    int totalForStats = std::max(m.frame_count, actualFrames);
    m.shot_length_stats =
        core::calculateShotLengthStats(sceneChanges, totalForStats);
  }

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
