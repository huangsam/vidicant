#include "vidicant/video.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

// Maximum number of consecutive frame pairs sampled for optical flow.
// Balances flow estimation accuracy against processing time.
constexpr int kOpticalFlowMaxPairs = 20;

// Maximum number of frames sampled for per-frame brightness analyses
// (flicker score, temporal brightness curve).  Caps memory and runtime for
// long videos while still capturing meaningful temporal patterns.
constexpr int kMaxBrightnessCurveFrames = 100;

// Number of frame pairs compared when computing video similarity.
// 10 pairs give a lightweight but representative cross-video histogram
// distance; increasing it improves accuracy at the cost of more I/O.
constexpr int kVideoCompareSampleCount = 10;

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
  return loader_->open(filename);
}

int VideoHandler::getFrameCount() { return loader_->getFrameCount(); }

double VideoHandler::getFPS() { return loader_->getFPS(); }

std::pair<int, int> VideoHandler::getResolution() {
  return loader_->getResolution();
}

double VideoHandler::getDuration() {
  int frameCount = loader_->getFrameCount();
  double fps = loader_->getFPS();
  if (fps <= 0)
    return -1.0;
  return frameCount / fps;
}

cv::Mat VideoHandler::extractFirstFrame() {
  // Create a temporary loader to read the first frame
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return cv::Mat();
  return tempLoader->readFrame();
}

double VideoHandler::getAverageBrightness() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1.0;
  cv::Mat frame;
  double totalBrightness = 0.0;
  int frameCount = 0;
  frame = tempLoader->readFrame();
  while (!frame.empty()) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    totalBrightness += brightness;
    frameCount++;
    if (frameCount > 100)
      break; // Limit to first 100 frames for speed
    frame = tempLoader->readFrame();
  }
  return frameCount > 0 ? totalBrightness / frameCount : -1.0;
}

bool VideoHandler::isGrayscale() {
  cv::Mat frame = extractFirstFrame();
  return frame.channels() == 1;
}

bool VideoHandler::saveFirstFrameAsImage(const std::string &imagePath) {
  cv::Mat frame = extractFirstFrame();
  if (frame.empty())
    return false;
  return cv::imwrite(imagePath, frame);
}

double VideoHandler::getMotionScore() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1.0;
  cv::Mat prevFrame = tempLoader->readFrame();
  if (prevFrame.empty())
    return 0.0;
  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

  double totalMotion = 0.0;
  int frameCount = 1;
  cv::Mat currFrame = tempLoader->readFrame();
  while (!currFrame.empty() && frameCount < 50) { // Limit to 50 frames
    cv::Mat grayCurr;
    if (currFrame.channels() == 1)
      grayCurr = currFrame;
    else
      cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);
    cv::Mat diff;
    cv::absdiff(prevGray, grayCurr, diff);
    cv::Scalar meanDiff = cv::mean(diff);
    totalMotion += meanDiff[0];
    prevGray = grayCurr;
    frameCount++;
    currFrame = tempLoader->readFrame();
  }
  return frameCount > 1 ? totalMotion / (frameCount - 1) : 0.0;
}

std::vector<std::array<double, 3>> VideoHandler::getDominantColors() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return {};
  std::vector<cv::Mat> frames;
  cv::Mat frame;
  int count = 0;
  frame = tempLoader->readFrame();
  while (!frame.empty() && count < 10) { // Sample first 10 frames
    frames.push_back(frame.clone());
    count++;
    frame = tempLoader->readFrame();
  }
  if (frames.empty())
    return {};

  // Concatenate all frames into one big image for k-means
  cv::Mat data;
  for (const auto &f : frames) {
    cv::Mat temp;
    f.convertTo(temp, CV_32F);
    temp = temp.reshape(1, temp.total());
    if (data.empty()) {
      data = temp;
    } else {
      cv::vconcat(data, temp, data);
    }
  }

  std::vector<int> labels;
  cv::Mat centers;
  cv::kmeans(data, 3, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              10, 1.0),
             3, cv::KMEANS_PP_CENTERS, centers);

  std::vector<std::array<double, 3>> dominantColors;
  for (int i = 0; i < 3; ++i) {
    dominantColors.push_back({centers.at<float>(i, 0), centers.at<float>(i, 1),
                              centers.at<float>(i, 2)});
  }
  return dominantColors;
}

std::vector<int> VideoHandler::detectSceneChanges(double threshold) {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return {};
  cv::Mat prevFrame = tempLoader->readFrame();
  if (prevFrame.empty())
    return {};
  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
  std::vector<int> sceneChanges;
  int frameIndex = 1;
  cv::Mat currFrame = tempLoader->readFrame();
  while (!currFrame.empty()) {
    cv::Mat grayCurr;
    if (currFrame.channels() == 1)
      grayCurr = currFrame;
    else
      cv::cvtColor(currFrame, grayCurr, cv::COLOR_BGR2GRAY);
    cv::Mat diff;
    cv::absdiff(prevGray, grayCurr, diff);
    cv::Scalar meanDiff = cv::mean(diff);
    if (meanDiff[0] > threshold) {
      sceneChanges.push_back(frameIndex);
    }
    prevGray = grayCurr;
    frameIndex++;
    currFrame = tempLoader->readFrame();
  }
  return sceneChanges;
}

double VideoHandler::getFrameRateStability() {
  // For simplicity, we'll check if FPS is consistent across the video
  // In a real implementation, you'd analyze frame timestamps
  double fps = getFPS();
  if (fps <= 0)
    return -1.0;
  // This is a simplified implementation - real frame rate stability
  // would require analyzing actual frame timing
  return 0.0; // Perfect stability for now (placeholder)
}

double VideoHandler::getColorConsistency() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1.0;
  cv::Mat frame;
  std::vector<double> brightnesses;
  int count = 0;
  frame = tempLoader->readFrame();
  while (!frame.empty() && count < 50) { // Sample 50 frames
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    brightnesses.push_back(brightness);
    count++;
    frame = tempLoader->readFrame();
  }
  if (brightnesses.empty())
    return -1.0;
  // Calculate coefficient of variation (lower = more consistent)
  double mean = std::accumulate(brightnesses.begin(), brightnesses.end(), 0.0) /
                brightnesses.size();
  double variance = 0.0;
  for (double b : brightnesses) {
    variance += (b - mean) * (b - mean);
  }
  variance /= brightnesses.size();
  double stddev = sqrt(variance);
  return mean > 0 ? (stddev / mean) : 0.0; // Coefficient of variation
}

double VideoHandler::getOpticalFlowMagnitude() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1.0;

  cv::Mat prevFrame = tempLoader->readFrame();
  if (prevFrame.empty())
    return 0.0;
  cv::Mat prevGray;
  if (prevFrame.channels() == 1)
    prevGray = prevFrame;
  else
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

  double totalMagnitude = 0.0;
  int pairCount = 0;

  cv::Mat currFrame = tempLoader->readFrame();
  while (!currFrame.empty() && pairCount < kOpticalFlowMaxPairs) {
    cv::Mat currGray;
    if (currFrame.channels() == 1)
      currGray = currFrame;
    else
      cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevGray, currGray, flow, 0.5, 3, 15, 3, 5,
                                 1.2, 0);

    // Split flow into x and y components, compute magnitude
    std::vector<cv::Mat> flowParts(2);
    cv::split(flow, flowParts);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle);
    totalMagnitude += cv::mean(magnitude)[0];

    prevGray = currGray;
    pairCount++;
    currFrame = tempLoader->readFrame();
  }
  return pairCount > 0 ? totalMagnitude / pairCount : 0.0;
}

bool VideoHandler::hasAudioTrack() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return false;
  return tempLoader->getProperty(cv::CAP_PROP_AUDIO_BASE_INDEX) >= 0.0;
}

ShotLengthStats VideoHandler::getShotLengthStats(double threshold) {
  std::vector<int> changes = detectSceneChanges(threshold);
  int totalFrames = loader_->getFrameCount();

  std::vector<double> lengths;
  int prev = 0;
  for (int changeFrame : changes) {
    lengths.push_back(static_cast<double>(changeFrame - prev));
    prev = changeFrame;
  }
  // Last shot to end of video
  lengths.push_back(static_cast<double>(totalFrames - prev));

  double meanVal =
      std::accumulate(lengths.begin(), lengths.end(), 0.0) / lengths.size();
  double minVal = *std::min_element(lengths.begin(), lengths.end());
  double maxVal = *std::max_element(lengths.begin(), lengths.end());

  double variance = 0.0;
  for (double l : lengths)
    variance += (l - meanVal) * (l - meanVal);
  variance /= lengths.size();

  return {meanVal, std::sqrt(variance), minVal, maxVal,
          static_cast<int>(lengths.size())};
}

double VideoHandler::getFlickerScore() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1.0;

  std::vector<double> brightnesses;
  cv::Mat frame = tempLoader->readFrame();
  while (!frame.empty() &&
         static_cast<int>(brightnesses.size()) < kMaxBrightnessCurveFrames) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    brightnesses.push_back(brightness);
    frame = tempLoader->readFrame();
  }

  if (brightnesses.size() < 2)
    return 0.0;

  // Compute first differences
  std::vector<double> diffs;
  for (size_t i = 1; i < brightnesses.size(); ++i) {
    diffs.push_back(brightnesses[i] - brightnesses[i - 1]);
  }

  double meanDiff =
      std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
  double variance = 0.0;
  for (double d : diffs)
    variance += (d - meanDiff) * (d - meanDiff);
  variance /= diffs.size();
  return std::sqrt(variance);
}

int VideoHandler::getBestThumbnailIndex() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return -1;

  int totalFrames = tempLoader->getFrameCount();
  // Sample ~20 evenly-spaced frames across the full video.
  int stepSize = std::max(1, totalFrames / 20);

  int bestIndex = 0;
  double bestScore = -1.0;
  int frameIndex = 0;

  cv::Mat frame = tempLoader->readFrame();
  while (!frame.empty()) {
    if (frameIndex % stepSize == 0) {
      cv::Mat gray;
      if (frame.channels() > 1)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      else
        gray = frame;

      // Sharpness: Laplacian variance
      cv::Mat laplacian;
      cv::Laplacian(gray, laplacian, CV_64F);
      cv::Scalar lMean, lStddev;
      cv::meanStdDev(laplacian, lMean, lStddev);
      double blurScore = lStddev[0] * lStddev[0];

      // Brightness score: parabola penalizing very dark or very bright frames
      cv::Scalar brightness = cv::mean(gray);
      double brightnessPenalty = 1.0 - std::abs(brightness[0] - 128.0) / 128.0;
      brightnessPenalty = std::max(0.0, brightnessPenalty);

      double score = blurScore * brightnessPenalty;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = frameIndex;
      }
    }
    frameIndex++;
    frame = tempLoader->readFrame();
  }
  return bestIndex;
}

std::vector<double> VideoHandler::getTemporalBrightnessCurve() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return {};

  std::vector<double> curve;
  cv::Mat frame = tempLoader->readFrame();
  while (!frame.empty() &&
         static_cast<int>(curve.size()) < kMaxBrightnessCurveFrames) {
    cv::Scalar mean = cv::mean(frame);
    double brightness =
        (frame.channels() == 1) ? mean[0] : (mean[0] + mean[1] + mean[2]) / 3.0;
    curve.push_back(brightness);
    frame = tempLoader->readFrame();
  }
  return curve;
}

std::string VideoHandler::getCodecFourcc() {
  auto tempLoader = std::make_unique<OpenCVVideoLoader>();
  if (!tempLoader->open(filename_))
    return "";

  double fourccCode = tempLoader->getProperty(cv::CAP_PROP_FOURCC);
  if (fourccCode <= 0.0)
    return "";

  int fourcc = static_cast<int>(fourccCode);
  char chars[5];
  chars[0] = static_cast<char>(fourcc & 0xFF);
  chars[1] = static_cast<char>((fourcc >> 8) & 0xFF);
  chars[2] = static_cast<char>((fourcc >> 16) & 0xFF);
  chars[3] = static_cast<char>((fourcc >> 24) & 0xFF);
  chars[4] = '\0';
  return std::string(chars);
}

double VideoHandler::compareVideos(const std::string &otherFilename) {
  auto loader1 = std::make_unique<OpenCVVideoLoader>();
  auto loader2 = std::make_unique<OpenCVVideoLoader>();
  if (!loader1->open(filename_) || !loader2->open(otherFilename))
    return -1.0;

  const int sampleCount = kVideoCompareSampleCount;
  const int histSize = 64;
  float range[] = {0, 256};
  const float *histRange = {range};

  std::vector<double> distances;
  for (int i = 0; i < sampleCount; ++i) {
    cv::Mat f1 = loader1->readFrame();
    cv::Mat f2 = loader2->readFrame();
    if (f1.empty() || f2.empty())
      break;

    cv::Mat g1, g2;
    if (f1.channels() == 1)
      g1 = f1;
    else
      cv::cvtColor(f1, g1, cv::COLOR_BGR2GRAY);
    if (f2.channels() == 1)
      g2 = f2;
    else
      cv::cvtColor(f2, g2, cv::COLOR_BGR2GRAY);

    cv::Mat h1, h2;
    cv::calcHist(&g1, 1, 0, cv::Mat(), h1, 1, &histSize, &histRange);
    cv::calcHist(&g2, 1, 0, cv::Mat(), h2, 1, &histSize, &histRange);
    cv::normalize(h1, h1, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(h2, h2, 1.0, 0.0, cv::NORM_L1);

    double dist = cv::compareHist(h1, h2, cv::HISTCMP_BHATTACHARYYA);
    distances.push_back(dist);
  }

  if (distances.empty())
    return -1.0;
  double avgDist = std::accumulate(distances.begin(), distances.end(), 0.0) /
                   distances.size();
  const double similarity = 1.0 - avgDist;
  return std::clamp(similarity, 0.0, 1.0);
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
    return {0.0, 0.0, 0.0, 0.0, 0};
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
  VideoMetrics m{};
  m.frame_count = -1;
  if (!handler.open(filename))
    return m;
  return handler.getMetrics();
}

} // namespace vidicant
