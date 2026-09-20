// File: video_ops.cpp
// Implementation of core CV algorithms for video frame processing.

#include "vidicant/core/video_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vidicant::core {

double calculateFrameMotion(const cv::Mat &prevGray, const cv::Mat &currGray) {
  if (prevGray.empty() || currGray.empty())
    return 0.0;
  cv::Mat diff;
  cv::absdiff(prevGray, currGray, diff);
  return cv::mean(diff)[0];
}

std::vector<std::array<double, 3>>
extractVideoDominantColors(const std::vector<cv::Mat> &frames, int k) {
  if (frames.empty() || k <= 0)
    return {};

  std::vector<cv::Mat> sampledMats;
  int totalPixels = 0;
  for (const auto &f : frames) {
    if (f.empty())
      continue;
    cv::Mat rgb;
    if (f.channels() == 1) {
      cv::cvtColor(f, rgb, cv::COLOR_GRAY2RGB);
    } else if (f.channels() == 4) {
      cv::cvtColor(f, rgb, cv::COLOR_BGRA2RGB);
    } else {
      cv::cvtColor(f, rgb, cv::COLOR_BGR2RGB);
    }

    cv::Mat small;
    if (rgb.rows > 64 || rgb.cols > 64) {
      cv::resize(rgb, small, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    } else {
      small = rgb;
    }

    cv::Mat floatMat;
    small.convertTo(floatMat, CV_32F);
    floatMat = floatMat.reshape(1, static_cast<int>(floatMat.total()));
    totalPixels += floatMat.rows;
    sampledMats.push_back(floatMat);
  }

  if (sampledMats.empty() || totalPixels <= 0)
    return {};

  cv::Mat data(totalPixels, 3, CV_32F);
  int currentRow = 0;
  for (const auto &m : sampledMats) {
    m.copyTo(data.rowRange(currentRow, currentRow + m.rows));
    currentRow += m.rows;
  }

  int effectiveK = std::min(k, totalPixels);
  if (totalPixels <= effectiveK) {
    std::vector<std::array<double, 3>> dominantColors;
    dominantColors.reserve(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
      dominantColors.push_back(
          {data.at<float>(i, 0), data.at<float>(i, 1), data.at<float>(i, 2)});
    }
    return dominantColors;
  }

  std::vector<int> labels;
  cv::Mat centers;
  cv::kmeans(data, effectiveK, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              10, 1.0),
             3, cv::KMEANS_PP_CENTERS, centers);

  std::vector<std::array<double, 3>> dominantColors;
  dominantColors.reserve(effectiveK);
  for (int i = 0; i < effectiveK; ++i) {
    dominantColors.push_back({centers.at<float>(i, 0), centers.at<float>(i, 1),
                              centers.at<float>(i, 2)});
  }
  return dominantColors;
}

double calculateColorConsistency(const std::vector<double> &brightnesses) {
  if (brightnesses.empty())
    return -1.0;
  double mean = std::accumulate(brightnesses.begin(), brightnesses.end(), 0.0) /
                brightnesses.size();
  double variance = 0.0;
  for (double b : brightnesses) {
    variance += (b - mean) * (b - mean);
  }
  variance /= brightnesses.size();
  double stddev = std::sqrt(variance);
  return mean > 0.0 ? (stddev / mean) : 0.0;
}

double calculateOpticalFlowMagnitude(const std::vector<cv::Mat> &frames,
                                     int maxPairs) {
  if (frames.size() < 2)
    return 0.0;

  double totalMagnitude = 0.0;
  int pairCount = 0;
  size_t limit = std::min(frames.size() - 1, static_cast<size_t>(maxPairs));

  for (size_t i = 0; i < limit; ++i) {
    cv::Mat prevGray, currGray;
    if (frames[i].channels() == 1)
      prevGray = frames[i];
    else
      cv::cvtColor(frames[i], prevGray, cv::COLOR_BGR2GRAY);

    if (frames[i + 1].channels() == 1)
      currGray = frames[i + 1];
    else
      cv::cvtColor(frames[i + 1], currGray, cv::COLOR_BGR2GRAY);

    cv::Mat smallPrev, smallCurr;
    double scale = 1.0;
    if (prevGray.cols > 320 || prevGray.rows > 320) {
      scale = 320.0 / std::max(prevGray.cols, prevGray.rows);
      cv::resize(prevGray, smallPrev, cv::Size(), scale, scale, cv::INTER_AREA);
      cv::resize(currGray, smallCurr, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
      smallPrev = prevGray;
      smallCurr = currGray;
    }

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(smallPrev, smallCurr, flow, 0.5, 3, 15, 3, 5,
                                 1.2, 0);

    std::vector<cv::Mat> flowParts(2);
    cv::split(flow, flowParts);
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle);
    totalMagnitude += cv::mean(magnitude)[0] / scale;
    pairCount++;
  }

  return pairCount > 0 ? (totalMagnitude / pairCount) : 0.0;
}

ShotLengthStats calculateShotLengthStats(const std::vector<int> &sceneChanges,
                                         int totalFrames) {
  std::vector<double> lengths;
  int prev = 0;
  for (int changeFrame : sceneChanges) {
    if (changeFrame > prev) {
      lengths.push_back(static_cast<double>(changeFrame - prev));
      prev = changeFrame;
    }
  }
  if (totalFrames > prev) {
    lengths.push_back(static_cast<double>(totalFrames - prev));
  } else if (lengths.empty() && totalFrames > 0) {
    lengths.push_back(static_cast<double>(totalFrames));
  }

  if (lengths.empty())
    return {0.0, 0.0, 0.0, 0.0, 0};

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

double calculateFlickerScore(const std::vector<double> &brightnesses) {
  if (brightnesses.size() < 2)
    return 0.0;

  std::vector<double> diffs;
  diffs.reserve(brightnesses.size() - 1);
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

int evaluateThumbnailFrame(const cv::Mat &frame, int frameIndex,
                           double &bestScore, int &bestIndex) {
  if (frame.empty())
    return bestIndex;

  cv::Mat gray;
  if (frame.channels() > 1)
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  else
    gray = frame;

  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar lMean, lStddev;
  cv::meanStdDev(laplacian, lMean, lStddev);
  double blurScore = lStddev[0] * lStddev[0];

  cv::Scalar brightness = cv::mean(gray);
  double brightnessPenalty = 1.0 - std::abs(brightness[0] - 128.0) / 128.0;
  brightnessPenalty = std::max(0.0, brightnessPenalty);

  double score = blurScore * brightnessPenalty;
  if (score > bestScore) {
    bestScore = score;
    bestIndex = frameIndex;
  }
  return bestIndex;
}

double compareVideoHistograms(const std::vector<cv::Mat> &frames1,
                              const std::vector<cv::Mat> &frames2,
                              int sampleCount) {
  const int histSize = 64;
  float range[] = {0, 256};
  const float *histRange = {range};

  size_t count = std::min(
      {frames1.size(), frames2.size(), static_cast<size_t>(sampleCount)});
  std::vector<double> distances;
  for (size_t i = 0; i < count; ++i) {
    const cv::Mat &f1 = frames1[i];
    const cv::Mat &f2 = frames2[i];
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

std::string decodeFourcc(double fourccCode) {
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

} // namespace vidicant::core
