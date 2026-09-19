// File: image_ops.cpp
// Implementation of core CV algorithms on cv::Mat instances.

#include "vidicant/core/image_ops.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace vidicant::core {

cv::Mat toGrayscale(const cv::Mat &image) {
  if (image.empty())
    return cv::Mat();
  if (image.channels() == 1)
    return image;
  cv::Mat gray;
  if (image.channels() == 4)
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  else
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

double calculateAverageBrightness(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Scalar mean = cv::mean(image);
  if (image.channels() == 1)
    return mean[0];
  return (mean[0] + mean[1] + mean[2]) / 3.0;
}

std::vector<std::array<double, 3>> extractDominantColors(const cv::Mat &image,
                                                         int k) {
  if (image.empty() || k <= 0)
    return {};

  // Standardize to 3-channel BGR
  cv::Mat bgr;
  if (image.channels() == 1) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else {
    bgr = image;
  }

  // Convert to RGB so dominant colors match documentation
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  // Downsample to max 128x128 for efficient k-means clustering
  cv::Mat sampleImg;
  if (rgb.rows > 128 || rgb.cols > 128) {
    cv::resize(rgb, sampleImg, cv::Size(128, 128), 0, 0, cv::INTER_AREA);
  } else {
    sampleImg = rgb;
  }

  int totalPixels = static_cast<int>(sampleImg.total());
  if (totalPixels <= 0)
    return {};

  int effectiveK = std::min(k, totalPixels);
  if (totalPixels <= effectiveK) {
    std::vector<std::array<double, 3>> dominantColors;
    dominantColors.reserve(totalPixels);
    for (int r = 0; r < sampleImg.rows; ++r) {
      for (int c = 0; c < sampleImg.cols; ++c) {
        cv::Vec3b p = sampleImg.at<cv::Vec3b>(r, c);
        dominantColors.push_back({static_cast<double>(p[0]),
                                  static_cast<double>(p[1]),
                                  static_cast<double>(p[2])});
      }
    }
    return dominantColors;
  }

  cv::Mat data;
  sampleImg.convertTo(data, CV_32F);
  data = data.reshape(1, totalPixels);

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

double calculateContrastRatio(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);
  double minVal = 0.0, maxVal = 0.0;
  cv::minMaxLoc(gray, &minVal, &maxVal);
  return maxVal > 0 ? maxVal / std::max(minVal, 1.0) : 0.0;
}

double calculateSaturationLevel(const cv::Mat &image) {
  if (image.empty() || image.channels() < 3)
    return -1.0;
  cv::Mat bgr = image;
  if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  }
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Scalar mean = cv::mean(hsv);
  return mean[1];
}

std::vector<std::vector<int>> calculateHistogram(const cv::Mat &image) {
  if (image.empty())
    return {};
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  std::vector<std::vector<int>> histograms;
  histograms.reserve(channels.size());
  for (const auto &channel : channels) {
    std::vector<int> hist(256, 0);
    if (channel.isContinuous()) {
      const uchar *p = channel.data;
      const size_t n = channel.total();
      for (size_t idx = 0; idx < n; ++idx) {
        hist[p[idx]]++;
      }
    } else {
      for (int i = 0; i < channel.rows; ++i) {
        const uchar *row = channel.ptr<uchar>(i);
        for (int j = 0; j < channel.cols; ++j) {
          hist[row[j]]++;
        }
      }
    }
    histograms.push_back(std::move(hist));
  }
  return histograms;
}

double calculateWhiteBalanceScore(const cv::Mat &image) {
  if (image.empty() || image.channels() < 3)
    return -1.0;
  cv::Scalar means = cv::mean(image);
  double b = means[0], g = means[1], r = means[2];
  double overall = (r + g + b) / 3.0;
  if (overall <= 0.0)
    return 0.0;
  double maxDev = std::max(
      {std::abs(r - overall), std::abs(g - overall), std::abs(b - overall)});
  return maxDev / overall;
}

std::vector<int> calculateHueHistogram(const cv::Mat &image) {
  if (image.empty() || image.channels() < 3)
    return {};
  cv::Mat bgr = image;
  if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  }
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);
  const cv::Mat &hue = channels[0];

  std::vector<int> hist(36, 0);
  if (hue.isContinuous()) {
    const uchar *p = hue.data;
    const size_t n = hue.total();
    for (size_t idx = 0; idx < n; ++idx) {
      hist[std::min(p[idx] / 5, 35)]++;
    }
  } else {
    for (int r = 0; r < hue.rows; ++r) {
      const uchar *row = hue.ptr<uchar>(r);
      for (int c = 0; c < hue.cols; ++c) {
        hist[std::min(row[c] / 5, 35)]++;
      }
    }
  }
  return hist;
}

int calculateEdgeCount(const cv::Mat &image) {
  if (image.empty())
    return -1;
  cv::Mat gray = toGrayscale(image);
  cv::Mat edges;
  cv::Canny(gray, edges, 100, 200);
  return cv::countNonZero(edges);
}

double calculateBlurScore(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0] * stddev[0];
}

double calculateEntropy(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);
  cv::Mat hist;
  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};
  cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
  hist /= gray.total();
  double entropy = 0.0;
  for (int i = 0; i < histSize; ++i) {
    float p = hist.at<float>(i);
    if (p > 0) {
      entropy -= p * std::log2(p);
    }
  }
  return entropy;
}

double calculateNoiseEstimate(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);
  int m = gray.rows, n = gray.cols;
  if (m <= 2 || n <= 2)
    return 0.0;

  cv::Mat grayF;
  gray.convertTo(grayF, CV_32F);
  static const float kData[9] = {1.0f,  -2.0f, 1.0f,  -2.0f, 4.0f,
                                 -2.0f, 1.0f,  -2.0f, 1.0f};
  cv::Mat kernel(3, 3, CV_32F, const_cast<float *>(kData));
  cv::Mat filtered;
  cv::filter2D(grayF, filtered, CV_32F, kernel);
  cv::Rect interior(1, 1, n - 2, m - 2);
  double sumAbs = cv::norm(filtered(interior), cv::NORM_L1);
  return std::sqrt(CV_PI / 2.0) / (6.0 * (m - 2) * (n - 2)) * sumAbs;
}

double calculateSymmetryScore(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);

  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};

  int midCol = gray.cols / 2;
  double hSym = 1.0;
  if (midCol > 0) {
    cv::Mat left = gray(cv::Rect(0, 0, midCol, gray.rows));
    cv::Mat right =
        gray(cv::Rect(gray.cols - midCol, 0, midCol, gray.rows)).clone();
    cv::flip(right, right, 1);
    cv::Mat histL, histR;
    cv::calcHist(&left, 1, 0, cv::Mat(), histL, 1, &histSize, &histRange);
    cv::calcHist(&right, 1, 0, cv::Mat(), histR, 1, &histSize, &histRange);
    cv::normalize(histL, histL, 0, 1, cv::NORM_MINMAX);
    cv::normalize(histR, histR, 0, 1, cv::NORM_MINMAX);
    hSym = cv::compareHist(histL, histR, cv::HISTCMP_CORREL);
  }

  int midRow = gray.rows / 2;
  double vSym = 1.0;
  if (midRow > 0) {
    cv::Mat top = gray(cv::Rect(0, 0, gray.cols, midRow));
    cv::Mat bottom =
        gray(cv::Rect(0, gray.rows - midRow, gray.cols, midRow)).clone();
    cv::flip(bottom, bottom, 0);
    cv::Mat histT, histB;
    cv::calcHist(&top, 1, 0, cv::Mat(), histT, 1, &histSize, &histRange);
    cv::calcHist(&bottom, 1, 0, cv::Mat(), histB, 1, &histSize, &histRange);
    cv::normalize(histT, histT, 0, 1, cv::NORM_MINMAX);
    cv::normalize(histB, histB, 0, 1, cv::NORM_MINMAX);
    vSym = cv::compareHist(histT, histB, cv::HISTCMP_CORREL);
  }

  return (hSym + vSym) / 2.0;
}

TextureFeatures calculateTextureFeatures(const cv::Mat &image) {
  if (image.empty())
    return {-1.0, -1.0, -1.0, -1.0};
  cv::Mat gray = toGrayscale(image);

  cv::Mat workImg = gray;
  if (gray.rows > 256 || gray.cols > 256) {
    cv::resize(gray, workImg, cv::Size(256, 256));
  }

  const int levels = 16;
  cv::Mat quantized;
  workImg.convertTo(quantized, CV_32S);
  quantized = quantized * levels / 256;

  cv::Mat glcm = cv::Mat::zeros(levels, levels, CV_64F);
  for (int r = 0; r < quantized.rows; ++r) {
    for (int c = 0; c < quantized.cols - 1; ++c) {
      int i = std::min(quantized.at<int>(r, c), levels - 1);
      int j = std::min(quantized.at<int>(r, c + 1), levels - 1);
      glcm.at<double>(i, j) += 1.0;
      glcm.at<double>(j, i) += 1.0;
    }
  }

  double total = cv::sum(glcm)[0];
  if (total > 0)
    glcm /= total;

  double contrast = 0.0, energy = 0.0, homogeneity = 0.0;
  double mean_i = 0.0, mean_j = 0.0;
  for (int i = 0; i < levels; ++i) {
    for (int j = 0; j < levels; ++j) {
      double p = glcm.at<double>(i, j);
      contrast += p * (i - j) * (i - j);
      energy += p * p;
      homogeneity += p / (1.0 + std::abs(i - j));
      mean_i += p * i;
      mean_j += p * j;
    }
  }

  double var_i = 0.0, var_j = 0.0, correlation = 0.0;
  for (int i = 0; i < levels; ++i) {
    for (int j = 0; j < levels; ++j) {
      double p = glcm.at<double>(i, j);
      var_i += p * (i - mean_i) * (i - mean_i);
      var_j += p * (j - mean_j) * (j - mean_j);
    }
  }
  double denom = std::sqrt(var_i * var_j);
  if (denom > 1e-10) {
    for (int i = 0; i < levels; ++i) {
      for (int j = 0; j < levels; ++j) {
        correlation +=
            glcm.at<double>(i, j) * (i - mean_i) * (j - mean_j) / denom;
      }
    }
  }

  return {contrast, energy, homogeneity, correlation};
}

double calculateSharpnessScore(const cv::Mat &image) {
  if (image.empty())
    return -1.0;
  cv::Mat gray = toGrayscale(image);
  cv::Mat gradX, gradY, magnitude;
  cv::Sobel(gray, gradX, CV_32F, 1, 0);
  cv::Sobel(gray, gradY, CV_32F, 0, 1);
  cv::magnitude(gradX, gradY, magnitude);
  return cv::mean(magnitude)[0];
}

std::string classifyNoiseType(const cv::Mat &image) {
  if (image.empty())
    return "";
  cv::Mat gray = toGrayscale(image);

  cv::Mat medianFiltered;
  cv::medianBlur(gray, medianFiltered, 3);
  cv::Mat residual;
  cv::absdiff(gray, medianFiltered, residual);

  constexpr double kSaltPepperResidualThreshold = 50.0;
  constexpr double kSaltPepperRatioThreshold = 0.01;

  cv::Mat extremeMask;
  cv::threshold(residual, extremeMask, kSaltPepperResidualThreshold, 255,
                cv::THRESH_BINARY);
  double extremeCount = cv::countNonZero(extremeMask);
  double extremeRatio = extremeCount / static_cast<double>(residual.total());

  if (extremeRatio > kSaltPepperRatioThreshold) {
    return "salt_and_pepper";
  }
  return "gaussian";
}

uint64_t calculatePerceptualHash(const cv::Mat &image) {
  if (image.empty())
    return 0;
  cv::Mat gray = toGrayscale(image);

  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(9, 8));

  uint64_t hash = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (resized.at<uchar>(r, c) > resized.at<uchar>(r, c + 1)) {
        hash |= (uint64_t(1) << (r * 8 + c));
      }
    }
  }
  return hash;
}

double calculateSSIM(const cv::Mat &img1, const cv::Mat &img2) {
  if (img1.empty() || img2.empty())
    return -1.0;

  cv::Mat gray1 = toGrayscale(img1);
  cv::Mat gray2 = toGrayscale(img2);

  if (gray1.size() != gray2.size()) {
    cv::resize(gray2, gray2, gray1.size());
  }

  gray1.convertTo(gray1, CV_32F);
  gray2.convertTo(gray2, CV_32F);

  const float c1 = 6.5025f;  // (0.01 * 255)^2
  const float c2 = 58.5225f; // (0.03 * 255)^2

  cv::Mat I1_sq, I2_sq, I1_I2;
  cv::multiply(gray1, gray1, I1_sq);
  cv::multiply(gray2, gray2, I2_sq);
  cv::multiply(gray1, gray2, I1_I2);

  cv::Mat mu1, mu2;
  cv::GaussianBlur(gray1, mu1, cv::Size(11, 11), 1.5);
  cv::GaussianBlur(gray2, mu2, cv::Size(11, 11), 1.5);

  cv::Mat mu1_sq, mu2_sq, mu1_mu2;
  cv::multiply(mu1, mu1, mu1_sq);
  cv::multiply(mu2, mu2, mu2_sq);
  cv::multiply(mu1, mu2, mu1_mu2);

  cv::Mat sigma1_sq, sigma2_sq, sigma12;
  cv::GaussianBlur(I1_sq, sigma1_sq, cv::Size(11, 11), 1.5);
  sigma1_sq -= mu1_sq;
  cv::GaussianBlur(I2_sq, sigma2_sq, cv::Size(11, 11), 1.5);
  sigma2_sq -= mu2_sq;
  cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  cv::Mat numerator, denominator;
  cv::multiply(2.0f * mu1_mu2 + c1, 2.0f * sigma12 + c2, numerator);
  cv::multiply(mu1_sq + mu2_sq + c1, sigma1_sq + sigma2_sq + c2, denominator);

  cv::Mat ssim_map;
  cv::divide(numerator, denominator, ssim_map);
  return static_cast<double>(cv::mean(ssim_map)[0]);
}

} // namespace vidicant::core
