#include "vidicant/image.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_map>

cv::Mat OpenCVImageLoader::imread(const std::string &filename) {
  return cv::imread(filename);
}

cv::Mat ImageHandler::loadCached(const std::string &filename) {
  auto it = cache_.find(filename);
  if (it != cache_.end()) {
    return it->second;
  }
  cv::Mat image = loader_->imread(filename);
  if (!image.empty()) {
    cache_[filename] = image;
  }
  return image;
}

ImageHandler::ImageHandler(std::unique_ptr<IImageLoader> loader)
    : loader_(std::move(loader)) {}

std::pair<int, int> ImageHandler::getDimensions(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty()) {
    std::cerr << "Could not open or find the image: " << filename << std::endl;
    return {-1, -1};
  }
  return {image.cols, image.rows};
}

bool ImageHandler::isGrayscale(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return false;
  return image.channels() == 1;
}

double ImageHandler::getAverageBrightness(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Scalar mean = cv::mean(image);
  if (image.channels() == 1)
    return mean[0];
  return (mean[0] + mean[1] + mean[2]) / 3.0;
}

int ImageHandler::getNumberOfChannels(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1;
  return image.channels();
}

int ImageHandler::getEdgeCount(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1;
  cv::Mat gray, edges;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Canny(gray, edges, 100, 200);
  return cv::countNonZero(edges);
}

std::vector<std::array<double, 3>>
ImageHandler::getDominantColors(const std::string &filename, int k) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return {};
  cv::Mat data;
  image.convertTo(data, CV_32F);
  data = data.reshape(1, data.total());

  std::vector<int> labels;
  cv::Mat centers;
  cv::kmeans(data, k, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              10, 1.0),
             3, cv::KMEANS_PP_CENTERS, centers);

  std::vector<std::array<double, 3>> dominantColors;
  for (int i = 0; i < k; ++i) {
    dominantColors.push_back({centers.at<float>(i, 0), centers.at<float>(i, 1),
                              centers.at<float>(i, 2)});
  }
  return dominantColors;
}

double ImageHandler::getBlurScore(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray, laplacian;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0] * stddev[0]; // Variance
}

double ImageHandler::getContrastRatio(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  double minVal, maxVal;
  cv::minMaxLoc(gray, &minVal, &maxVal);
  return maxVal > 0 ? maxVal / (minVal + 1e-6) : 0.0; // Avoid division by zero
}

double ImageHandler::getSaturationLevel(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty() || image.channels() < 3)
    return -1.0;
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  cv::Scalar mean = cv::mean(hsv);
  return mean[1]; // Saturation channel
}

std::vector<std::vector<int>>
ImageHandler::getHistogram(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return {};
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  std::vector<std::vector<int>> histograms;
  for (const auto &channel : channels) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < channel.rows; ++i) {
      for (int j = 0; j < channel.cols; ++j) {
        hist[channel.at<uchar>(i, j)]++;
      }
    }
    histograms.push_back(hist);
  }
  return histograms;
}

double ImageHandler::getAspectRatio(const std::string &filename) {
  auto [width, height] = getDimensions(filename);
  return height > 0 ? static_cast<double>(width) / height : 0.0;
}

double ImageHandler::getImageEntropy(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat hist;
  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};
  cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
  hist /= gray.total(); // Normalize
  double entropy = 0.0;
  for (int i = 0; i < histSize; ++i) {
    float p = hist.at<float>(i);
    if (p > 0) {
      entropy -= p * log2(p);
    }
  }
  return entropy;
}

double ImageHandler::getNoiseEstimate(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  int m = gray.rows, n = gray.cols;
  if (m <= 2 || n <= 2)
    return 0.0;

  cv::Mat grayF;
  gray.convertTo(grayF, CV_64F);
  // Immerkær (1996) kernel for noise estimation
  cv::Mat kernel = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
  cv::Mat filtered;
  cv::filter2D(grayF, filtered, CV_64F, kernel);
  double sumAbs = cv::norm(filtered, cv::NORM_L1);
  return std::sqrt(CV_PI / 2.0) / (6.0 * (m - 2) * (n - 2)) * sumAbs;
}

double ImageHandler::getSymmetryScore(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};

  // Horizontal symmetry: left half vs right half (flipped)
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

  // Vertical symmetry: top half vs bottom half (flipped)
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

TextureFeatures ImageHandler::getTextureFeatures(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return {-1.0, -1.0, -1.0, -1.0};
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  // Downsample for performance on large images
  cv::Mat workImg = gray;
  if (gray.rows > 256 || gray.cols > 256) {
    cv::resize(gray, workImg, cv::Size(256, 256));
  }

  // Quantize to 16 levels
  const int levels = 16;
  cv::Mat quantized;
  workImg.convertTo(quantized, CV_32S);
  quantized = quantized * levels / 256;

  // Build symmetric GLCM (horizontal neighbor, distance = 1)
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

uint64_t ImageHandler::getPerceptualHash(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return 0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  // dHash: resize to 9×8, compare adjacent pixels in each row
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

double ImageHandler::getWhiteBalanceScore(const std::string &filename) {
  cv::Mat image = loadCached(filename);
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

std::vector<int> ImageHandler::getHueHistogram(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty() || image.channels() < 3)
    return {};
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);
  const cv::Mat &hue = channels[0]; // Hue in [0, 180) in OpenCV

  // 36 bins of 5 degrees each
  std::vector<int> hist(36, 0);
  for (int r = 0; r < hue.rows; ++r) {
    for (int c = 0; c < hue.cols; ++c) {
      int h = hue.at<uchar>(r, c);
      hist[std::min(h / 5, 35)]++;
    }
  }
  return hist;
}

double ImageHandler::getSharpnessScore(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return -1.0;
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat gradX, gradY, magnitude;
  cv::Sobel(gray, gradX, CV_64F, 1, 0);
  cv::Sobel(gray, gradY, CV_64F, 0, 1);
  cv::magnitude(gradX, gradY, magnitude);
  return cv::mean(magnitude)[0];
}

double ImageHandler::compareImages(const std::string &filename1,
                                   const std::string &filename2) {
  cv::Mat img1 = loadCached(filename1);
  cv::Mat img2 = loadCached(filename2);
  if (img1.empty() || img2.empty())
    return -1.0;

  cv::Mat gray1, gray2;
  if (img1.channels() > 1)
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  else
    gray1 = img1;
  if (img2.channels() > 1)
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
  else
    gray2 = img2;

  // Resize second image to match first if sizes differ
  if (gray1.size() != gray2.size()) {
    cv::resize(gray2, gray2, gray1.size());
  }

  gray1.convertTo(gray1, CV_64F);
  gray2.convertTo(gray2, CV_64F);

  // SSIM constants (standard values for 8-bit images)
  const double c1 = 6.5025;  // (0.01 * 255)^2
  const double c2 = 58.5225; // (0.03 * 255)^2

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
  cv::multiply(2 * mu1_mu2 + c1, 2 * sigma12 + c2, numerator);
  cv::multiply(mu1_sq + mu2_sq + c1, sigma1_sq + sigma2_sq + c2, denominator);

  cv::Mat ssim_map;
  cv::divide(numerator, denominator, ssim_map);
  return cv::mean(ssim_map)[0];
}

std::string ImageHandler::getNoiseType(const std::string &filename) {
  cv::Mat image = loadCached(filename);
  if (image.empty())
    return "";
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  // Compute residual relative to median-filtered version
  cv::Mat medianFiltered;
  cv::medianBlur(gray, medianFiltered, 3);
  cv::Mat residual;
  cv::absdiff(gray, medianFiltered, residual);
  residual.convertTo(residual, CV_64F);

  // Pixels whose absolute residual from the median filter exceeds this
  // threshold are considered impulsive (salt-and-pepper) outliers.
  // A value of 50 is intentionally broad: it avoids false positives on
  // high-frequency but non-impulsive textures, while still catching the
  // extreme (0 or 255) outliers that characterise salt-and-pepper noise.
  constexpr double kSaltPepperResidualThreshold = 50.0;
  // If more than 1% of pixels are impulsive outliers we classify the dominant
  // noise as salt-and-pepper rather than Gaussian.
  constexpr double kSaltPepperRatioThreshold = 0.01;

  cv::Mat extremeMask;
  cv::threshold(residual, extremeMask, kSaltPepperResidualThreshold, 1.0,
                cv::THRESH_BINARY);
  double extremeRatio = cv::sum(extremeMask)[0] / residual.total();

  if (extremeRatio > kSaltPepperRatioThreshold) {
    return "salt_and_pepper";
  }
  return "gaussian";
}

// Global thread-safe model cache for ONNX networks to avoid repeated file I/O
static std::mutex g_model_mutex;
static std::unordered_map<std::string, cv::dnn::Net> g_model_cache;

std::pair<double, double>
ImageHandler::assessQualityDNN(const std::string &filename,
                               const std::string &model_path) {
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    return {-1.0, -1.0};
  }

  cv::Mat image = loadCached(filename);
  if (image.empty()) {
    return {-1.0, -1.0};
  }

  try {
    cv::dnn::Net net;
    {
      std::lock_guard<std::mutex> lock(g_model_mutex);
      auto it = g_model_cache.find(model_path);
      if (it != g_model_cache.end()) {
        net = it->second;
      } else {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
          return {-1.0, -1.0};
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        g_model_cache[model_path] = net;
      }
    }

    cv::Mat inputImg = image;
    if (inputImg.channels() == 1) {
      cv::cvtColor(inputImg, inputImg, cv::COLOR_GRAY2BGR);
    } else if (inputImg.channels() == 4) {
      cv::cvtColor(inputImg, inputImg, cv::COLOR_BGRA2BGR);
    }

    // Preprocessing: standard 224x224 RGB input with ImageNet mean subtraction
    // & scaling
    cv::Mat blob =
        cv::dnn::blobFromImage(inputImg, 1.0 / 255.0, cv::Size(224, 224),
                               cv::Scalar(0.485, 0.456, 0.406), true, false);

    net.setInput(blob);
    cv::Mat prob = net.forward();

    double aesthetic_score = -1.0;
    double technical_score = -1.0;

    if (prob.total() == 10) {
      // 10-bin NIMA distribution representing scores 1..10
      cv::Mat expProb;
      cv::exp(prob, expProb);
      cv::Scalar sumExp = cv::sum(expProb);
      if (sumExp[0] > 0) {
        expProb /= sumExp[0];
      }
      double mean_score = 0.0;
      for (int i = 0; i < 10; ++i) {
        mean_score += (i + 1) * expProb.at<float>(0, i);
      }
      aesthetic_score = mean_score; // 1.0 - 10.0 scale
      technical_score = std::clamp((mean_score - 1.0) / 9.0, 0.0, 1.0);
    } else if (prob.total() == 1) {
      // Direct scalar output
      aesthetic_score = static_cast<double>(prob.at<float>(0, 0));
      technical_score = std::clamp((aesthetic_score - 1.0) / 9.0, 0.0, 1.0);
    } else {
      double sum = 0.0;
      for (size_t i = 0; i < prob.total(); ++i) {
        sum += prob.at<float>(0, i);
      }
      aesthetic_score = sum / prob.total();
      technical_score = std::clamp(aesthetic_score, 0.0, 1.0);
    }

    return {aesthetic_score, technical_score};
  } catch (const std::exception &e) {
    std::cerr << "OpenCV DNN inference error: " << e.what() << std::endl;
    return {-1.0, -1.0};
  } catch (...) {
    return {-1.0, -1.0};
  }
}

ImageMetrics ImageHandler::getMetrics(const std::string &filename,
                                      const std::string &model_path) {
  ImageMetrics m{};
  auto [w, h] = getDimensions(filename);
  m.width = w;
  m.height = h;
  if (w == -1)
    return m; // File failed to load

  m.is_grayscale = isGrayscale(filename);
  m.average_brightness = getAverageBrightness(filename);
  m.channels = getNumberOfChannels(filename);
  m.edge_count = getEdgeCount(filename);
  m.dominant_colors = getDominantColors(filename);
  m.blur_score = getBlurScore(filename);
  m.contrast_ratio = getContrastRatio(filename);
  m.saturation_level = getSaturationLevel(filename);
  m.histogram = getHistogram(filename);
  m.aspect_ratio = getAspectRatio(filename);
  m.entropy = getImageEntropy(filename);
  m.noise_estimate = getNoiseEstimate(filename);
  m.symmetry_score = getSymmetryScore(filename);
  m.texture = getTextureFeatures(filename);
  m.perceptual_hash = getPerceptualHash(filename);
  m.white_balance_score = getWhiteBalanceScore(filename);
  m.hue_histogram = getHueHistogram(filename);
  m.sharpness_score = getSharpnessScore(filename);
  m.noise_type = getNoiseType(filename);

  if (!model_path.empty()) {
    auto [aesthetic, technical] = assessQualityDNN(filename, model_path);
    m.aesthetic_score = aesthetic;
    m.technical_quality_score = technical;
    m.ml_evaluated = (aesthetic >= 0.0);
  } else {
    m.aesthetic_score = -1.0;
    m.technical_quality_score = -1.0;
    m.ml_evaluated = false;
  }

  return m;
}

namespace vidicant {

std::pair<int, int> getImageDimensions(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDimensions(filename);
}

bool isImageGrayscale(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.isGrayscale(filename);
}

double getImageAverageBrightness(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAverageBrightness(filename);
}

int getImageNumberOfChannels(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNumberOfChannels(filename);
}

int getImageEdgeCount(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getEdgeCount(filename);
}

std::vector<std::array<double, 3>>
getImageDominantColors(const std::string &filename, int k) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getDominantColors(filename, k);
}

double getImageBlurScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getBlurScore(filename);
}

double getImageContrastRatio(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getContrastRatio(filename);
}

double getImageSaturationLevel(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSaturationLevel(filename);
}

std::vector<std::vector<int>> getImageHistogram(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHistogram(filename);
}

double getImageAspectRatio(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getAspectRatio(filename);
}

double getImageEntropy(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getImageEntropy(filename);
}

double getImageNoiseEstimate(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseEstimate(filename);
}

double getImageSymmetryScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSymmetryScore(filename);
}

TextureFeatures getImageTextureFeatures(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getTextureFeatures(filename);
}

uint64_t getImagePerceptualHash(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getPerceptualHash(filename);
}

double getImageWhiteBalanceScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getWhiteBalanceScore(filename);
}

std::vector<int> getImageHueHistogram(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getHueHistogram(filename);
}

double getImageSharpnessScore(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getSharpnessScore(filename);
}

double compareImages(const std::string &filename1,
                     const std::string &filename2) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.compareImages(filename1, filename2);
}

std::string getImageNoiseType(const std::string &filename) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getNoiseType(filename);
}

std::pair<double, double> assessImageQualityDNN(const std::string &filename,
                                                const std::string &model_path) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.assessQualityDNN(filename, model_path);
}

ImageMetrics getImageMetrics(const std::string &filename,
                             const std::string &model_path) {
  auto loader = std::make_unique<OpenCVImageLoader>();
  ImageHandler handler(std::move(loader));
  return handler.getMetrics(filename, model_path);
}

std::vector<ImageMetrics>
getBatchImageMetrics(const std::vector<std::string> &filenames,
                     const std::string &model_path) {
  // Bound concurrency to avoid spawning an unbounded number of threads.
  const size_t maxConcurrency =
      std::max(1u, std::thread::hardware_concurrency());

  std::vector<ImageMetrics> results(filenames.size());

  for (size_t start = 0; start < filenames.size(); start += maxConcurrency) {
    const size_t end = std::min(start + maxConcurrency, filenames.size());

    std::vector<std::future<ImageMetrics>> futures;
    futures.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      const std::string &fn = filenames[i];
      futures.push_back(std::async(std::launch::async, [fn, model_path]() {
        auto loader = std::make_unique<OpenCVImageLoader>();
        ImageHandler handler(std::move(loader));
        return handler.getMetrics(fn, model_path);
      }));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      results[start + i] = futures[i].get();
    }
  }
  return results;
}

} // namespace vidicant
