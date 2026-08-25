// File: dnn_engine.cpp
// Implementation of the OpenCV DNN inference engine.

#include "vidicant/dnn/dnn_engine.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <vector>

namespace vidicant::dnn {

static std::mutex g_model_mutex;
static std::unordered_map<std::string, cv::dnn::Net> g_model_cache;

static void decodeClassification(const cv::Mat &prob, ImageMetrics &metrics,
                                 int top_k) {
  if (prob.empty() || prob.total() == 0)
    return;

  const float *data = prob.ptr<float>();
  size_t total = prob.total();

  float maxVal = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < total; ++i) {
    if (data[i] > maxVal)
      maxVal = data[i];
  }

  std::vector<float> probs(total);
  double sumExp = 0.0;
  for (size_t i = 0; i < total; ++i) {
    probs[i] = std::exp(data[i] - maxVal);
    sumExp += probs[i];
  }
  if (sumExp > 0.0) {
    for (size_t i = 0; i < total; ++i) {
      probs[i] = static_cast<float>(probs[i] / sumExp);
    }
  }

  std::vector<std::pair<float, int>> indexed;
  indexed.reserve(total);
  for (size_t i = 0; i < total; ++i) {
    indexed.emplace_back(probs[i], static_cast<int>(i));
  }
  std::sort(indexed.begin(), indexed.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  int k = std::min(top_k, static_cast<int>(total));
  metrics.top_labels.clear();
  metrics.top_labels.reserve(k);
  for (int i = 0; i < k; ++i) {
    ClassificationLabel lbl;
    lbl.class_id = indexed[i].second;
    lbl.confidence = indexed[i].first;
    lbl.label = "class_" + std::to_string(lbl.class_id);
    metrics.top_labels.push_back(lbl);
  }
}

static void decodeDetection(const cv::Mat &prob, int img_width, int img_height,
                            ImageMetrics &metrics, float conf_threshold,
                            float nms_threshold) {
  if (prob.empty() || prob.total() == 0)
    return;

  std::vector<cv::Rect2d> candidate_boxes;
  std::vector<float> candidate_scores;
  std::vector<int> candidate_classes;

  if (prob.dims == 4 && prob.size[3] == 7) {
    int numDetections = prob.size[2];
    const float *data = prob.ptr<float>();
    for (int i = 0; i < numDetections; ++i) {
      const float *row = data + i * 7;
      float conf = row[2];
      if (conf >= conf_threshold) {
        int class_id = static_cast<int>(row[1]);
        float x1 = row[3] * img_width;
        float y1 = row[4] * img_height;
        float x2 = row[5] * img_width;
        float y2 = row[6] * img_height;
        float w = std::max(0.0f, x2 - x1);
        float h = std::max(0.0f, y2 - y1);
        candidate_boxes.emplace_back(x1, y1, w, h);
        candidate_scores.push_back(conf);
        candidate_classes.push_back(class_id);
      }
    }
  } else {
    int rows = 0;
    int cols = 0;
    bool transposed = false;

    if (prob.dims == 4 && prob.size[2] == 1 && prob.size[3] == 1) {
      if (prob.total() % 6 == 0) {
        cols = 6;
        rows = static_cast<int>(prob.total() / 6);
      } else if (prob.total() % 15 == 0) {
        cols = 15;
        rows = static_cast<int>(prob.total() / 15);
      } else if (prob.total() >= 3) {
        cols = static_cast<int>(prob.total());
        rows = 1;
      }
    } else if (prob.dims == 3) {
      if (prob.size[1] < prob.size[2] && prob.size[1] <= 85) {
        rows = prob.size[2];
        cols = prob.size[1];
        transposed = true;
      } else {
        rows = prob.size[1];
        cols = prob.size[2];
      }
    } else if (prob.dims == 2) {
      rows = prob.size[0];
      cols = prob.size[1];
    }

    if (rows > 0 && cols >= 3) {
      for (int r = 0; r < rows; ++r) {
        std::vector<float> row(cols);
        if (transposed) {
          for (int c = 0; c < cols; ++c) {
            row[c] = prob.at<float>(0, c, r);
          }
        } else if (prob.dims == 3) {
          const float *ptr = prob.ptr<float>(0, r);
          std::copy(ptr, ptr + cols, row.begin());
        } else if (prob.dims == 4 && prob.size[2] == 1 && prob.size[3] == 1) {
          const float *ptr = prob.ptr<float>() + r * cols;
          std::copy(ptr, ptr + cols, row.begin());
        } else {
          const float *ptr = prob.ptr<float>(r);
          std::copy(ptr, ptr + cols, row.begin());
        }

        float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
        float conf = 0.0f;
        int class_id = 0;

        if (cols == 3) {
          x = 0.0f;
          y = 0.0f;
          w = row[0];
          h = row[1];
          conf = row[2];
          class_id = 0;
        } else if (cols == 6) {
          float x1 = row[0];
          float y1 = row[1];
          float x2 = row[2];
          float y2 = row[3];
          conf = row[4];
          class_id = static_cast<int>(row[5]);

          if (x2 > x1 && y2 > y1) {
            x = x1;
            y = y1;
            w = x2 - x1;
            h = y2 - y1;
          } else {
            x = x1;
            y = y1;
            w = x2;
            h = y2;
          }
        } else if (cols == 15) {
          x = row[0];
          y = row[1];
          w = row[2];
          h = row[3];
          conf = row[14];
          class_id = 0;
        } else if (cols >= 5) {
          float cx = row[0];
          float cy = row[1];
          w = row[2];
          h = row[3];
          x = cx - w / 2.0f;
          y = cy - h / 2.0f;

          if (cols == 5) {
            conf = row[4];
            class_id = 0;
          } else {
            float max_cls_score = -1.0f;
            int best_cls = 0;
            for (int c = 4; c < cols; ++c) {
              if (row[c] > max_cls_score) {
                max_cls_score = row[c];
                best_cls = c - 4;
              }
            }
            conf = max_cls_score;
            class_id = best_cls;
          }
        }

        if (conf >= conf_threshold) {
          if (w <= 1.0f && h <= 1.0f && (x + w) <= 1.01f && (y + h) <= 1.01f) {
            x *= img_width;
            y *= img_height;
            w *= img_width;
            h *= img_height;
          }
          candidate_boxes.emplace_back(x, y, w, h);
          candidate_scores.push_back(conf);
          candidate_classes.push_back(class_id);
        }
      }
    }
  }

  if (candidate_boxes.empty())
    return;

  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(candidate_boxes, candidate_scores, conf_threshold,
                    nms_threshold, nms_indices);

  metrics.detected_objects.clear();
  metrics.detected_objects.reserve(nms_indices.size());
  for (int idx : nms_indices) {
    DetectedObject obj;
    obj.box.x = static_cast<float>(candidate_boxes[idx].x);
    obj.box.y = static_cast<float>(candidate_boxes[idx].y);
    obj.box.width = static_cast<float>(candidate_boxes[idx].width);
    obj.box.height = static_cast<float>(candidate_boxes[idx].height);
    obj.confidence = candidate_scores[idx];
    obj.class_name =
        (candidate_classes[idx] == 0 && prob.size[prob.dims - 1] == 15)
            ? "face"
            : ("class_" + std::to_string(candidate_classes[idx]));
    metrics.detected_objects.push_back(obj);
  }
}

static void decodeEmbedding(const cv::Mat &prob, ImageMetrics &metrics) {
  if (prob.empty() || prob.total() == 0)
    return;
  const float *data = prob.ptr<float>();
  metrics.embedding.assign(data, data + prob.total());
}

static void decodeQuality(const cv::Mat &prob, ImageMetrics &metrics) {
  if (prob.empty() || prob.total() == 0)
    return;

  double aesthetic_score = -1.0;
  double technical_score = -1.0;

  if (prob.total() == 10) {
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
    aesthetic_score = mean_score;
    technical_score = std::clamp((mean_score - 1.0) / 9.0, 0.0, 1.0);
  } else if (prob.total() == 1) {
    aesthetic_score = static_cast<double>(prob.at<float>(0, 0));
    technical_score = std::clamp((aesthetic_score - 1.0) / 9.0, 0.0, 1.0);
  } else {
    double sum = 0.0;
    const float *data = prob.ptr<float>();
    for (size_t i = 0; i < prob.total(); ++i) {
      sum += data[i];
    }
    aesthetic_score = sum / prob.total();
    technical_score = std::clamp(aesthetic_score, 0.0, 1.0);
  }

  metrics.aesthetic_score = aesthetic_score;
  metrics.technical_quality_score = technical_score;
}

void DnnEngine::runInference(const cv::Mat &image,
                             const std::string &model_path,
                             ImageMetrics &metrics, const std::string &task,
                             int top_k, float conf_threshold,
                             float nms_threshold) {
  if (model_path.empty() || !std::filesystem::exists(model_path) ||
      image.empty()) {
    metrics.ml_evaluated = false;
    return;
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
          metrics.ml_evaluated = false;
          return;
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

    cv::Mat blob =
        cv::dnn::blobFromImage(inputImg, 1.0 / 255.0, cv::Size(224, 224),
                               cv::Scalar(0.485, 0.456, 0.406), true, false);

    net.setInput(blob);
    cv::Mat prob = net.forward();

    if (task == "classify") {
      decodeClassification(prob, metrics, top_k);
    } else if (task == "detect") {
      decodeDetection(prob, image.cols, image.rows, metrics, conf_threshold,
                      nms_threshold);
    } else if (task == "embed") {
      decodeEmbedding(prob, metrics);
    } else if (task == "quality") {
      decodeQuality(prob, metrics);
    } else {
      if (prob.dims == 4 && prob.size[3] == 7) {
        decodeDetection(prob, image.cols, image.rows, metrics, conf_threshold,
                        nms_threshold);
      } else if (prob.dims == 3 && (prob.size[1] >= 4 || prob.size[2] >= 4)) {
        decodeDetection(prob, image.cols, image.rows, metrics, conf_threshold,
                        nms_threshold);
      } else if (prob.total() == 10) {
        decodeQuality(prob, metrics);
      } else if (prob.total() > 10 && prob.total() <= 1000 && prob.dims <= 2) {
        decodeClassification(prob, metrics, top_k);
      } else {
        decodeEmbedding(prob, metrics);
      }
    }

    metrics.ml_evaluated = true;
  } catch (const std::exception &e) {
    std::cerr << "OpenCV DNN inference error: " << e.what() << std::endl;
    metrics.ml_evaluated = false;
  } catch (...) {
    metrics.ml_evaluated = false;
  }
}

std::pair<double, double>
DnnEngine::assessQuality(const cv::Mat &image, const std::string &model_path) {
  ImageMetrics m{};
  runInference(image, model_path, m, "quality");
  return {m.aesthetic_score, m.technical_quality_score};
}

} // namespace vidicant::dnn
