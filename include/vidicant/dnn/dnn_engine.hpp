// File: dnn_engine.hpp
// Neural network inference engine utilizing OpenCV DNN module.

#ifndef VIDICANT_DNN_DNN_ENGINE_HPP
#define VIDICANT_DNN_DNN_ENGINE_HPP

#include "vidicant/types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <utility>

namespace vidicant::dnn {

class DnnEngine {
public:
  static void runInference(const cv::Mat &image, const std::string &model_path,
                           ImageMetrics &metrics,
                           const std::string &task = "quality", int top_k = 5,
                           float conf_threshold = 0.5f,
                           float nms_threshold = 0.4f);

  static std::pair<double, double> assessQuality(const cv::Mat &image,
                                                 const std::string &model_path);
};

} // namespace vidicant::dnn

#endif // VIDICANT_DNN_DNN_ENGINE_HPP
