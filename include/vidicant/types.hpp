// File: types.hpp
// Common data structures and metrics types for the Vidicant library.

#ifndef VIDICANT_TYPES_HPP
#define VIDICANT_TYPES_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vidicant {

// Struct: TextureFeatures
// Gray-level co-occurrence matrix (GLCM) derived texture metrics.
struct TextureFeatures {
  double contrast{0.0}; // Measures local intensity variations.
  double energy{0.0};   // Sum of squared elements (texture uniformity).
  double homogeneity{
      0.0}; // Closeness of the distribution to the GLCM diagonal.
  double correlation{0.0}; // Linear dependency of gray-level pairs.
};

// Struct: BoundingBox
// 2D bounding box coordinates [x, y, width, height].
struct BoundingBox {
  float x{0.0f};
  float y{0.0f};
  float width{0.0f};
  float height{0.0f};
};

// Struct: DetectedObject
// An object or face detected by a neural network model.
struct DetectedObject {
  BoundingBox box;
  std::string class_name;
  float confidence{0.0f};
};

// Struct: ClassificationLabel
// A classification label and its predicted probability.
struct ClassificationLabel {
  std::string label;
  float confidence{0.0f};
  int class_id{-1};
};

// Struct: ImageMetrics
// Aggregates all per-image analysis results into a single object.
struct ImageMetrics {
  int width{-1};
  int height{-1};
  bool is_grayscale{false};
  double average_brightness{0.0};
  int channels{0};
  int edge_count{0};
  std::vector<std::array<double, 3>> dominant_colors;
  double blur_score{0.0};
  double contrast_ratio{0.0};
  double saturation_level{0.0};
  std::vector<std::vector<int>> histogram;
  double aspect_ratio{0.0};
  double entropy{0.0};
  double noise_estimate{0.0};
  double symmetry_score{0.0};
  TextureFeatures texture;
  uint64_t perceptual_hash{0};
  double white_balance_score{0.0};
  std::vector<int> hue_histogram;
  double sharpness_score{0.0};
  std::string noise_type;
  double aesthetic_score{-1.0}; // Aesthetic score [1.0 - 10.0] via ONNX/DNN
                                // (-1.0 if not evaluated).
  double technical_quality_score{
      -1.0}; // Technical quality score [0.0 - 1.0] (-1.0 if not evaluated).
  bool ml_evaluated{false}; // True if DNN model inference was executed.
  std::vector<DetectedObject> detected_objects;
  std::vector<ClassificationLabel> top_labels;
  std::vector<float> embedding;
};

// Struct: ShotLengthStats
// Statistics over shot (scene segment) durations in a video, in frames.
struct ShotLengthStats {
  double mean{0.0};   // Average shot length in frames.
  double stddev{0.0}; // Standard deviation of shot lengths.
  double min{0.0};    // Shortest shot in frames.
  double max{0.0};    // Longest shot in frames.
  int count{0};       // Total number of shots.
};

// Struct: VideoMetrics
// Aggregates all per-video analysis results into a single object.
struct VideoMetrics {
  int frame_count{0};
  double fps{0.0};
  int width{0};
  int height{0};
  double duration{0.0};
  bool is_grayscale{false};
  double average_brightness{0.0};
  double motion_score{0.0};
  std::vector<std::array<double, 3>> dominant_colors;
  double frame_rate_stability{0.0};
  double color_consistency{0.0};
  double optical_flow_magnitude{0.0};
  bool has_audio_track{false};
  ShotLengthStats shot_length_stats;
  double flicker_score{0.0};
  int best_thumbnail_frame{0};
  std::vector<double> temporal_brightness_curve;
  std::string codec_fourcc;
};

} // namespace vidicant

#endif // VIDICANT_TYPES_HPP
