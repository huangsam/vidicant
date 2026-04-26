// File: video.hpp
// Header file for video processing functionalities in the Vidicant library.
//
// This file defines interfaces and classes for loading and analyzing videos.
// It provides an abstraction layer over video loading mechanisms and offers
// various video analysis functions such as frame extraction, motion detection,
// color analysis, and metadata retrieval.

#ifndef VIDICANT_VIDEO_HPP
#define VIDICANT_VIDEO_HPP

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

// Struct: ShotLengthStats
// Statistics over shot (scene segment) durations in a video, in frames.
struct ShotLengthStats {
  double mean;   // Average shot length in frames.
  double stddev; // Standard deviation of shot lengths.
  double min;    // Shortest shot in frames.
  double max;    // Longest shot in frames.
  int count;     // Total number of shots.
};

// Struct: VideoMetrics
// Aggregates all per-video analysis results into a single object.
struct VideoMetrics {
  int frame_count;
  double fps;
  int width;
  int height;
  double duration;
  bool is_grayscale;
  double average_brightness;
  double motion_score;
  std::vector<std::array<double, 3>> dominant_colors;
  double frame_rate_stability;
  double color_consistency;
  double optical_flow_magnitude;
  bool has_audio_track;
  ShotLengthStats shot_length_stats;
  double flicker_score;
  int best_thumbnail_frame;
  std::vector<double> temporal_brightness_curve;
  std::string codec_fourcc;
};

// Class: IVideoLoader
// Abstract interface for video loading and frame reading operations.
//
// This interface defines the contract for opening videos, retrieving metadata,
// and reading frames. Implementations can use different libraries or methods
// for video processing.
class IVideoLoader {
public:
  // Virtual destructor for proper cleanup of derived classes.
  virtual ~IVideoLoader() = default;

  // Opens a video file for reading.
  virtual bool open(const std::string &filename) = 0;

  // Gets the total number of frames in the video.
  virtual int getFrameCount() = 0;

  // Gets the frames per second (FPS) of the video.
  virtual double getFPS() = 0;

  // Gets the resolution (width, height) of the video.
  virtual std::pair<int, int> getResolution() = 0;

  // Reads the next frame from the video.
  virtual cv::Mat readFrame() = 0;

  // Retrieves a raw VideoCapture property by ID (default: -1.0 if unsupported).
  virtual double getProperty(int /*propId*/) { return -1.0; }
};

// Class: OpenCVVideoLoader
// Concrete implementation of IVideoLoader using OpenCV.
//
// This class uses OpenCV's VideoCapture to handle video operations.
class OpenCVVideoLoader : public IVideoLoader {
public:
  // Opens a video file using OpenCV's VideoCapture.
  // @param filename The path to the video file.
  // @return True if the video was opened successfully, false otherwise.
  bool open(const std::string &filename) override;

  // Gets the frame count using OpenCV.
  int getFrameCount() override;

  // Gets the FPS using OpenCV.
  double getFPS() override;

  // Gets the resolution using OpenCV.
  std::pair<int, int> getResolution() override;

  // Reads the next frame using OpenCV.
  cv::Mat readFrame() override;

  // Retrieves a VideoCapture property by ID.
  double getProperty(int propId) override;

private:
  cv::VideoCapture cap_; // OpenCV VideoCapture object for video operations.
};

// Class: VideoHandler
// High-level handler for video analysis operations.
//
// This class provides various methods to analyze videos, including
// metadata retrieval, frame extraction, motion analysis, and color analysis.
// It uses a loader object to handle video operations internally.
class VideoHandler {
private:
  std::unique_ptr<IVideoLoader>
      loader_;           // Pointer to the video loader implementation.
  std::string filename_; // Stored filename for reopening if needed.

public:
  // Constructs a VideoHandler with the specified loader.
  // @param loader A unique pointer to an IVideoLoader implementation.
  explicit VideoHandler(std::unique_ptr<IVideoLoader> loader);

  // Opens a video file for analysis.
  // @param filename The path to the video file.
  // @return True if the video was opened successfully, false otherwise.
  bool open(const std::string &filename);

  // Gets the total frame count of the video.
  // @return The number of frames.
  int getFrameCount();

  // Gets the frames per second of the video.
  // @return The FPS value.
  double getFPS();

  // Gets the resolution of the video.
  // @return A pair containing width and height.
  std::pair<int, int> getResolution();

  // Calculates the duration of the video in seconds.
  // @return The duration in seconds.
  double getDuration();

  // Extracts the first frame of the video.
  // @return A cv::Mat object containing the first frame.
  cv::Mat extractFirstFrame();

  // Calculates the average brightness across all frames.
  // @return The average brightness value (0-255).
  double getAverageBrightness();

  // Checks if the video is grayscale.
  // @return True if the video is grayscale, false otherwise.
  bool isGrayscale();

  // Saves the first frame as an image file.
  // @param imagePath The path where to save the image.
  // @return True if saved successfully, false otherwise.
  bool saveFirstFrameAsImage(const std::string &imagePath);

  // Calculates a motion score based on frame differences.
  // @return A motion score (higher values indicate more motion).
  double getMotionScore();

  // Extracts dominant colors from the video frames.
  // @return A vector of arrays representing dominant colors in RGB format.
  std::vector<std::array<double, 3>> getDominantColors();

  // Detects scene changes in the video.
  // @return A vector of frame indices where scene changes occur.
  std::vector<int> detectSceneChanges(double threshold = 30.0);

  // Calculates frame rate stability (coefficient of variation).
  // @return Frame rate stability score (lower is more stable).
  double getFrameRateStability();

  // Calculates color consistency across frames.
  // @return Color consistency score (higher means more consistent).
  double getColorConsistency();

  // Calculates optical flow magnitude using Farneback dense optical flow.
  // @return Mean flow magnitude over sampled frame pairs (higher = more motion
  // detail).
  double getOpticalFlowMagnitude();

  // Checks whether the video contains an audio track.
  // @return True if an audio track is detected.
  bool hasAudioTrack();

  // Returns statistics on shot (scene segment) lengths in frames.
  // @param threshold Mean pixel difference threshold for scene change
  // detection.
  // @return ShotLengthStats with mean, stddev, min, max, and count.
  ShotLengthStats getShotLengthStats(double threshold = 30.0);

  // Measures flicker intensity as the standard deviation of frame-to-frame
  // brightness changes.
  // @return Flicker score (higher = more flickering).
  double getFlickerScore();

  // Returns the index of the highest-quality frame among sampled frames.
  // Quality is scored by sharpness (Laplacian variance) × brightness penalty.
  // @return 0-based frame index in the video stream.
  int getBestThumbnailIndex();

  // Returns per-frame brightness values as a time series (up to 100 frames).
  // @return Vector of brightness values.
  std::vector<double> getTemporalBrightnessCurve();

  // Returns the codec FOURCC string (e.g. "mp4v", "avc1").
  // @return 4-character codec identifier, or empty string on failure.
  std::string getCodecFourcc();

  // Computes a similarity score between this video and another via
  // Bhattacharyya histogram distance on sampled frames.
  // @param otherFilename Path to the second video.
  // @return Similarity in [0, 1] (1 = most similar), or -1 on error.
  double compareVideos(const std::string &otherFilename);

  // Returns a VideoMetrics struct populated with all analyses.
  VideoMetrics getMetrics();
};

// Namespace: vidicant
// Namespace containing convenience functions for video analysis.
//
// This namespace provides standalone functions that mirror the VideoHandler
// methods, allowing for easier use without creating handler objects.
namespace vidicant {

// Convenience function to get video frame count.
int getVideoFrameCount(const std::string &filename);

// Convenience function to get video FPS.
double getVideoFPS(const std::string &filename);

// Convenience function to get video resolution.
std::pair<int, int> getVideoResolution(const std::string &filename);

// Convenience function to get video duration.
double getVideoDuration(const std::string &filename);

// Convenience function to extract first frame.
cv::Mat extractFirstFrame(const std::string &filename);

// Convenience function to get average brightness.
double getVideoAverageBrightness(const std::string &filename);

// Convenience function to check if video is grayscale.
bool isVideoGrayscale(const std::string &filename);

// Convenience function to save first frame as image.
bool saveFirstFrameAsImage(const std::string &videoPath,
                           const std::string &imagePath);

// Convenience function to get motion score.
double getVideoMotionScore(const std::string &filename);

// Convenience function to get dominant colors.
std::vector<std::array<double, 3>>
getVideoDominantColors(const std::string &filename);

// Convenience function to detect scene changes.
std::vector<int> detectVideoSceneChanges(const std::string &filename,
                                         double threshold = 30.0);

// Convenience function to get frame rate stability.
double getVideoFrameRateStability(const std::string &filename);

// Convenience function to get color consistency.
double getVideoColorConsistency(const std::string &filename);

// Convenience function to get optical flow magnitude.
double getVideoOpticalFlowMagnitude(const std::string &filename);

// Convenience function to check for audio track.
bool videoHasAudioTrack(const std::string &filename);

// Convenience function to get shot length statistics.
ShotLengthStats getVideoShotLengthStats(const std::string &filename,
                                        double threshold = 30.0);

// Convenience function to get flicker score.
double getVideoFlickerScore(const std::string &filename);

// Convenience function to get best thumbnail frame index.
int getVideoBestThumbnailIndex(const std::string &filename);

// Convenience function to get temporal brightness curve.
std::vector<double>
getVideoTemporalBrightnessCurve(const std::string &filename);

// Convenience function to get codec FOURCC string.
std::string getVideoCodecFourcc(const std::string &filename);

// Convenience function to compare two videos.
double compareVideos(const std::string &filename1,
                     const std::string &filename2);

// Convenience function to get all video metrics at once.
VideoMetrics getVideoMetrics(const std::string &filename);

} // namespace vidicant

#endif // VIDICANT_VIDEO_HPP
