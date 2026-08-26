// File: video.hpp
// Header file for video processing functionalities in the Vidicant library.
//
// This file defines interfaces and classes for loading and analyzing videos.
// It provides an abstraction layer over video loading mechanisms and offers
// various video analysis functions such as frame extraction, motion detection,
// color analysis, and metadata retrieval.

#ifndef VIDICANT_VIDEO_HPP
#define VIDICANT_VIDEO_HPP

#include "vidicant/types.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace vidicant {

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
  virtual bool open(const std::filesystem::path &filename) = 0;

  // Gets the total number of frames in the video.
  virtual int getFrameCount() = 0;

  // Gets the frames per second (FPS) of the video.
  virtual double getFPS() = 0;

  // Gets the resolution (width, height) of the video.
  virtual std::pair<int, int> getResolution() = 0;

  // Reads the next frame from the video.
  virtual cv::Mat readFrame() = 0;

  // Seeks to a specific frame index.
  virtual bool seekFrame(int /*frameIndex*/) { return false; }

  // Grabs the next frame without decoding (for efficient skipping).
  virtual bool grabFrame() { return false; }

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
  bool open(const std::filesystem::path &filename) override;

  // Gets the frame count using OpenCV.
  int getFrameCount() override;

  // Gets the FPS using OpenCV.
  double getFPS() override;

  // Gets the resolution using OpenCV.
  std::pair<int, int> getResolution() override;

  // Reads the next frame using OpenCV.
  cv::Mat readFrame() override;

  // Seeks to a specific frame index.
  bool seekFrame(int frameIndex) override;

  // Grabs the next frame without decoding.
  bool grabFrame() override;

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
      loader_; // Pointer to the video loader implementation.
  std::filesystem::path filename_; // Stored filename for reopening if needed.

public:
  // Constructs a VideoHandler with the specified loader.
  // @param loader A unique pointer to an IVideoLoader implementation.
  explicit VideoHandler(std::unique_ptr<IVideoLoader> loader);

  // Opens a video file for analysis.
  // @param filename The path to the video file.
  // @return True if the video was opened successfully, false otherwise.
  bool open(const std::filesystem::path &filename);

  // Gets the total frame count of the video. Returns nullopt on failure.
  // @return The number of frames.
  std::optional<int> getFrameCount();

  // Gets the frames per second of the video. Returns nullopt on failure.
  // @return The FPS value.
  std::optional<double> getFPS();

  // Gets the resolution of the video. Returns nullopt on failure.
  // @return A pair containing width and height.
  std::optional<std::pair<int, int>> getResolution();

  // Calculates the duration of the video in seconds. Returns nullopt on
  // failure.
  // @return The duration in seconds.
  std::optional<double> getDuration();

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
  bool saveFirstFrameAsImage(const std::filesystem::path &imagePath);

  // Calculates a motion score based on frame differences.
  // @param stride Frame stride for sampling (1 = every frame).
  // @return A motion score (higher values indicate more motion).
  double getMotionScore(int stride = 1);

  // Extracts dominant colors from the video frames.
  // @return A vector of arrays representing dominant colors in RGB format.
  std::vector<std::array<double, 3>> getDominantColors();

  // Detects scene changes in the video.
  // @param threshold Pixel difference threshold for detecting transitions.
  // @param stride Frame stride for sampling (1 = every frame).
  // @return A vector of frame indices where scene changes occur.
  std::vector<int> detectSceneChanges(double threshold = 30.0, int stride = 1);

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
  // @param stride Frame stride for sampling (1 = every frame).
  // @return ShotLengthStats with mean, stddev, min, max, and count.
  ShotLengthStats getShotLengthStats(double threshold = 30.0, int stride = 1);

  // Measures flicker intensity as the standard deviation of frame-to-frame
  // brightness changes.
  // @return Flicker score (higher = more flickering).
  double getFlickerScore();

  // Returns the index of the highest-quality frame among sampled frames.
  // Quality is scored by sharpness (Laplacian variance) × brightness penalty.
  // @return 0-based frame index in the video stream.
  int getBestThumbnailIndex();

  // Extracts and saves the sharpest thumbnail for each detected scene cut to
  // outputDir.
  std::vector<SceneThumbnail>
  exportSceneThumbnails(const std::vector<int> &sceneChanges,
                        const std::filesystem::path &outputDir);

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
  double compareVideos(const std::filesystem::path &otherFilename);

  // Returns a VideoMetrics struct populated with all analyses. Returns nullopt
  // on failure.
  std::optional<VideoMetrics> getMetrics();

  // Returns a VideoMetrics struct populated with all analyses using options.
  // Returns nullopt on failure.
  std::optional<VideoMetrics> getMetrics(const VideoAnalysisOptions &options);
};

// Convenience functions for video analysis.
//
// These standalone functions mirror the VideoHandler methods, allowing for
// direct use without manually constructing handler objects.

// Convenience function to get video frame count. Returns nullopt on failure.
std::optional<int> getVideoFrameCount(const std::filesystem::path &filename);

// Convenience function to get video FPS. Returns nullopt on failure.
std::optional<double> getVideoFPS(const std::filesystem::path &filename);

// Convenience function to get video resolution. Returns nullopt on failure.
std::optional<std::pair<int, int>>
getVideoResolution(const std::filesystem::path &filename);

// Convenience function to get video duration. Returns nullopt on failure.
std::optional<double> getVideoDuration(const std::filesystem::path &filename);

// Convenience function to extract first frame.
cv::Mat extractFirstFrame(const std::filesystem::path &filename);

// Convenience function to get average brightness.
double getVideoAverageBrightness(const std::filesystem::path &filename);

// Convenience function to check if video is grayscale.
bool isVideoGrayscale(const std::filesystem::path &filename);

// Convenience function to save first frame as image.
bool saveFirstFrameAsImage(const std::filesystem::path &videoPath,
                           const std::filesystem::path &imagePath);

// Convenience function to get motion score with stride.
double getVideoMotionScore(const std::filesystem::path &filename,
                           int stride = 1);

// Convenience function to get dominant colors.
std::vector<std::array<double, 3>>
getVideoDominantColors(const std::filesystem::path &filename);

// Convenience function to detect scene changes.
std::vector<int> detectVideoSceneChanges(const std::filesystem::path &filename,
                                         double threshold = 30.0,
                                         int stride = 1);

// Convenience function to get frame rate stability.
double getVideoFrameRateStability(const std::filesystem::path &filename);

// Convenience function to get color consistency.
double getVideoColorConsistency(const std::filesystem::path &filename);

// Convenience function to get optical flow magnitude.
double getVideoOpticalFlowMagnitude(const std::filesystem::path &filename);

// Convenience function to check for audio track.
bool videoHasAudioTrack(const std::filesystem::path &filename);

// Convenience function to get shot length statistics.
ShotLengthStats getVideoShotLengthStats(const std::filesystem::path &filename,
                                        double threshold = 30.0,
                                        int stride = 1);

// Convenience function to get flicker score.
double getVideoFlickerScore(const std::filesystem::path &filename);

// Convenience function to get best thumbnail frame index.
int getVideoBestThumbnailIndex(const std::filesystem::path &filename);

// Convenience function to export scene thumbnails to an output directory.
std::vector<SceneThumbnail>
exportVideoSceneThumbnails(const std::filesystem::path &filename,
                           const std::vector<int> &sceneChanges,
                           const std::filesystem::path &outputDir);

// Convenience function to get temporal brightness curve.
std::vector<double>
getVideoTemporalBrightnessCurve(const std::filesystem::path &filename);

// Convenience function to get codec FOURCC string.
std::string getVideoCodecFourcc(const std::filesystem::path &filename);

// Convenience function to compare two videos.
double compareVideos(const std::filesystem::path &filename1,
                     const std::filesystem::path &filename2);

// Convenience function to get all video metrics at once. Returns nullopt on
// failure.
std::optional<VideoMetrics>
getVideoMetrics(const std::filesystem::path &filename);

// Convenience function to get all video metrics using options. Returns nullopt
// on failure.
std::optional<VideoMetrics>
getVideoMetrics(const std::filesystem::path &filename,
                const VideoAnalysisOptions &options);

} // namespace vidicant

#endif // VIDICANT_VIDEO_HPP
