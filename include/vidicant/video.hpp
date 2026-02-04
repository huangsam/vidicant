// File: video.hpp
// Header file for video processing functionalities in the Vidicant library.
//
// This file defines interfaces and classes for loading and analyzing videos.
// It provides an abstraction layer over video loading mechanisms and offers
// various video analysis functions such as frame extraction, motion detection,
// color analysis, and metadata retrieval.

#ifndef VIDICANT_VIDEO_HPP
#define VIDICANT_VIDEO_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

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

} // namespace vidicant

#endif // VIDICANT_VIDEO_HPP
