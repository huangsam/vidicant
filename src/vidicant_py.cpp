// vidicant_py.cpp
// Python bindings for Vidicant using pybind11
//
// This file exposes the C++ media processing functionality
// to Python, allowing the library to be used as a pip package.

#include "controller.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(vidicant_py, m) {
  m.doc() = "Vidicant Python bindings for cross-platform media analysis";

  // Bind utility functions
  m.def("is_image_file", &isImageFile,
        "Check if a file is a supported image format", py::arg("filename"));

  m.def("is_video_file", &isVideoFile,
        "Check if a file is a supported video format", py::arg("filename"));

  // Bind main processing functions
  m.def("process_image", &processImage,
        "Process an image file and return analysis results as JSON",
        py::arg("filename"));

  m.def("process_video", &processVideo,
        "Process a video file and return analysis results as JSON",
        py::arg("filename"));
}
