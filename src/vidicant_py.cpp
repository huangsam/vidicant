// vidicant_py.cpp
// Python bindings for Vidicant using pybind11
//
// This file exposes the C++ media processing functionality
// to Python, allowing the library to be used as a pip package.

#include "controller.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper function to convert nlohmann::json to Python object
py::object json_to_python(const nlohmann::json &j) {
  if (j.is_null()) {
    return py::none();
  } else if (j.is_boolean()) {
    return py::bool_(j.get<bool>());
  } else if (j.is_number_integer()) {
    return py::int_(j.get<int64_t>());
  } else if (j.is_number_unsigned()) {
    return py::int_(j.get<uint64_t>());
  } else if (j.is_number_float()) {
    return py::float_(j.get<double>());
  } else if (j.is_string()) {
    return py::str(j.get<std::string>());
  } else if (j.is_array()) {
    py::list list;
    for (const auto &elem : j) {
      list.append(json_to_python(elem));
    }
    return list;
  } else if (j.is_object()) {
    py::dict dict;
    for (auto it = j.begin(); it != j.end(); ++it) {
      dict[py::str(it.key())] = json_to_python(it.value());
    }
    return dict;
  }
  return py::none();
}

// Wrapper for processImage that returns Python dict
py::object process_image_wrapper(const std::string &filename) {
  nlohmann::json result = processImage(filename);
  return json_to_python(result);
}

// Wrapper for processVideo that returns Python dict
py::object process_video_wrapper(const std::string &filename) {
  nlohmann::json result = processVideo(filename);
  return json_to_python(result);
}

PYBIND11_MODULE(vidicant_py, m) {
  m.doc() = "Vidicant Python bindings for cross-platform media analysis";

  // Bind utility functions
  m.def("is_image_file", &isImageFile,
        "Check if a file is a supported image format", py::arg("filename"));

  m.def("is_video_file", &isVideoFile,
        "Check if a file is a supported video format", py::arg("filename"));

  // Bind main processing functions with wrappers that convert JSON to Python
  m.def("process_image", &process_image_wrapper,
        "Process an image file and return analysis results as a dictionary",
        py::arg("filename"));

  m.def("process_video", &process_video_wrapper,
        "Process a video file and return analysis results as a dictionary",
        py::arg("filename"));
}
