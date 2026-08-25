// File: vidicant.hpp
// Top-level umbrella / facade header for the Vidicant library.
//
// Including this single header provides access to the complete Vidicant C++
// API, including handlers, options structs, operations, deduplication, and file
// detection.

#ifndef VIDICANT_VIDICANT_HPP
#define VIDICANT_VIDICANT_HPP

#include "vidicant/core/dedupe.hpp"
#include "vidicant/core/image_ops.hpp"
#include "vidicant/core/video_ops.hpp"
#include "vidicant/dnn/dnn_engine.hpp"
#include "vidicant/image.hpp"
#include "vidicant/io/file_detector.hpp"
#include "vidicant/pipeline.hpp"
#include "vidicant/types.hpp"
#include "vidicant/video.hpp"

#endif // VIDICANT_VIDICANT_HPP
