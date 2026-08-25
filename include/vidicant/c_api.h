// File: c_api.h
// C-ABI header for the Vidicant media analysis library.
//
// This file declares the stable C interface for libvidicant, suitable for
// consumption by C/C++ clients or foreign function interfaces (Python ctypes,
// Rust, Go, Swift, etc.).
//
// All functions returning `const char*` return a heap-allocated JSON string
// (or NULL on failure) that MUST be freed by calling `vidicant_free_string`.

#ifndef VIDICANT_C_API_H
#define VIDICANT_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(VIDICANT_BUILD_SHARED)
#define VIDICANT_API __declspec(dllexport)
#elif defined(VIDICANT_USE_SHARED)
#define VIDICANT_API __declspec(dllimport)
#else
#define VIDICANT_API
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define VIDICANT_API __attribute__((visibility("default")))
#else
#define VIDICANT_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Checks whether the specified file is a valid image file.
VIDICANT_API bool vidicant_is_image_file(const char *filename);

/// Checks whether the specified file is a valid video file.
VIDICANT_API bool vidicant_is_video_file(const char *filename);

/// Analyzes an image file with standard heuristic metrics and returns a JSON
/// string. Caller must free the returned string with vidicant_free_string().
VIDICANT_API const char *vidicant_process_image(const char *filename);

/// Analyzes an image file with aesthetic/technical ML quality assessment.
/// Caller must free the returned string with vidicant_free_string().
VIDICANT_API const char *vidicant_process_image_ml(const char *filename,
                                                   const char *model_path);

/// Analyzes an image file with a specified neural task ("quality", "classify",
/// "detect", "embed"). Caller must free the returned string with
/// vidicant_free_string().
VIDICANT_API const char *vidicant_process_image_dnn(const char *filename,
                                                    const char *model_path,
                                                    const char *task, int top_k,
                                                    float conf_threshold,
                                                    float nms_threshold);

/// Analyzes an in-memory encoded image buffer (JPEG, PNG, etc.) with neural
/// inference. Caller must free the returned string with vidicant_free_string().
VIDICANT_API const char *vidicant_process_image_bytes(
    const uint8_t *buffer, size_t len, const char *model_path, const char *task,
    int top_k, float conf_threshold, float nms_threshold);

/// Analyzes a video file and returns a JSON string with video metrics and shot
/// statistics. Caller must free the returned string with
/// vidicant_free_string().
VIDICANT_API const char *vidicant_process_video(const char *filename);

/// Clusters duplicate/near-duplicate images in a directory via perceptual
/// hashing. Caller must free the returned string with vidicant_free_string().
VIDICANT_API const char *
vidicant_dedupe_directory(const char *dir, int threshold, bool recursive);

/// Frees a string allocated by any `vidicant_process_*` or `vidicant_dedupe_*`
/// function.
VIDICANT_API void vidicant_free_string(const char *str);

#ifdef __cplusplus
}
#endif

#endif // VIDICANT_C_API_H
