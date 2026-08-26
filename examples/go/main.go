package main

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../zig-out/lib -lvidicant -Wl,-rpath,../../zig-out/lib
#include "vidicant/c_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"
)

// ImageMetrics represents structured image analysis results from Vidicant.
type ImageMetrics struct {
	Filename       string      `json:"filename"`
	Width          int         `json:"width"`
	Height         int         `json:"height"`
	BlurScore      float64     `json:"blur_score"`
	Brightness     float64     `json:"average_brightness"`
	ContrastRatio  float64     `json:"contrast_ratio"`
	NoiseEstimate  float64     `json:"noise_estimate"`
	IsGrayscale    bool        `json:"is_grayscale"`
	DominantColors [][]float64 `json:"dominant_colors"`
	Error          string      `json:"error,omitempty"`
}

// VideoMetrics represents structured video analytics results from Vidicant.
type VideoMetrics struct {
	Filename         string  `json:"filename"`
	DurationSeconds  float64 `json:"duration_seconds"`
	Width            int     `json:"width"`
	Height           int     `json:"height"`
	FPS              float64 `json:"fps"`
	FrameCount       int     `json:"frame_count"`
	MotionScore      float64 `json:"motion_score"`
	OpticalFlowMag   float64 `json:"optical_flow_magnitude"`
	BestThumbnailIdx int     `json:"best_thumbnail_frame"`
	Error            string  `json:"error,omitempty"`
}

// ProcessImage parses an image on disk via libvidicant C-ABI.
func ProcessImage(filename string) (*ImageMetrics, error) {
	cPath := C.CString(filename)
	defer C.free(unsafe.Pointer(cPath))

	cJSON := C.vidicant_process_image(cPath)
	if cJSON == nil {
		return nil, fmt.Errorf("failed to process image: %s", filename)
	}
	defer C.vidicant_free_string(cJSON)

	var metrics ImageMetrics
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &metrics); err != nil {
		return nil, fmt.Errorf("json unmarshal failed: %w", err)
	}
	return &metrics, nil
}

// ProcessImageBytes parses an in-memory image byte buffer via libvidicant C-ABI.
func ProcessImageBytes(data []byte) (*ImageMetrics, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty buffer")
	}

	cBuf := (*C.uint8_t)(unsafe.Pointer(&data[0]))
	cLen := C.size_t(len(data))

	cJSON := C.vidicant_process_image_bytes(cBuf, cLen, nil, nil, 0, 0, 0)
	if cJSON == nil {
		return nil, fmt.Errorf("failed to process image bytes")
	}
	defer C.vidicant_free_string(cJSON)

	var metrics ImageMetrics
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &metrics); err != nil {
		return nil, fmt.Errorf("json unmarshal failed: %w", err)
	}
	return &metrics, nil
}

// ProcessVideo parses a video on disk via libvidicant C-ABI.
func ProcessVideo(filename string) (*VideoMetrics, error) {
	cPath := C.CString(filename)
	defer C.free(unsafe.Pointer(cPath))

	cJSON := C.vidicant_process_video(cPath)
	if cJSON == nil {
		return nil, fmt.Errorf("failed to process video: %s", filename)
	}
	defer C.vidicant_free_string(cJSON)

	var metrics VideoMetrics
	if err := json.Unmarshal([]byte(C.GoString(cJSON)), &metrics); err != nil {
		return nil, fmt.Errorf("json unmarshal failed: %w", err)
	}
	return &metrics, nil
}

// UploadPreflightHandler demonstrates a zero-disk HTTP upload validation endpoint.
func UploadPreflightHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil || len(body) == 0 {
		http.Error(w, "Invalid or empty image upload", http.StatusBadRequest)
		return
	}

	metrics, err := ProcessImageBytes(body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Analysis failed: %v", err), http.StatusInternalServerError)
		return
	}

	accepted := metrics.BlurScore >= 40.0 && metrics.Brightness >= 20.0 && metrics.Brightness <= 240.0
	status := "ACCEPTED"
	if !accepted {
		status = "REJECTED"
	}

	resp := map[string]interface{}{
		"status":     status,
		"accepted":   accepted,
		"dimensions": fmt.Sprintf("%dx%d", metrics.Width, metrics.Height),
		"blur_score": metrics.BlurScore,
		"brightness": metrics.Brightness,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func main() {
	sampleImg := filepath.Join("..", "sample.jpg")
	sampleVideo := filepath.Join("..", "sample.mp4")

	if _, err := os.Stat(sampleImg); os.IsNotExist(err) {
		sampleImg = "examples/sample.jpg"
		sampleVideo = "examples/sample.mp4"
	}

	fmt.Println("============================================================")
	fmt.Println("VIDICANT GO / CGO REFERENCE APPLICATION")
	fmt.Println("============================================================")

	// 1. In-Memory Byte Processing via Simulated HTTP Endpoint
	fmt.Println("\n[1] Testing In-Memory HTTP Upload Preflight Gate (net/http)...")
	imgBytes, err := os.ReadFile(sampleImg)
	if err != nil {
		fmt.Printf("Error reading %s: %v\n", sampleImg, err)
		return
	}

	req := httptest.NewRequest(http.MethodPost, "/api/upload/preflight", bytes.NewReader(imgBytes))
	rec := httptest.NewRecorder()
	UploadPreflightHandler(rec, req)

	fmt.Printf("HTTP Status Code: %d\n", rec.Code)
	fmt.Printf("Response Body:    %s", rec.Body.String())

	// 2. High-Concurrency Goroutine Worker Pool
	fmt.Println("\n[2] Testing High-Concurrency Goroutine Worker Pool...")
	const concurrency = 4
	const totalTasks = 8

	tasks := make(chan string, totalTasks)
	results := make(chan string, totalTasks)
	var wg sync.WaitGroup

	start := time.Now()
	for workerID := 1; workerID <= concurrency; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for task := range tasks {
				if filepath.Ext(task) == ".mp4" {
					vm, err := ProcessVideo(task)
					if err != nil {
						results <- fmt.Sprintf("Worker %d: video error: %v", id, err)
					} else {
						results <- fmt.Sprintf("Worker %d: [Video] %s (%.1fs, motion: %.2f)", id, filepath.Base(task), vm.DurationSeconds, vm.MotionScore)
					}
				} else {
					im, err := ProcessImage(task)
					if err != nil {
						results <- fmt.Sprintf("Worker %d: image error: %v", id, err)
					} else {
						results <- fmt.Sprintf("Worker %d: [Image] %s (%dx%d, blur: %.1f)", id, filepath.Base(task), im.Width, im.Height, im.BlurScore)
					}
				}
			}
		}(workerID)
	}

	// Dispatch tasks
	for i := 0; i < totalTasks; i++ {
		if i%2 == 0 {
			tasks <- sampleImg
		} else {
			tasks <- sampleVideo
		}
	}
	close(tasks)

	wg.Wait()
	close(results)
	duration := time.Since(start)

	for res := range results {
		fmt.Printf("  • %s\n", res)
	}
	fmt.Printf("\n✓ Successfully processed %d media tasks concurrently across %d goroutines in %v\n", totalTasks, concurrency, duration)
}
