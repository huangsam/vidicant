// File: formatters.hpp
// CLI output serialization and formatting helpers.

#ifndef VIDICANT_CLI_FORMATTERS_HPP
#define VIDICANT_CLI_FORMATTERS_HPP

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace vidicant::cli {

inline std::string escapeCsv(const std::string &str) {
  if (str.find(',') != std::string::npos ||
      str.find('"') != std::string::npos ||
      str.find('\n') != std::string::npos) {
    std::string res = "\"";
    for (char c : str) {
      if (c == '"')
        res += "\"\"";
      else
        res += c;
    }
    res += "\"";
    return res;
  }
  return str;
}

inline void writeJsonlRecord(std::ostream &os, const std::string &type,
                             const nlohmann::json &res) {
  nlohmann::json line = res;
  line["media_type"] = type;
  os << line.dump() << std::endl;
}

inline void writeCsvHeader(std::ostream &os) {
  os << "filename,media_type,width,height,channels,is_grayscale,average_"
        "brightness,blur_score,contrast_ratio,saturation_level,aspect_ratio,"
        "entropy,noise_estimate,perceptual_hash,ml_evaluated,aesthetic_score,"
        "technical_quality_score,duration_seconds,frame_count,fps,motion_"
        "score\n";
}

inline void writeCsvRecord(std::ostream &os, const std::string &type,
                           const nlohmann::json &res) {
  std::string fn = res.value("filename", "");
  os << escapeCsv(fn) << "," << type << ",";

  if (type == "image") {
    os << res.value("width", 0) << "," << res.value("height", 0) << ","
       << res.value("channels", 0) << ","
       << (res.value("is_grayscale", false) ? "true" : "false") << ","
       << res.value("average_brightness", 0.0) << ","
       << res.value("blur_score", 0.0) << ","
       << res.value("contrast_ratio", 0.0) << ","
       << res.value("saturation_level", 0.0) << ","
       << res.value("aspect_ratio", 0.0) << "," << res.value("entropy", 0.0)
       << "," << res.value("noise_estimate", 0.0) << ","
       << res.value("perceptual_hash", 0ULL) << ","
       << (res.value("ml_evaluated", false) ? "true" : "false") << ",";

    if (res.contains("aesthetic_score") && !res["aesthetic_score"].is_null()) {
      os << res["aesthetic_score"].get<double>() << ",";
    } else {
      os << ",";
    }

    if (res.contains("technical_quality_score") &&
        !res["technical_quality_score"].is_null()) {
      os << res["technical_quality_score"].get<double>() << ",";
    } else {
      os << ",";
    }

    os << ",,,"; // Video fields empty
  } else {
    // Video
    os << res.value("width", 0) << "," << res.value("height", 0) << ",,"
       << (res.value("is_grayscale", false) ? "true" : "false") << ","
       << res.value("average_brightness", 0.0) << ",,,,,,,,false,,,";
    os << res.value("duration_seconds", 0.0) << ","
       << res.value("frame_count", 0) << "," << res.value("fps", 0.0) << ","
       << res.value("motion_score", 0.0);
  }
  os << "\n";
}

} // namespace vidicant::cli

#endif // VIDICANT_CLI_FORMATTERS_HPP
