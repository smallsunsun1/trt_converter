#ifndef INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_
#define INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_

#include <unordered_map>
#include <string>

namespace sss {

inline constexpr int kDefaultMaxBatch = 1;
inline constexpr int kDefaultWorkspace = 16;
inline constexpr int kDefaultMinTiming = 1;
inline constexpr int kDefaultAvgTiming = 8;

// System default params
inline constexpr int kDefaultDevice = 0;

// Inference default params
inline constexpr int kDefaultBatch = 1;
inline constexpr int kDefaultStreams = 1;
inline constexpr int kDefaultIterations = 10;
inline constexpr int kDefaultWarmUp = 200;
inline constexpr int kDefaultDuration = 3;
inline constexpr int kDefaultSleep = 0;

// Reporting default params
inline constexpr int kDefaultAvgRuns = 10;
inline constexpr float kDefaultPercentile = 99;

using Arguments = std::unordered_map<std::string, std::string>;

struct Options {
    virtual void Parse(Arguments& arguments) = 0;
};



}

#endif /* INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_ */
