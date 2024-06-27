// File: include/types/image_comparison_result.hpp

#ifndef IMAGE_COMPARISON_RESULT_HPP
#define IMAGE_COMPARISON_RESULT_HPP

#include <string>
#include <map>

namespace types {
    struct ImageComparisonResult {
        std::string method;
        double score{};
        std::map<std::string, double> additional_metrics;

        ImageComparisonResult() = default;

        ImageComparisonResult(std::string method, double score, std::map<std::string, double> additional_metrics)
            : method(std::move(method)), score(score), additional_metrics(std::move(additional_metrics)) {
        }
    };
}

#endif // IMAGE_COMPARISON_RESULT_HPP
