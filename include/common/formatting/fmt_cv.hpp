// File: common/formatting/fmt_cv.hpp

#ifndef FMT_OPENCV_HPP
#define FMT_OPENCV_HPP

#include <fmt/format.h>
#include <opencv2/core.hpp>
#include <sstream>
#include <iomanip>
#include <string>

/*
 * Specialized formatter for OpenCV matrices (cv::Mat) using fmt library.
 * Supports 'f' (fixed-point), 'e' (scientific), 'g' (general), and 'a' (hexadecimal) notation.
 * Default: 'f' with 4 decimal places.
 * Example: fmt::print("Matrix:\n{:.2e}\n", mat); // Scientific with 2 decimal places
 */

template<>
struct fmt::formatter<cv::Mat> {
    // Default format specifiers
    char presentation = 'f'; // 'f' for fixed precision (floating point), 'e' for scientific notation, etc
    int precision = 4; // Default precision

    // Parse format specifications
    constexpr auto parse(fmt::format_parse_context &ctx) {
        auto it = ctx.begin();
        auto end = ctx.end();

        // Parse precision
        if (it != end && *it >= '0' && *it <= '9') {
            int parsed_precision = 0;
            while (it != end && *it >= '0' && *it <= '9') {
                parsed_precision = parsed_precision * 10 + (*it - '0');
                ++it;
            }
            precision = parsed_precision;
        }

        // Parse presentation type
        if (it != end && *it != '}') {
            presentation = *it++;
        }

        // Validate presentation type
        if (presentation != 'f' && presentation != 'e' && presentation != 'g' && presentation != 'a') {
            throw fmt::format_error("Invalid format specifier");
        }

        return it;
    }

    // Format the matrix
    template<typename FormatContext>
    auto format(const cv::Mat &mat, FormatContext &ctx) const {
        std::ostringstream oss;

        // Set precision and format
        oss << std::setprecision(precision);
        if (presentation == 'f') {
            oss << std::fixed;
        } else if (presentation == 'e') {
            oss << std::scientific;
        }

        // Function to format a single element based on type
        auto format_element = [&](int row, int col, int ch) {
            switch (mat.type()) {
                case CV_8U:
                    oss << static_cast<int>(mat.at<uchar>(row, col * mat.channels() + ch));
                    break;
                case CV_8S:
                    oss << static_cast<int>(mat.at<schar>(row, col * mat.channels() + ch));
                    break;
                case CV_16U:
                    oss << mat.at<ushort>(row, col * mat.channels() + ch);
                    break;
                case CV_16S:
                    oss << mat.at<short>(row, col * mat.channels() + ch);
                    break;
                case CV_32S:
                    oss << mat.at<int>(row, col * mat.channels() + ch);
                    break;
                case CV_32F:
                    oss << mat.at<float>(row, col * mat.channels() + ch);
                    break;
                case CV_64F:
                    oss << mat.at<double>(row, col * mat.channels() + ch);
                    break;
                default:
                    throw fmt::format_error("Unsupported cv::Mat type");
            }
        };

        // Iterate over the matrix elements and format each element
        for (int row = 0; row < mat.rows; ++row) {
            for (int col = 0; col < mat.cols; ++col) {
                if (mat.channels() == 1) {
                    format_element(row, col, 0);
                } else {
                    oss << "(";
                    for (int ch = 0; ch < mat.channels(); ++ch) {
                        format_element(row, col, ch);
                        if (ch < mat.channels() - 1) {
                            oss << ", ";
                        }
                    }
                    oss << ")";
                }
                if (col < mat.cols - 1) {
                    oss << ", "; // Add a comma between elements
                }
            }
            if (row < mat.rows - 1) {
                oss << "\n"; // Add a newline between rows
            }
        }

        // Return the formatted string
        return fmt::format_to(ctx.out(), "\n{}", oss.str());
    }
};

#endif // FMT_OPENCV_HPP
