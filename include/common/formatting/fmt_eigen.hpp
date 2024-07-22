// File: common/formatting/fmt_eigen.hpp

#ifndef FMT_EIGEN_HPP
#define FMT_EIGEN_HPP

#include <Eigen/Core>
#include <fmt/format.h>
#include <iomanip> // For setting precision
#include <sstream>
#include <string>

/*
 * Specialized formatter for Eigen matrices using fmt library.
 * Supports 'f' (fixed-point), 'e' (scientific), 'g' (general), and 'a' (hexadecimal) notation.
 * Default: 'f' with 4 decimal places.
 * Example: fmt::print("Matrix:\n{:.2e}\n", mat); // Scientific with 2 decimal places
 */

// Specialize the fmt::formatter for Eigen::Matrix
template<typename Scalar, int Rows, int Cols>
struct fmt::formatter<Eigen::Matrix<Scalar, Rows, Cols>> {
    // Default format specifiers
    char presentation = 'f'; // 'f' for fixed precision (floating point), 'e' for scientific notation, etc
    int precision = -1; // Default precision to full precision

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
    auto format(const Eigen::Matrix<Scalar, Rows, Cols> &mat, FormatContext &ctx) const {
        std::ostringstream oss;

        // Set precision and format
        if (precision >= 0) {
            oss << std::setprecision(precision);
        } else {
            oss << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1);
        }
        if (presentation == 'f') {
            oss << std::fixed;
        } else if (presentation == 'e') {
            oss << std::scientific;
        }

        // Iterate over the matrix elements and format each element
        for (int row = 0; row < mat.rows(); ++row) {
            for (int col = 0; col < mat.cols(); ++col) {
                oss << mat(row, col);
                if (col < mat.cols() - 1) {
                    oss << ", "; // Add a comma between elements
                }
            }
            if (row < mat.rows() - 1) {
                oss << "\n"; // Add a newline between rows
            }
        }

        // Return the formatted string
        return fmt::format_to(ctx.out(), "\n{}", oss.str());
    }
};

// Specialize the fmt::formatter for Eigen::Transpose<Eigen::Matrix>
template<typename Scalar, int Rows, int Cols>
struct fmt::formatter<Eigen::Transpose<Eigen::Matrix<Scalar, Rows, Cols>>> {
    // Default format specifiers
    char presentation = 'f'; // 'f' for fixed precision (floating point), 'e' for scientific notation, etc
    int precision = -1; // Default precision to full precision

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

    // Format the transpose matrix
    template<typename FormatContext>
    auto format(const Eigen::Transpose<Eigen::Matrix<Scalar, Rows, Cols>> &mat, FormatContext &ctx) const {
        std::ostringstream oss;

        // Set precision and format
        if (precision >= 0) {
            oss << std::setprecision(precision);
        } else {
            oss << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1);
        }
        if (presentation == 'f') {
            oss << std::fixed;
        } else if (presentation == 'e') {
            oss << std::scientific;
        }

        // Iterate over the matrix elements and format each element
        for (int col = 0; col < mat.cols(); ++col) {
            for (int row = 0; row < mat.rows(); ++row) {
                oss << mat(row, col);
                if (row < mat.rows() - 1) {
                    oss << ", "; // Add a comma between elements
                }
            }
            if (col < mat.cols() - 1) {
                oss << "\n"; // Add a newline between columns
            }
        }

        // Return the formatted string
        return fmt::format_to(ctx.out(), "\n{}", oss.str());
    }
};

#endif // FMT_EIGEN_HPP
