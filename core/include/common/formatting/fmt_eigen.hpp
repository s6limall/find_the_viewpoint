// File: common/formatting/fmt_eigen.hpp

#ifndef FMT_EIGEN_HPP
#define FMT_EIGEN_HPP

#include <Eigen/Core>
#include <fmt/format.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

// Helper type trait to detect if a type is a vector
template<typename T>
struct is_eigen_vector : std::false_type {};

template<typename Scalar, int Rows>
struct is_eigen_vector<Eigen::Matrix<Scalar, Rows, 1>> : std::true_type {};

template<typename Scalar, int Cols>
struct is_eigen_vector<Eigen::Matrix<Scalar, 1, Cols>> : std::true_type {};

// Helper type trait to detect if a type is a transposed vector
template<typename T>
struct is_eigen_transposed_vector : std::false_type {};

template<typename Scalar, int Rows>
struct is_eigen_transposed_vector<Eigen::Transpose<Eigen::Matrix<Scalar, Rows, 1>>> : std::true_type {};

template<typename Scalar, int Cols>
struct is_eigen_transposed_vector<Eigen::Transpose<Eigen::Matrix<Scalar, 1, Cols>>> : std::true_type {};

// Base formatter for Eigen types
template<typename Derived>
struct EigenFormatterBase {
    constexpr auto parse(fmt::format_parse_context &ctx) -> decltype(ctx.begin()) { return ctx.end(); }

    template<typename FormatContext>
    auto format(const Eigen::MatrixBase<Derived> &mat, FormatContext &ctx) const {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<typename Derived::Scalar>::max_digits10);

        if constexpr (is_eigen_vector<Derived>::value || is_eigen_transposed_vector<Derived>::value) {
            // Vector format: [x, y, z, ...]
            oss << "[";
            for (int i = 0; i < mat.size(); ++i) {
                oss << mat(i);
                if (i < mat.size() - 1)
                    oss << ", ";
            }
            oss << "]";
        } else {
            // Matrix format: multi-line
            for (int row = 0; row < mat.rows(); ++row) {
                for (int col = 0; col < mat.cols(); ++col) {
                    oss << mat(row, col);
                    if (col < mat.cols() - 1)
                        oss << ", ";
                }
                if (row < mat.rows() - 1)
                    oss << "\n";
            }
        }

        return fmt::format_to(ctx.out(), "{}", oss.str());
    }
};

// Specialize the fmt::formatter for Eigen::Matrix
template<typename Scalar, int Rows, int Cols>
struct fmt::formatter<Eigen::Matrix<Scalar, Rows, Cols>> : EigenFormatterBase<Eigen::Matrix<Scalar, Rows, Cols>> {};

// Specialize the fmt::formatter for Eigen::Transpose<Eigen::Matrix>
template<typename Scalar, int Rows, int Cols>
struct fmt::formatter<Eigen::Transpose<Eigen::Matrix<Scalar, Rows, Cols>>>
    : EigenFormatterBase<Eigen::Transpose<Eigen::Matrix<Scalar, Rows, Cols>>> {};

#endif // FMT_EIGEN_HPP
