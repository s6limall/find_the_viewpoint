// // File: common/formatting/fmt_camera_parameters.hpp
//
// #ifndef FMT_CAMERA_PARAMETERS_HPP
// #define FMT_CAMERA_PARAMETERS_HPP
//
// #include <fmt/format.h>
// #include "core/camera.hpp"
// #include <sstream>
//
// // Custom formatter for core::Camera::CameraParameters
// template<>
// struct formatter<core::Camera::CameraParameters> {
//     // Parse format specifications (not needed for this example, so we leave it empty)
//     constexpr auto parse(fmt::format_parse_context &ctx) {
//         return ctx.begin();
//     }
//
//     // Format the CameraParameters struct
//     template<typename FormatContext>
//     auto format(const core::Camera::CameraParameters &params, FormatContext &ctx) {
//         std::ostringstream oss;
//
//         // Format width, height, and FOV
//         oss << "Camera Parameters:\n";
//         oss << fmt::format("  Image Width: {} pixels\n", params.width);
//         oss << fmt::format("  Image Height: {} pixels\n", params.height);
//         oss << fmt::format("  Horizontal Field of View (FOV X): {:.4f} radians\n", params.fov_x);
//         oss << fmt::format("  Vertical Field of View (FOV Y): {:.4f} radians\n", params.fov_y);
//
//         // Format intrinsics matrix with detailed description
//         oss << "  Intrinsic Matrix:\n";
//         auto format_row = [&](int row) {
//             oss << "    [ ";
//             for (int col = 0; col < params.intrinsics.cols(); ++col) {
//                 oss << fmt::format("{: .4f}", params.intrinsics(row, col));
//                 if (col < params.intrinsics.cols() - 1)
//                     oss << ", ";
//             }
//             oss << " ]\n";
//         };
//         for (int row = 0; row < params.intrinsics.rows(); ++row) {
//             format_row(row);
//         }
//
//         // Format additional intrinsic parameters
//         oss << fmt::format("  Focal Length X: {:.4f}\n", params.getFocalLengthX());
//         oss << fmt::format("  Focal Length Y: {:.4f}\n", params.getFocalLengthY());
//         oss << fmt::format("  Principal Point X: {:.4f}\n", params.getPrincipalPointX());
//         oss << fmt::format("  Principal Point Y: {:.4f}\n", params.getPrincipalPointY());
//
//         // Return the formatted string
//         return fmt::format_to(ctx.out(), "{}", oss.str());
//     }
// };
//
// #endif //FMT_CAMERA_PARAMETERS_HPP
