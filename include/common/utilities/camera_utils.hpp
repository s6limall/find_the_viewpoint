// File: common/utilities/camera_utils.hpp

#ifndef CAMERA_UTILS_HPP
#define CAMERA_UTILS_HPP

#include <cmath>
#include <type_traits>
#include "common/logging/logger.hpp"

namespace common::utilities {

    template<typename T>
    constexpr bool isRadians(T angle) {
        static_assert(std::is_arithmetic_v<T>, "Angle must be a numeric type.");
        const bool result = angle <= T(2.0 * M_PI);
        LOG_DEBUG("The angle {} is in {}.", angle, result ? "radians" : "degrees");
        return result;
    }

    template<typename T>
    constexpr T toRadians(T degrees) {
        static_assert(std::is_arithmetic_v<T>, "Degrees must be a numeric type.");
        T radians = degrees * T(M_PI) / T(180.0);
        LOG_DEBUG("Converted degrees = {}, to radians = {}", degrees, radians);
        return radians;
    }

    template<typename T>
    constexpr T toRadiansIfDegrees(T angle) {
        static_assert(std::is_arithmetic_v<T>, "Angle must be a numeric type.");
        T radians = utilities::isRadians(angle) ? angle : utilities::toRadians(angle);
        return radians;
    }

    template<typename SizeType, typename FovType>
    constexpr auto calculateFocalLength(SizeType size, FovType fov) {
        static_assert(std::is_arithmetic_v<SizeType> && std::is_arithmetic_v<FovType>,
                      "Size and FOV must be numeric types.");
        if (size <= SizeType(0) || fov <= FovType(0)) {
            LOG_ERROR("Invalid size or FOV: size = {}, fov = {}", size, fov);
            throw std::invalid_argument("Size and FOV must be positive.");
        }

        LOG_DEBUG("Received FOV angle {}, ensuring it is in radians...", fov);
        float fov_rad = toRadiansIfDegrees(fov);
        FovType focal_length = size / (FovType(2.0) * std::tan(fov_rad / FovType(2.0)));
        LOG_DEBUG("Calculated focal_length = {}, using size = {} and FOV = {} radians.", focal_length, size, fov_rad);
        return focal_length;
    }

}

#endif //CAMERA_UTILS_HPP
