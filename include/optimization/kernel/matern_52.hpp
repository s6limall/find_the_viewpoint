// File: optimization/kernel/matern_52.hpp

#ifndef KERNEL_MATERN_52_HPP
#define KERNEL_MATERN_52_HPP

#include "common/logging/logger.hpp"
#include "optimization/kernel/kernel.hpp"

namespace optimization::kernel {

    template<typename T = double>
    class Matern52Kernel final : public Kernel<T> {
    public:
        Matern52Kernel(T length_scale, T variance) : length_scale_(length_scale), variance_(variance) {
            if (length_scale_ <= 0 || variance_ <= 0) {
                LOG_ERROR("Length scale = {}, Variance = {} must be positive!", length_scale_, variance_);
                throw std::invalid_argument("Length scale and variance must be positive.");
            }
        }

        T compute(const Eigen::Matrix<T, Eigen::Dynamic, 1> &first_point,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1> &second_point) const noexcept override {
            LOG_TRACE("First point size: {}, second point size: {}", first_point.size(), second_point.size());
            const T sqrt_five = std::sqrt(static_cast<T>(5.0));
            const T distance = (first_point - second_point).norm();
            const T scaled_distance = sqrt_five * distance / length_scale_;

            LOG_TRACE("Distance = {}, Scaled distance: {}", distance, scaled_distance);
            return variance_ * (1 + scaled_distance + (scaled_distance * scaled_distance) / static_cast<T>(3.0)) *
                   std::exp(-scaled_distance);
        }

    private:
        const T length_scale_;
        const T variance_;
    };
} // namespace optimization::kernel


#endif // KERNEL_MATERN_52_HPP
