// File: optimization/kernel/kernel.hpp

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <Eigen/Dense>

namespace optimization::kernel {
    template<typename T = double>
    class Kernel {
    public:
        virtual ~Kernel() = default;
        virtual T compute(const Eigen::Matrix<T, Eigen::Dynamic, 1> &first_point,
                          const Eigen::Matrix<T, Eigen::Dynamic, 1> &second_point) const noexcept = 0;
    };

} // namespace optimization::kernel

#endif // KERNEL_HPP
