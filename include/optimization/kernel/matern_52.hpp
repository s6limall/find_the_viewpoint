// File: optimization/kernel/matern_52.hpp

#ifndef KERNEL_MATERN_52_HPP
#define KERNEL_MATERN_52_HPP

#include "common/logging/logger.hpp"
#include "optimization/kernel/kernel.hpp"

namespace optimization::kernel {

    template<typename T = double>
    class Matern52Kernel final : public Kernel<T> {
    public:
        Matern52Kernel(T length_scale, T variance, T noise_variance = 1e-6) :
            length_scale_(length_scale), variance_(variance), noise_variance_(noise_variance) {
            validateParameters();
        }

        T compute(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1> &y) const noexcept override {
            const T r = (x - y).norm();
            const T sqrt_5 = std::sqrt(5.0);
            const T z = sqrt_5 * r / length_scale_;
            return variance_ * (1 + z + z * z / 3.0) * std::exp(-z);
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        computeGramMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) const {
            const int n = X.rows();
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(n, n);

            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    K(i, j) = compute(X.row(i).transpose(), X.row(j).transpose());
                    if (i == j) {
                        K(i, i) += noise_variance_;
                    } else {
                        K(j, i) = K(i, j);
                    }
                }
            }

            return K;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> computeGradient(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
                                                            const Eigen::Matrix<T, Eigen::Dynamic, 1> &y) const {
            Eigen::Matrix<T, Eigen::Dynamic, 1> grad(3);
            const T r = (x - y).norm();
            const T sqrt_5 = std::sqrt(5.0);
            const T z = sqrt_5 * r / length_scale_;
            const T exp_term = std::exp(-z);
            const T common_term = variance_ * exp_term;

            // Gradient w.r.t. length_scale
            grad(0) = common_term * (5.0 / 3.0) * (z * z) * (1 + z) / length_scale_;

            // Gradient w.r.t. variance
            grad(1) = (1 + z + z * z / 3.0) * exp_term;

            // Gradient w.r.t. noise variance
            grad(2) = (x.isApprox(y)) ? 2 * noise_variance_ : 0;

            return grad;
        }

        void setParameters(T length_scale, T variance, T noise_variance) {
            length_scale_ = length_scale;
            variance_ = variance;
            noise_variance_ = noise_variance;
            validateParameters();
        }

        Eigen::Matrix<T, 3, 1> getParameters() const {
            return Eigen::Matrix<T, 3, 1>(length_scale_, variance_, noise_variance_);
        }

    private:
        T length_scale_;
        T variance_;
        T noise_variance_;

        void validateParameters() const {
            if (length_scale_ <= 0 || variance_ <= 0 || noise_variance_ < 0) {
                LOG_ERROR("Invalid kernel parameters: length_scale = {}, variance = {}, noise_variance = {}",
                          length_scale_, variance_, noise_variance_);
                throw std::invalid_argument("Kernel parameters must be positive (noise variance can be zero).");
            }
        }
    };

} // namespace optimization::kernel


#endif // KERNEL_MATERN_52_HPP
