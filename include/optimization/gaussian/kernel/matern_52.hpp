// File: optimization/kernel/matern_52.hpp

#ifndef KERNEL_MATERN_52_HPP
#define KERNEL_MATERN_52_HPP

#include <Eigen/Dense>
#include <cmath>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/kernel.hpp"

namespace optimization::kernel {

template<typename T = double>
class Matern52 final : public Kernel<T> {
public:
    explicit Matern52(T length_scale = 1.0, T variance = 1.0, T noise_variance = 1e-6)
        : length_scale_(length_scale), variance_(variance), noise_variance_(noise_variance) {
        validateParameters();
        LOG_INFO("Initialized Matern 5/2 kernel with length_scale={}, variance={}, noise_variance={}",
                 length_scale_, variance_, noise_variance_);
    }

    // Compute the kernel function k(x, y)
    // Formula: k(x, y) = s^2 * (1 + sqrt(5)r/l + 5r^2/(3l^2)) * exp(-sqrt(5)r/l)
    // where r = ||x - y||, l is length_scale, s^2 is variance
    T compute(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
              const Eigen::Matrix<T, Eigen::Dynamic, 1> &y) const noexcept override {
        const T r = (x - y).norm();
        const T z = sqrt_5_ * r / length_scale_;
        return variance_ * (1.0 + z + z * z / 3.0) * std::exp(-z);
    }

    // Compute the Gram matrix K where K_ij = k(x_i, y_j)
    // This method calculates the kernel function for every pair of points in X and Y
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    computeGramMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) const {
        const int n = X.rows();
        const int m = Y.rows();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(n, m);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                K(i, j) = compute(X.row(i).transpose(), Y.row(j).transpose());
            }
        }
        return K;
    }

    // Compute the Gram matrix K where K_ij = k(x_i, x_j) + noise_variance if i=j
    // This method is optimized for the case when X = Y, utilizing symmetry
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    computeGramMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) const {
        const int n = X.rows();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(n, n);

        #pragma omp parallel for
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

    // Compute gradient of k(x, y) with respect to hyperparameters [length_scale, variance, noise_variance]
    // This method is crucial for optimizing the kernel parameters
    Eigen::Matrix<T, Eigen::Dynamic, 1> computeGradient(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
                                                        const Eigen::Matrix<T, Eigen::Dynamic, 1> &y) const {
        Eigen::Matrix<T, Eigen::Dynamic, 1> grad(3);
        const T r = (x - y).norm();
        const T z = sqrt_5_ * r / length_scale_;
        const T exp_term = std::exp(-z);
        const T common_term = variance_ * exp_term;

        // Gradient with respect to length_scale
        // This term captures how the kernel changes as the characteristic length scale varies
        grad(0) = common_term * (5.0 / 3.0) * (z * z) * (1.0 + z) / length_scale_;

        // Gradient with respect to variance
        // This term represents how the kernel changes with the signal variance
        grad(1) = (1.0 + z + z * z / 3.0) * exp_term;

        // Gradient with respect to noise_variance
        // This is non-zero only when x equals y (i.e., on the diagonal of the Gram matrix)
        grad(2) = (x.isApprox(y)) ? 1.0 : 0.0;

        return grad;
    }

    // Compute the gradient of the Gram matrix with respect to a specific hyperparameter
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    computeGradientMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, int param_index) const {
        const int n = X.rows();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K_grad(n, n);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                Eigen::Matrix<T, Eigen::Dynamic, 1> x_i = X.row(i).transpose();
                Eigen::Matrix<T, Eigen::Dynamic, 1> x_j = X.row(j).transpose();

                const T r = (x_i - x_j).norm();
                const T z = sqrt_5_ * r / length_scale_;
                const T exp_term = std::exp(-z);

                switch (param_index) {
                    case 0: // Gradient w.r.t. length_scale
                        K_grad(i, j) = variance_ * exp_term * (5.0 / 3.0) * (z * z) * (1.0 + z) /
                                       (length_scale_ * length_scale_);
                        break;
                    case 1: // Gradient w.r.t. variance
                        K_grad(i, j) = exp_term * (1.0 + z + z * z / 3.0);
                        break;
                    case 2: // Gradient w.r.t. noise_variance (only on diagonal)
                        K_grad(i, j) = (i == j) ? 1.0 : 0.0;
                        break;
                    default:
                        K_grad(i, j) = 0.0;
                        LOG_WARN("Invalid parameter index in computeGradientMatrix: {}", param_index);
                }

                if (i != j) {
                    K_grad(j, i) = K_grad(i, j);
                }
            }
        }

        return K_grad;
    }

    // Update the kernel parameters
    void setParameters(T length_scale, T variance, T noise_variance) {
        T eps = std::numeric_limits<T>::epsilon();
        length_scale_ = std::max(length_scale, eps);
        variance_ = std::max(variance, eps);
        noise_variance_ = std::max(noise_variance, T(0));

        if (length_scale != length_scale_ || variance != variance_ || noise_variance != noise_variance_) {
            LOG_WARN("Kernel parameters adjusted to ensure validity: length_scale={}, variance={}, noise_variance={}",
                     length_scale_, variance_, noise_variance_);
        } else {
            LOG_DEBUG("Updated kernel parameters: length_scale={}, variance={}, noise_variance={}",
                      length_scale_, variance_, noise_variance_);
        }
    }

    // Retrieve the current kernel parameters
    Eigen::Matrix<T, 3, 1> getParameters() const {
        return Eigen::Matrix<T, 3, 1>(length_scale_, variance_, noise_variance_);
    }

private:
    T length_scale_; // Characteristic length scale of the kernel
    T variance_; // Signal variance
    T noise_variance_; // Noise variance (jitter term for numerical stability)
    static constexpr T sqrt_5_ = 2.236067977499790; // Pre-computed sqrt(5)

    // Ensure that all parameters are valid (positive, except noise_variance which can be zero)
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