// File: optimization/gaussian/kernel/matern_52.hpp

#ifndef KERNEL_MATERN52_HPP
#define KERNEL_MATERN52_HPP

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/kernel.hpp"

namespace optimization::kernel {

    template<FloatingPoint T = double>
    class Matern52 final : public Kernel<T> {
    public:
        using typename Kernel<T>::VectorType;
        using typename Kernel<T>::MatrixType;

        explicit Matern52(T length_scale = 1.0, T variance = 1.0, T noise_variance = 1e-6) :
            length_scale_(length_scale), variance_(variance), noise_variance_(noise_variance) {
            validateAndAdjustParameters();
            LOG_INFO("Initialized Matern 5/2 kernel with length_scale={}, variance={}, noise_variance={}",
                     length_scale_, variance_, noise_variance_);
        }

        [[nodiscard]] std::shared_ptr<Kernel<T>> clone() const override { return std::make_shared<Matern52>(*this); }

        [[nodiscard]] int getParameterCount() const override { return 3; }

        [[nodiscard]] T compute(const VectorType &x, const VectorType &y) const noexcept override {
            const T r = (x - y).norm();
            const T z = sqrt_5_ * r / length_scale_;
            const T exp_term = std::exp(std::max(-z, log_min_));
            return variance_ * (1.0 + z + z * z / 3.0) * exp_term + std::numeric_limits<T>::epsilon();
        }

        [[nodiscard]] MatrixType computeGramMatrix(const MatrixType &X, const MatrixType &Y) const override {
            return computeGramMatrixAdaptive(X, Y, false);
        }

        [[nodiscard]] MatrixType computeGramMatrix(const MatrixType &X) const override {
            return computeGramMatrixAdaptive(X, X, true);
        }

        [[nodiscard]] VectorType computeGradient(const VectorType &x, const VectorType &y) const override {
            VectorType grad(3);
            const T r = (x - y).norm();
            const T z = sqrt_5_ * r / length_scale_;
            const T exp_term = std::exp(std::max(-z, log_min_));
            const T common_term = variance_ * exp_term;

            grad(0) = common_term * (5.0 / 3.0) * (z * z) * (1.0 + z) / length_scale_;
            grad(1) = (1.0 + z + z * z / 3.0) * exp_term;
            grad(2) = (x.isApprox(y, 1e-8)) ? 1.0 : 0.0;

            return grad;
        }

        void setParameters(const VectorType &params) override {
            Kernel<T>::validateParameters(params, getParameterNames());
            setParameters(params(0), params(1), params(2));
        }

        void setParameters(T length_scale, T variance, T noise_variance) {
            T eps = std::numeric_limits<T>::epsilon();
            T old_length_scale = length_scale_;
            T old_variance = variance_;
            T old_noise_variance = noise_variance_;

            length_scale_ = std::max(length_scale, eps);
            variance_ = std::max(variance, eps);
            noise_variance_ = std::max(noise_variance, T(0));

            // Adaptive regularization: gradually increase noise_variance if parameters change significantly
            if (std::abs(length_scale_ - old_length_scale) / old_length_scale > 0.1 ||
                std::abs(variance_ - old_variance) / old_variance > 0.1) {
                noise_variance_ = std::max(noise_variance_, old_noise_variance * 1.1);
            }

            if (length_scale != length_scale_ || variance != variance_ || noise_variance != noise_variance_) {
                LOG_WARN("Kernel parameters adjusted to ensure validity: length_scale={}, variance={}, "
                         "noise_variance={}",
                         length_scale_, variance_, noise_variance_);
            } else {
                LOG_DEBUG("Updated kernel parameters: length_scale={}, variance={}, noise_variance={}", length_scale_,
                          variance_, noise_variance_);
            }
        }


        [[nodiscard]] VectorType getParameters() const override {
            VectorType params(3);
            params << length_scale_, variance_, noise_variance_;
            return params;
        }

        [[nodiscard]] MatrixType computeGradientMatrix(const MatrixType &X, const int param_index) const override {
            return computeGradientMatrixAdaptive(X, param_index);
        }

        [[nodiscard]] std::vector<std::string> getParameterNames() const override {
            return {"length_scale", "variance", "noise_variance"};
        }

        [[nodiscard]] bool isStationary() const override { return true; }
        [[nodiscard]] bool isIsotropic() const override { return true; }

        [[nodiscard]] std::string getKernelType() const override { return "Matern52"; }

        [[nodiscard]] MatrixType computeBasisFunctions(const MatrixType &X) const override {
            const int num_basis = std::min(100, static_cast<int>(X.rows()));
            MatrixType Phi(X.rows(), num_basis);

            std::mt19937 gen(std::random_device{}());
            std::normal_distribution<T> dist(0.0, 1.0);

            for (int i = 0; i < num_basis; ++i) {
                VectorType omega = VectorType::NullaryExpr(X.cols(), [&]() { return dist(gen); });
                omega *= std::sqrt(5.0) / length_scale_;
                Phi.col(i) = std::sqrt(4.0 * M_PI * variance_ / std::pow(1.0 + omega.squaredNorm(), 3)) *
                             (X * omega).array().cos();
            }

            return Phi;
        }

        [[nodiscard]] VectorType computeEigenvalues(int num_eigenvalues) const override {
            VectorType eigenvalues(num_eigenvalues);
            T scale_factor = std::pow(length_scale_, -5);

            for (int i = 0; i < num_eigenvalues; ++i) {
                T lambda = static_cast<T>(i + 1);
                eigenvalues(i) = variance_ * scale_factor / std::pow(lambda, 5);
            }

            return eigenvalues;
        }

    private:
        T length_scale_;
        T variance_;
        T noise_variance_;
        static constexpr T sqrt_5_ = 2.236067977499790;
        static constexpr T log_min_ = std::log(std::numeric_limits<T>::min());

        void validateAndAdjustParameters() {
            T eps = std::numeric_limits<T>::epsilon();
            if (length_scale_ <= 0 || variance_ <= 0 || noise_variance_ < 0) {
                LOG_ERROR("Invalid kernel parameters: length_scale = {}, variance = {}, noise_variance = {}",
                          length_scale_, variance_, noise_variance_);
                length_scale_ = std::max(length_scale_, eps);
                variance_ = std::max(variance_, eps);
                noise_variance_ = std::max(noise_variance_, T(0));
                LOG_WARN("Parameters adjusted to: length_scale = {}, variance = {}, noise_variance = {}", length_scale_,
                         variance_, noise_variance_);
            }
        }

        void setParametersAdaptive(T new_length_scale, T new_variance, T new_noise_variance) {
            T eps = std::numeric_limits<T>::epsilon();
            T old_length_scale = length_scale_;
            T old_variance = variance_;
            T old_noise_variance = noise_variance_;

            length_scale_ = std::max(new_length_scale, eps);
            variance_ = std::max(new_variance, eps);
            noise_variance_ = std::max(new_noise_variance, T(0));

            if (std::abs(length_scale_ - old_length_scale) / old_length_scale > 0.1 ||
                std::abs(variance_ - old_variance) / old_variance > 0.1) {
                noise_variance_ = std::max(noise_variance_, old_noise_variance * 1.1);
                LOG_INFO("Adaptive regularization applied. New noise_variance: {}", noise_variance_);
            }

            if (new_length_scale != length_scale_ || new_variance != variance_ ||
                new_noise_variance != noise_variance_) {
                LOG_WARN("Kernel parameters adjusted to ensure validity: length_scale={}, variance={}, "
                         "noise_variance={}",
                         length_scale_, variance_, noise_variance_);
            } else {
                LOG_DEBUG("Updated kernel parameters: length_scale={}, variance={}, noise_variance={}", length_scale_,
                          variance_, noise_variance_);
            }
        }

        [[nodiscard]] MatrixType computeGramMatrixAdaptive(const MatrixType &X, const MatrixType &Y,
                                                           bool add_noise) const {
            const int n = X.rows();
            const int m = Y.rows();
            MatrixType K(n, m);

            const bool use_parallel = (n * m > 1000);
            constexpr int block_size = 32;

            if (use_parallel) {
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
                for (int i = 0; i < n; i += block_size) {
                    for (int j = 0; j < m; j += block_size) {
                        computeGramMatrixBlock(X, Y, K, i, j, std::min(block_size, n - i), std::min(block_size, m - j),
                                               add_noise);
                    }
                }
            } else {
                for (int i = 0; i < n; i += block_size) {
                    for (int j = 0; j < m; j += block_size) {
                        computeGramMatrixBlock(X, Y, K, i, j, std::min(block_size, n - i), std::min(block_size, m - j),
                                               add_noise);
                    }
                }
            }

            return K;
        }

        void computeGramMatrixBlock(const MatrixType &X, const MatrixType &Y, MatrixType &K, const int start_row,
                                    const int start_col, const int num_rows, const int num_cols, bool add_noise) const {
            for (int i = 0; i < num_rows; ++i) {
                for (int j = 0; j < num_cols; ++j) {
                    K(start_row + i, start_col + j) =
                            compute(X.row(start_row + i).transpose(), Y.row(start_col + j).transpose());
                    if (add_noise && &X == &Y && start_row + i == start_col + j) {
                        K(start_row + i, start_col + j) += noise_variance_;
                    }
                }
            }
        }

        [[nodiscard]] MatrixType computeGradientMatrixAdaptive(const MatrixType &X, int param_index) const {
            const int n = X.rows();
            MatrixType K_grad(n, n);

            const bool use_parallel = (n * n > 1000);
            constexpr int block_size = 32;

            if (use_parallel) {
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
                for (int i = 0; i < n; i += block_size) {
                    for (int j = 0; j < n; j += block_size) {
                        computeGradientMatrixBlock(X, K_grad, i, j, std::min(block_size, n - i),
                                                   std::min(block_size, n - j), param_index);
                    }
                }
            } else {
                for (int i = 0; i < n; i += block_size) {
                    for (int j = 0; j < n; j += block_size) {
                        computeGradientMatrixBlock(X, K_grad, i, j, std::min(block_size, n - i),
                                                   std::min(block_size, n - j), param_index);
                    }
                }
            }

            return K_grad;
        }

        void computeGradientMatrixBlock(const MatrixType &X, MatrixType &K_grad, const int start_row,
                                        const int start_col, const int num_rows, const int num_cols,
                                        int param_index) const {
            for (int i = 0; i < num_rows; ++i) {
                for (int j = 0; j < num_cols; ++j) {
                    VectorType grad =
                            computeGradient(X.row(start_row + i).transpose(), X.row(start_col + j).transpose());
                    K_grad(start_row + i, start_col + j) = grad(param_index);
                }
            }
        }
    };

} // namespace optimization::kernel

#endif // KERNEL_MATERN52_HPP


/*
#ifndef KERNEL_MATERN52_HPP
#define KERNEL_MATERN52_HPP

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/kernel.hpp"

namespace optimization::kernel {

    template<typename T = double>
    class Matern52 final : public Kernel<T> {
    public:
        using typename Kernel<T>::VectorType;
        using typename Kernel<T>::MatrixType;

        explicit Matern52(T lengthScale = 1.0, T variance = 1.0, T noiseVariance = 1e-6) :
            lengthScale_(lengthScale), variance_(variance), noiseVariance_(noiseVariance) {
            validateParameters();
            LOG_INFO("Initialized Matern 5/2 kernel with length_scale={}, variance={}, noise_variance={}", lengthScale_,
                     variance_, noiseVariance_);
        }

        [[nodiscard]] std::shared_ptr<Kernel<T>> clone() const override { return std::make_shared<Matern52>(*this); }

        [[nodiscard]] int parameterCount() const override { return 3; }

        // Compute the kernel function k(x, y)
        // Formula: k(x, y) = s^2 * (1 + sqrt(5)r/l + 5r^2/(3l^2)) * exp(-sqrt(5)r/l)
        // where r = ||x - y||, l is lengthScale, s^2 is variance
        [[nodiscard]] T compute(const VectorType &x, const VectorType &y) const noexcept override {
            const T r = (x - y).norm();
            const T z = sqrt5_ * r / lengthScale_;
            const T expTerm = std::exp(std::max(-z, logMin_));
            return variance_ * (1.0 + z + z * z / 3.0) * expTerm + std::numeric_limits<T>::epsilon();
        }

        // Gram matrix K where K_ij = k(x_i, y_j)
        [[nodiscard]] MatrixType computeGramMatrix(const MatrixType &X, const MatrixType &Y) const override {
            const int n = X.rows();
            const int m = Y.rows();
            MatrixType K(n, m);

#pragma omp parallel for collapse(2)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    K(i, j) = compute(X.row(i).transpose(), Y.row(j).transpose());
                }
            }
            return K;
        }

        // Gram matrix K where K_ij = k(x_i, x_j) + noiseVariance if i=j
        [[nodiscard]] MatrixType computeGramMatrix(const MatrixType &X) const override {
            const int n = X.rows();
            MatrixType K(n, n);

#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    K(i, j) = compute(X.row(i).transpose(), X.row(j).transpose());
                    if (i == j) {
                        K(i, i) += noiseVariance_;
                    } else {
                        K(j, i) = K(i, j);
                    }
                }
            }

            return K;
        }

        // gradient of k(x, y) with respect to hyperparameters [lengthScale, variance, noiseVariance]
        [[nodiscard]] VectorType computeGradient(const VectorType &x, const VectorType &y) const override {
            VectorType grad(3);
            const T r = (x - y).norm();
            const T z = sqrt5_ * r / lengthScale_;
            const T expTerm = std::exp(std::max(-z, logMin_));
            const T commonTerm = variance_ * expTerm;

            // Gradient with respect to lengthScale
            grad(0) = commonTerm * (5.0 / 3.0) * (z * z) * (1.0 + z) / lengthScale_;

            // Gradient with respect to variance
            grad(1) = (1.0 + z + z * z / 3.0) * expTerm;

            // Gradient with respect to noiseVariance
            grad(2) = (x.isApprox(y, 1e-8)) ? 1.0 : 0.0;

            return grad;
        }

        void setParameters(const VectorType &params) override {
            Kernel<T>::validateParameters(params, getParameterNames());
            setParameters(params(0), params(1), params(2));
        }

        void setParameters(T newLengthScale, T newVariance, T newNoiseVariance) {
            T eps = std::numeric_limits<T>::epsilon();
            T oldLengthScale = lengthScale_;
            T oldVariance = variance_;
            T oldNoiseVariance = noiseVariance_;

            lengthScale_ = std::max(newLengthScale, eps);
            variance_ = std::max(newVariance, eps);
            noiseVariance_ = std::max(newNoiseVariance, T(0));

            // Adaptive regularization: gradually increase noiseVariance if parameters change significantly
            if (std::abs(lengthScale_ - oldLengthScale) / oldLengthScale > 0.1 ||
                std::abs(variance_ - oldVariance) / oldVariance > 0.1) {
                noiseVariance_ = std::max(noiseVariance_, oldNoiseVariance * 1.1);
            }

            if (newLengthScale != lengthScale_ || newVariance != variance_ || newNoiseVariance != noiseVariance_) {
                LOG_WARN("Kernel parameters adjusted to ensure validity: length_scale={}, variance={}, "
                         "noise_variance={}",
                         lengthScale_, variance_, noiseVariance_);
            } else {
                LOG_DEBUG("Updated kernel parameters: length_scale={}, variance={}, noise_variance={}", lengthScale_,
                          variance_, noiseVariance_);
            }
        }

        [[nodiscard]] VectorType getParameters() const override {
            VectorType params(3);
            params << lengthScale_, variance_, noiseVariance_;
            return params;
        }

        [[nodiscard]] MatrixType computeGradientMatrix(const MatrixType &X, int paramIndex) const override {
            const int n = X.rows();
            MatrixType KGrad(n, n);

#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    VectorType xi = X.row(i).transpose();
                    VectorType xj = X.row(j).transpose();

                    const T r = (xi - xj).norm();
                    const T z = sqrt5_ * r / lengthScale_;
                    const T expTerm = std::exp(std::max(-z, logMin_));

                    switch (paramIndex) {
                        case 0: // Gradient with respect to lengthScale
                            KGrad(i, j) = variance_ * expTerm * (5.0 / 3.0) * (z * z) * (1.0 + z) /
                                          (lengthScale_ * lengthScale_);
                            break;
                        case 1: // Gradient with respect to variance
                            KGrad(i, j) = (1.0 + z + z * z / 3.0) * expTerm;
                            break;
                        case 2: // Gradient with respect to noiseVariance (only on diagonal)
                            KGrad(i, j) = (i == j) ? 1.0 : 0.0;
                            break;
                        default:
                            KGrad(i, j) = 0.0;
                            LOG_WARN("Invalid parameter index in computeGradientMatrix: {}", paramIndex);
                    }

                    if (i != j) {
                        KGrad(j, i) = KGrad(i, j);
                    }
                }
            }

            return KGrad;
        }

        [[nodiscard]] std::vector<std::string> getParameterNames() const override {
            return {"length_scale", "variance", "noise_variance"};
        }

        [[nodiscard]] bool isStationary() const override { return true; }
        [[nodiscard]] bool isIsotropic() const override { return true; }

        [[nodiscard]] std::string getKernelType() const override { return "Matern52"; }

    private:
        T lengthScale_; // Characteristic length scale of the kernel
        T variance_; // Signal variance
        T noiseVariance_; // Noise variance (jitter for numerical stability)
        static constexpr T sqrt5_ = 2.236067977499790; // Precomputed sqrt(5)
        static constexpr T logMin_ = std::log(std::numeric_limits<T>::min());

        // Validate parameters to ensure numerical stability
        void validateParameters() {
            T eps = std::numeric_limits<T>::epsilon();
            if (lengthScale_ <= 0 || variance_ <= 0 || noiseVariance_ < 0) {
                LOG_ERROR("Invalid kernel parameters: length_scale = {}, variance = {}, noise_variance = {}",
                          lengthScale_, variance_, noiseVariance_);
                lengthScale_ = std::max(lengthScale_, eps);
                variance_ = std::max(variance_, eps);
                noiseVariance_ = std::max(noiseVariance_, T(0));
                LOG_WARN("Parameters adjusted to: length_scale = {}, variance = {}, noise_variance = {}", lengthScale_,
                         variance_, noiseVariance_);
            }
        }
    };

} // namespace optimization::kernel

#endif // KERNEL_MATERN52_HPP
*/
