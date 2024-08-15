// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcessRegression {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        explicit GaussianProcessRegression(const Kernel &kernel, double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(noise_variance), jitter_(1e-8) {
            LOG_INFO("Initialized Gaussian Process Regression with noise variance: {}", noise_variance);
        }

        void fit(const MatrixXd &X, const VectorXd &y) {
            X_ = X;
            y_ = y;
            updateModel();
            LOG_INFO("Fitted GPR model with {} data points", X_.rows());
        }

        [[nodiscard]] std::pair<double, double> predict(const VectorXd &x) const {
            if (X_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                throw std::runtime_error("Model not fitted.");
            }

            const VectorXd k_star = kernel_.computeGramMatrix(X_, x.transpose());
            const double mean = k_star.dot(alpha_);

            // Numerically stable variance computation
            const VectorXd v = L_.triangularView<Eigen::Lower>().solve(k_star);
            double variance = kernel_.compute(x, x) - v.dot(v);

            // Ensure non-negative variance
            variance = std::max(0.0, variance);

            LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean, variance);
            return {mean, variance};
        }

        void update(const VectorXd &new_x, const double new_y, const Eigen::Index max_points = 1000) {
            if (X_.rows() >= max_points) {
                // Use sliding window approach for online learning
                X_.bottomRows(max_points - 1) = X_.topRows(max_points - 1);
                y_.tail(max_points - 1) = y_.head(max_points - 1);
                X_.row(max_points - 1) = new_x.transpose();
                y_(max_points - 1) = new_y;
            } else {
                // Append new point if we haven't reached max_points
                X_.conservativeResize(X_.rows() + 1, Eigen::NoChange);
                X_.row(X_.rows() - 1) = new_x.transpose();
                y_.conservativeResize(y_.size() + 1);
                y_(y_.size() - 1) = new_y;
            }
            updateModel();
            LOG_INFO("Updated model with new point. Total points: {}", X_.rows());
        }

        void optimizeHyperparameters(const int max_iterations = 100) {
            VectorXd params = kernel_.getParameters();
            double best_lml = -std::numeric_limits<double>::infinity();
            VectorXd best_params = params;

            for (int i = 0; i < max_iterations; ++i) {
                const double lml = computeLogMarginalLikelihood();
                if (lml > best_lml) {
                    best_lml = lml;
                    best_params = params;
                }

                VectorXd grad = computeLogMarginalLikelihoodGradient();
                params += 0.01 * grad; // Simple gradient ascent
                params = clampParameters(params); // Clamp parameters to valid range
                setParameters(params);
                updateModel();
            }

            setParameters(best_params);
            updateModel();
            LOG_INFO("Hyperparameter optimization complete. Final LML: {}", best_lml);
        }

    private:
        Kernel kernel_;
        double noise_variance_;
        double jitter_;
        MatrixXd X_;
        VectorXd y_;
        MatrixXd L_; // Cholesky decomposition of K
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency

        void updateModel() {
            // Compute the kernel matrix K
            MatrixXd K = kernel_.computeGramMatrix(X_);
            K.diagonal().array() += noise_variance_ + jitter_;

            // Check condition number
            const Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
            double condition_number = eigensolver.eigenvalues().maxCoeff() / eigensolver.eigenvalues().minCoeff();
            if (condition_number > 1e6) {
                LOG_WARN("Kernel matrix is ill-conditioned. Condition number: {}", condition_number);
            }

            // Perform Cholesky decomposition: K = LL^T
            const Eigen::LLT<MatrixXd> llt(K);
            if (llt.info() != Eigen::Success) {
                LOG_ERROR("Cholesky decomposition failed");
                throw std::runtime_error("Cholesky decomposition failed.");
            }
            L_ = llt.matrixL();

            // Compute alpha = K^(-1) * y using Cholesky decomposition
            // First solve L * v = y, then L^T * alpha = v
            alpha_ = L_.triangularView<Eigen::Lower>().solve(y_);
            alpha_ = L_.triangularView<Eigen::Lower>().adjoint().solve(alpha_);
        }


        [[nodiscard]] VectorXd computeLogMarginalLikelihoodGradient() const {
            const MatrixXd K_inv = L_.triangularView<Eigen::Lower>().solve(
                    L_.triangularView<Eigen::Lower>().transpose().solve(MatrixXd::Identity(X_.rows(), X_.rows())));
            const MatrixXd alpha_alpha_t = alpha_ * alpha_.transpose();
            MatrixXd factor = (alpha_alpha_t - K_inv) * 0.5;

            VectorXd gradient(3);
            for (int i = 0; i < 3; ++i) {
                MatrixXd K_grad = kernel_.computeGradientMatrix(X_, i);
                gradient(i) = (factor.array() * K_grad.array()).sum();
            }
            return gradient;
        }

        [[nodiscard]] double computeLogMarginalLikelihood() const {
            const double log_det_K = 2 * L_.diagonal().array().log().sum(); // log|K| = 2 * sum(log(diag(L)))
            return -0.5 * (y_.dot(alpha_) + log_det_K + X_.rows() * std::log(2 * M_PI));
        }


        void setParameters(const VectorXd &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Expected 3 parameters for Matern52 kernel.");
            }
            kernel_.setParameters(params(0), params(1), params(2));
        }

        // Add this method to clamp parameter values:
        static VectorXd clampParameters(const VectorXd &params) {
            VectorXd clamped = params;
            clamped(0) = std::max(1e-6, params(0)); // length_scale
            clamped(1) = std::max(1e-6, params(1)); // variance
            clamped(2) = std::max(0.0, params(2)); // noise_variance
            return clamped;
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
