// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/hyperparameter_optimizer.hpp"
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcessRegression {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        explicit GaussianProcessRegression(const Kernel &kernel, double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(noise_variance),
            optimizer_() { // Initialize optimizer with default settings
            LOG_INFO("Initialized Gaussian Process Regression with noise variance: {}", noise_variance);
        }

        // Fit the GPR model to the training data
        void fit(const MatrixXd &X, const VectorXd &y) {
            X_ = X;
            y_ = y;
            updateModel();
            LOG_INFO("Fitted GPR model with {} data points", X_.rows());
        }

        // Predict the mean and variance for a new input point
        [[nodiscard]] std::pair<double, double> predict(const VectorXd &x) const {
            if (X_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                throw std::runtime_error("Model not fitted.");
            }

            // Compute k(X, x) - correlation between x and all training points
            const VectorXd k_star = kernel_.computeGramMatrix(X_, x.transpose());

            // Compute the mean: E[f(x)] = k(X, x)^T * alpha
            // alpha is precomputed in updateModel() for efficiency
            double mean = k_star.dot(alpha_);

            // Compute the variance: Var[f(x)] = k(x, x) - k(X, x)^T * K^(-1) * k(X, x)
            // Use Cholesky decomposition for numerical stability
            double variance = kernel_.compute(x, x) - k_star.dot(L_.triangularView<Eigen::Lower>().solve(k_star));

            // Ensure variance is never negative by adding a small epsilon
            constexpr double epsilon = 1e-10; // Small value to ensure variance stays positive
            variance = std::max(variance, epsilon);

            LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean, variance);
            return {mean, variance};
        }

        // Update the model with a new data point
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

        // Optimize the hyperparameters of the kernel using the persistent optimizer
        void optimizeHyperparameters() {
            LOG_INFO("Starting hyperparameter optimization");
            VectorXd best_params = optimizer_.optimize(X_, y_, kernel_);
            setParameters(best_params);
            updateModel();
            LOG_INFO("Hyperparameter optimization completed with parameters: {}", best_params);
        }

    private:
        Kernel kernel_;
        double noise_variance_;
        MatrixXd X_;
        VectorXd y_;
        MatrixXd L_; // Cholesky decomposition of K
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency
        HyperparameterOptimizer<Kernel> optimizer_; // Persistent optimizer

        // Update the internal model after changes to X_, y_, or kernel parameters
        void updateModel() {
            // Compute the kernel matrix K
            MatrixXd K = kernel_.computeGramMatrix(X_);
            K.diagonal().array() += noise_variance_;

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

        // Utility function to set kernel parameters
        void setParameters(const VectorXd &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Expected 3 parameters for Matern52 kernel.");
            }
            kernel_.setParameters(params(0), params(1), params(2));
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
