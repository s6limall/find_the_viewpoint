// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
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
            kernel_(kernel), noise_variance_(noise_variance) {
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
        std::pair<double, double> predict(const VectorXd &x) const {
            if (X_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                throw std::runtime_error("Model not fitted.");
            }

            // Compute k(X, x) - correlation between x and all training points
            VectorXd k_star = kernel_.computeGramMatrix(X_, x.transpose());

            // Compute the mean: E[f(x)] = k(X, x)^T * alpha
            // alpha is precomputed in updateModel() for efficiency
            double mean = k_star.dot(alpha_);

            // Compute the variance: Var[f(x)] = k(x, x) - k(X, x)^T * K^(-1) * k(X, x)
            // Use Cholesky decomposition for numerical stability
            double variance = kernel_.compute(x, x) - k_star.dot(L_.triangularView<Eigen::Lower>().solve(k_star));

            LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean, variance);
            return {mean, std::max(0.0, variance)};
        }

        // Update the model with a new data point
        void update(const VectorXd &new_x, double new_y, Eigen::Index max_points = 1000) {
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

        // Optimize the hyperparameters of the kernel
        void optimizeHyperparameters(int max_iterations = 100) {
            LOG_INFO("Starting hyperparameter optimization");
            VectorXd params = kernel_.getParameters();
            VectorXd best_params = params;
            double best_lml = -std::numeric_limits<double>::infinity();

            constexpr int m = 10; // L-BFGS memory
            std::array<VectorXd, m> s, y;
            int iter = 0;
            VectorXd grad = computeLogMarginalLikelihoodGradient();

            for (int i = 0; i < max_iterations; ++i) {
                // Compute search direction using L-BFGS
                VectorXd direction = computeLBFGSDirection(grad, s, y, iter);

                // Perform line search to find step size
                double step_size = backtrackingLineSearch(params, direction, grad);

                // Update parameters: theta_new = theta + alpha * direction
                VectorXd new_params = params + step_size * direction;
                new_params = new_params.cwiseMax(1e-6).cwiseMin(10.0); // Constrain parameters

                setParameters(new_params);
                updateModel();

                double lml = computeLogMarginalLikelihood();
                VectorXd new_grad = computeLogMarginalLikelihoodGradient();

                if (lml > best_lml) {
                    best_lml = lml;
                    best_params = new_params;
                    LOG_DEBUG("New best LML: {} at iteration {}", best_lml, i);
                }

                // Update L-BFGS memory
                VectorXd s_i = new_params - params;
                VectorXd y_i = new_grad - grad;

                if (iter < m) {
                    s[iter] = s_i;
                    y[iter] = y_i;
                } else {
                    std::rotate(s.begin(), s.begin() + 1, s.end());
                    std::rotate(y.begin(), y.begin() + 1, y.end());
                    s.back() = s_i;
                    y.back() = y_i;
                }

                params = new_params;
                grad = new_grad;
                iter = std::min(iter + 1, m);

                if (grad.norm() < 1e-5) {
                    LOG_INFO("Optimization converged after {} iterations", i);
                    break;
                }
            }

            setParameters(best_params);
            updateModel();
            LOG_INFO("Hyperparameter optimization completed. Final LML: {}", best_lml);
        }

    private:
        Kernel kernel_;
        double noise_variance_;
        MatrixXd X_;
        VectorXd y_;
        MatrixXd L_; // Cholesky decomposition of K
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency

        // Update the internal model after changes to X_, y_, or kernel parameters
        void updateModel() {
            // Compute the kernel matrix K
            MatrixXd K = kernel_.computeGramMatrix(X_);
            K.diagonal().array() += noise_variance_;

            // Perform Cholesky decomposition: K = LL^T
            Eigen::LLT<MatrixXd> llt(K);
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

        // Compute the log marginal likelihood
        // LML = -0.5 * (y^T * alpha + log|K| + n*log(2*pi))
        double computeLogMarginalLikelihood() const {
            double log_det_K = 2 * L_.diagonal().array().log().sum(); // log|K| = 2 * sum(log(diag(L)))
            return -0.5 * (y_.dot(alpha_) + log_det_K + X_.rows() * std::log(2 * M_PI));
        }

        // Compute the gradient of the log marginal likelihood
        // This is used for optimizing the hyperparameters
        VectorXd computeLogMarginalLikelihoodGradient() const {
            int n = X_.rows();
            // K^(-1) = L^(-T) * L^(-1)
            MatrixXd K_inv = L_.triangularView<Eigen::Lower>().solve(
                    L_.triangularView<Eigen::Lower>().transpose().solve(MatrixXd::Identity(n, n)));
            MatrixXd alpha_alpha_t = alpha_ * alpha_.transpose();
            // dLML/dtheta_i = 0.5 * tr((alpha*alpha^T - K^(-1)) * dK/dtheta_i)
            MatrixXd factor = (alpha_alpha_t - K_inv) * 0.5;

            VectorXd gradient(3);
            for (int i = 0; i < 3; ++i) {
                MatrixXd K_grad = kernel_.computeGradientMatrix(X_, i);
                gradient(i) = (factor.array() * K_grad.array()).sum();
            }
            return gradient;
        }

        // Compute the L-BFGS direction for optimization
        // This method implements the two-loop recursion algorithm
        VectorXd computeLBFGSDirection(const VectorXd &grad, const std::array<VectorXd, 10> &s,
                                       const std::array<VectorXd, 10> &y, int iter) const {
            VectorXd q = -grad;
            std::array<double, 10> alpha;

            // First loop
            for (int i = iter - 1; i >= 0; --i) {
                alpha[i] = s[i].dot(q) / y[i].dot(s[i]);
                q -= alpha[i] * y[i];
            }

            // Scaling factor
            VectorXd r = q * (s[iter - 1].dot(y[iter - 1]) / y[iter - 1].squaredNorm());

            // Second loop
            for (int i = 0; i < iter; ++i) {
                double beta = y[i].dot(r) / y[i].dot(s[i]);
                r += s[i] * (alpha[i] - beta);
            }

            return r;
        }

        // Perform backtracking line search to find an appropriate step size
        double backtrackingLineSearch(const VectorXd &x, const VectorXd &p, const VectorXd &grad) {
            double alpha = 1.0;
            const double c = 0.5;
            const double tau = 0.5;
            double f_x = computeLogMarginalLikelihood();
            double g_p = grad.dot(p);

            for (int i = 0; i < 10; ++i) { // Max 10 iterations
                VectorXd x_new = x + alpha * p;
                setParameters(x_new);
                updateModel();
                double f_new = computeLogMarginalLikelihood();

                // Check Armijo condition
                if (f_new >= f_x + c * alpha * g_p) {
                    return alpha;
                }
                alpha *= tau;
            }

            LOG_WARN("Line search did not converge, using minimum step size");
            return alpha;
        }

        // Utility function
        void setParameters(const VectorXd &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Expected 3 parameters for Matern52 kernel.");
            }
            kernel_.setParameters(params(0), params(1), params(2));
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
