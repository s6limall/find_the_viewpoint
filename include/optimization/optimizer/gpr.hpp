// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <vector>
#include "../../common/logging/logger.hpp"
#include "../kernel/matern_52.hpp"

namespace optimization {

    /*
     * Gaussian Process Regression / Kriging
     */
    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcessRegression {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        explicit GaussianProcessRegression(const Kernel &kernel, const double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(noise_variance) {}

        void initializeWithPoint(const Eigen::VectorXd &x, const double y) {
            X_ = x.transpose();
            y_ = Eigen::VectorXd::Constant(1, y);
            updateModel();
        }

        void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
            X_ = X;
            y_ = y;
            updateModel();
        }

        std::pair<double, double> predict(const Eigen::VectorXd &x) const {
            if (X_.rows() == 0)
                throw std::runtime_error("Model not fitted.");

            const Eigen::VectorXd k_star = computeKernelVector(x);
            double mean = k_star.dot(alpha_);
            const double variance = kernel_.compute(x, x) - k_star.dot(L_.triangularView<Eigen::Lower>().solve(k_star));

            return {mean, std::max(0.0, variance)};
        }

        void update(const Eigen::VectorXd &new_x, const double new_y, const size_t max_points = 1000) {
            X_.conservativeResize(X_.rows() + 1, Eigen::NoChange);
            X_.row(X_.rows() - 1) = new_x.transpose();
            y_.conservativeResize(y_.size() + 1);
            y_(y_.size() - 1) = new_y;

            if (X_.rows() > max_points) {
                // Remove the oldest point
                X_ = X_.bottomRows(max_points);
                y_ = y_.tail(max_points);
            }

            updateModel();
        }

        double expectedImprovement(const Eigen::VectorXd &x, const double f_best) const {
            auto [mu, sigma2] = predict(x);
            const double sigma = std::sqrt(sigma2);
            if (sigma < 1e-10)
                return 0.0;

            const double z = (mu - f_best) / sigma;
            return (mu - f_best) * 0.5 * std::erfc(-z / std::sqrt(2.0)) +
                   sigma * std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
        }

        void optimizeHyperparameters(const int max_iterations = 50) {
            Eigen::Vector3d params = kernel_.getParameters();
            double learning_rate = 0.01;
            double best_lml = -std::numeric_limits<double>::infinity();

            for (int i = 0; i < max_iterations; ++i) {
                Eigen::Vector3d gradient = computeLogMarginalLikelihoodGradient();
                params += learning_rate * gradient;
                params = params.cwiseMax(1e-6).cwiseMin(10.0); // Constrain parameters

                kernel_.setParameters(params(0), params(1), params(2));
                updateModel();

                const double lml = computeLogMarginalLikelihood();
                if (lml > best_lml) {
                    best_lml = lml;
                } else {
                    learning_rate *= 0.5; // Reduce learning rate if no improvement
                }

                if (gradient.norm() < 1e-5 || learning_rate < 1e-10) {
                    break; // Convergence criteria
                }
            }
        }

    private:
        Kernel kernel_;
        double noise_variance_;
        Eigen::MatrixXd X_;
        Eigen::VectorXd y_;
        Eigen::MatrixXd L_;
        Eigen::VectorXd alpha_;

        void updateModel() {
            Eigen::MatrixXd K = computeKernelMatrix();
            Eigen::LLT<Eigen::MatrixXd> llt(K);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("Cholesky decomposition failed.");
            }
            L_ = llt.matrixL();
            alpha_ = L_.triangularView<Eigen::Lower>().solve(y_);
            alpha_ = L_.triangularView<Eigen::Lower>().adjoint().solve(alpha_);
        }

        Eigen::MatrixXd computeKernelMatrix() const {
            Eigen::MatrixXd K = kernel_.computeGramMatrix(X_);
            K.diagonal().array() += noise_variance_;
            return K;
        }

        Eigen::VectorXd computeKernelVector(const Eigen::VectorXd &x) const {
            return kernel_.computeGramMatrix(X_, x.transpose().eval());
        }

        double computeLogMarginalLikelihood() const {
            const double log_det_K = 2 * L_.diagonal().array().log().sum();
            return -0.5 * (y_.dot(alpha_) + log_det_K + X_.rows() * std::log(2 * M_PI));
        }

        Eigen::Vector3d computeLogMarginalLikelihoodGradient() {
            int n = X_.rows();
            const MatrixXd K_inv = L_.triangularView<Eigen::Lower>().solve(
                    L_.triangularView<Eigen::Lower>().transpose().solve(MatrixXd::Identity(n, n)));
            const MatrixXd alpha_alpha_t = alpha_ * alpha_.transpose();
            MatrixXd factor = (alpha_alpha_t - K_inv) * 0.5;

            Eigen::Vector3d gradient;
            for (int i = 0; i < 3; ++i) {
                MatrixXd K_grad(n, n);
                for (int j = 0; j < n; ++j) {
                    for (int k = j; k < n; ++k) {
                        Eigen::VectorXd grad_jk = kernel_.computeGradient(X_.row(j).transpose(), X_.row(k).transpose());
                        K_grad(j, k) = grad_jk(i);
                        if (j != k) {
                            K_grad(k, j) = K_grad(j, k);
                        }
                    }
                }
                gradient(i) = (factor.array() * K_grad.array()).sum();
            }
            return gradient;
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
