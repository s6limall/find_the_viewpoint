// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcessRegression {
    public:
        explicit GaussianProcessRegression(const Kernel &kernel, double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(noise_variance) {}

        void initializeWithPoint(const Eigen::VectorXd &x, double y) {
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

            Eigen::VectorXd k_star = computeKernelVector(x);
            double mean = k_star.dot(alpha_);
            double variance = kernel_.compute(x, x) - k_star.dot(L_.triangularView<Eigen::Lower>().solve(k_star));

            return {mean, std::max(0.0, variance)};
        }

        void update(const Eigen::VectorXd &new_x, double new_y, size_t max_points = 1000) {
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

        double expectedImprovement(const Eigen::VectorXd &x, double f_best) const {
            auto [mu, sigma2] = predict(x);
            double sigma = std::sqrt(sigma2);
            if (sigma < 1e-10)
                return 0.0;

            double z = (mu - f_best) / sigma;
            return (mu - f_best) * 0.5 * std::erfc(-z / std::sqrt(2.0)) +
                   sigma * std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
        }

        void optimizeHyperparameters(int n_restarts = 10) {
            Eigen::Vector3d best_params = kernel_.getParameters();
            double best_lml = computeLogMarginalLikelihood();

            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<> dist(0.1, 10.0);

            for (int i = 0; i < n_restarts; ++i) {
                Eigen::Vector3d params(dist(gen), dist(gen), 1e-6 * dist(gen));
                kernel_.setParameters(params(0), params(1), params(2));
                updateModel();

                double lml = computeLogMarginalLikelihood();
                if (lml > best_lml) {
                    best_lml = lml;
                    best_params = params;
                }
            }

            kernel_.setParameters(best_params(0), best_params(1), best_params(2));
            updateModel();
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
            double log_det_K = 2 * L_.diagonal().array().log().sum();
            return -0.5 * (y_.dot(alpha_) + log_det_K + X_.rows() * std::log(2 * M_PI));
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
