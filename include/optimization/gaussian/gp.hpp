// File: optimization/gaussian_process.hpp

#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_HPP

#include <Eigen/Dense>
#include <stdexcept>
#include <limits>
#include "common/logging/logger.hpp"
#include "optimization/hyperparameter_optimizer.hpp"

namespace optimization {

template<typename Kernel = kernel::Matern52<>>
class GaussianProcess {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;

    explicit GaussianProcess(const Kernel& kernel, const double noise_variance = 1e-6)
        : kernel_(kernel), noise_variance_(std::max(noise_variance, std::numeric_limits<double>::epsilon())), optimizer_() {
        noise_variance_ = config::get("optimization.gp.kernel.hyperparameters.noise_variance", noise_variance_);
        LOG_INFO("Initialized Gaussian Process with noise variance: {}", noise_variance_);
    }

    void setData(const MatrixXd& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            LOG_ERROR("Attempted to set empty data!");
            throw std::invalid_argument("Input data X cannot be empty.");
        }
        X_ = X;
        updateCovariance();
        LOG_INFO("Set data with {} points of dimension {}", X_.rows(), X_.cols());
    }

    void optimizeHyperparameters(const VectorXd& y) {
        if (X_.rows() == 0 || y.size() == 0) {
            LOG_ERROR("Attempted hyperparameter optimization with no data");
            throw std::runtime_error("No data available for hyperparameter optimization.");
        }
        LOG_INFO("Starting hyperparameter optimization");
        VectorXd best_params = optimizer_.optimize(X_, y, kernel_);
        setParameters(best_params);
        updateCovariance();
        LOG_INFO("Hyperparameter optimization completed with parameters: {}", best_params.transpose());
    }

    // Compute the kernel function k(X1, X2)
    [[nodiscard]] MatrixXd computeCovariance(const MatrixXd& X1, const MatrixXd& X2) const {
        return kernel_.computeGramMatrix(X1, X2);
    }

    // Compute the posterior distribution
    // Returns mean and covariance of f_star ~ N(mean, cov)
    [[nodiscard]] std::pair<VectorXd, MatrixXd> posterior(const MatrixXd& X_star) const {
        if (X_.rows() == 0) {
            LOG_ERROR("Attempted posterior computation with no data");
            throw std::runtime_error("No data set for Gaussian Process.");
        }

        // Compute K(X, X_star)
        MatrixXd K_star = computeCovariance(X_, X_star);

        // Compute K(X_star, X_star)
        const MatrixXd K_star_star = computeCovariance(X_star, X_star);

        // Compute the mean: E[f_star] = K_star^T * K^(-1) * y = K_star^T * alpha
        VectorXd mean = K_star.transpose() * alpha_;

        // Compute the covariance: Var[f_star] = K_star_star - K_star^T * K^(-1) * K_star
        MatrixXd cov = K_star_star - K_star.transpose() * L_.triangularView<Eigen::Lower>().solve(K_star);

        // Ensure numerical stability of the covariance matrix
        cov = (cov + cov.transpose()) / 2.0;  // Ensure symmetry
        const Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(cov);
        const VectorXd eigenvalues = eigenSolver.eigenvalues().cwiseMax(0.0);  // Ensure non-negative eigenvalues
        cov = eigenSolver.eigenvectors() * eigenvalues.asDiagonal() * eigenSolver.eigenvectors().transpose();

        return {mean, cov};
    }

protected:
    Kernel kernel_;
    double noise_variance_;
    MatrixXd X_;
    MatrixXd L_;  // Cholesky decomposition of K
    VectorXd alpha_;  // alpha = K^(-1) * y, precomputed for efficiency
    HyperparameterOptimizer<Kernel> optimizer_;

    void updateCovariance() {
        if (X_.rows() == 0) {
            LOG_WARN("Attempted to update covariance with no data");
            return;
        }

        // Compute the kernel matrix K
        MatrixXd K = computeCovariance(X_, X_);
        K.diagonal().array() += noise_variance_;

        // Ensure K is symmetric
        K = (K + K.transpose()) / 2.0;

        // Perform Cholesky decomposition: K = LL^T
        const Eigen::LLT<MatrixXd> llt(K);
        if (llt.info() != Eigen::Success) {
            LOG_ERROR("Cholesky decomposition failed");
            throw std::runtime_error("Cholesky decomposition failed. The kernel matrix may not be positive definite.");
        }
        L_ = llt.matrixL();

        LOG_DEBUG("Updated covariance matrix");
    }

    void setParameters(const VectorXd& params) {
        if (params.size() != 3) {
            throw std::invalid_argument("Expected 3 parameters for kernel.");
        }
        kernel_.setParameters(params(0), params(1), params(2));
        LOG_DEBUG("Updated kernel parameters: {}", params);
    }
};

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_HPP