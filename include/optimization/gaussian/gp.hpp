// File: optimization/gaussian/gp.hpp


#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_HPP
#define OPTIMIZATION_GAUSIAN_PROCESS_HPP

#include <Eigen/Dense>
#include <limits>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/kernel.hpp"
#include "optimization/hyperparameter_optimizer.hpp"

namespace optimization {

    template<FloatingPoint T = double, optimization::IsKernel<T> KernelType = optimization::DefaultKernel<T>>
    class GaussianProcess {
    public:
        using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXd = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        explicit GaussianProcess(const KernelType &kernel, const T noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(std::max(noise_variance, std::numeric_limits<T>::epsilon())),
            optimizer_() {
            LOG_INFO("Initialized Gaussian Process with noise variance: {}", noise_variance_);
        }

        // Set the data matrix X and precompute the covariance matrix
        void setData(const MatrixXd &X) {
            if (X.rows() == 0 || X.cols() == 0) {
                LOG_ERROR("Attempted to set empty data");
                throw std::invalid_argument("Input data X cannot be empty.");
            }

            try {
                // Directly assign input data to avoid unnecessary resizing
                this->x_data_ = X;

                // Update the covariance matrix based on the new data
                updateCovariance();
                LOG_INFO("Set data with {} points of dimension {}", x_data_.rows(), x_data_.cols());
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to set data: {}. Attempting fallback.", e.what());
                fallbackSetData(X);
            }
        }


        // Optimize the kernel hyperparameters based on the provided target values
        void optimizeHyperparameters(const VectorXd &y, const VectorXd &lower_bounds, const VectorXd &upper_bounds) {
            LOG_DEBUG("Starting hyperparameter optimization");

            if (x_data_.rows() == 0 || y.size() == 0) {
                LOG_ERROR("No data available for hyperparameter optimization. x_data_ rows={}, y size={}",
                          x_data_.rows(), y.size());
                throw std::runtime_error("No data available for hyperparameter optimization.");
            }

            try {
                VectorXd best_params = optimizer_.optimizeBounded(x_data_, y, kernel_, lower_bounds, upper_bounds);
                setParameters(best_params);
                updateCovariance();
                LOG_INFO("Hyperparameter optimization completed successfully with parameters: {}", best_params);
            } catch (const std::exception &e) {
                LOG_ERROR("Hyperparameter optimization failed: {}. Attempting robust optimization.", e.what());
                robustOptimizeHyperparameters(y, lower_bounds, upper_bounds);
            }
        }

        // Compute covariance between two datasets X1 and X2
        [[nodiscard]] MatrixXd computeCovariance(const MatrixXd &X1, const MatrixXd &X2) const {
            return kernel_.computeGramMatrix(X1, X2);
        }

        // Compute the posterior mean and covariance for new test points X_star
        [[nodiscard]] std::pair<VectorXd, MatrixXd> posterior(const MatrixXd &X_star) const {
            if (x_data_.rows() == 0) {
                LOG_ERROR("Attempted posterior computation with no data");
                throw std::runtime_error("No data available for posterior computation.");
            }

            MatrixXd K_star = computeCovariance(x_data_, X_star);
            MatrixXd K_star_star = computeCovariance(X_star, X_star);

            VectorXd mean = K_star.transpose() * alpha_;

            MatrixXd cov;
            try {
                cov = K_star_star - K_star.transpose() * ldlt_.solve(K_star);
                LOG_DEBUG("Posterior covariance successfully computed");
            } catch (const std::exception &e) {
                LOG_ERROR("Posterior covariance computation failed: {}. Attempting fallback method.", e.what());
                return fallbackPosterior(X_star, K_star, K_star_star);
            }

            // Ensure positive semi-definiteness by cleaning covariance matrix
            cov = cleanCovarianceMatrix(cov);
            LOG_INFO("Posterior mean and covariance successfully computed");
            return {mean, cov};
        }

    protected:
        KernelType kernel_;
        T noise_variance_;
        MatrixXd x_data_;
        Eigen::LDLT<MatrixXd> ldlt_; // LDLT decomposition of the covariance matrix
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency
        HyperparameterOptimizer<T, KernelType> optimizer_;

        // Update the covariance matrix based on current data and kernel parameters
        void updateCovariance() {
            if (x_data_.rows() == 0) {
                LOG_WARN("No data available to update covariance matrix");
                return;
            }

            try {
                MatrixXd K = computeCovariance(x_data_, x_data_);
                K.diagonal().array() += noise_variance_;
                K = (K + K.transpose()) / 2.0; // Ensure symmetry

                // Perform regularized LDLT decomposition
                ldlt_ = performRegularizedLDLT(K);
                LOG_DEBUG("Covariance matrix updated successfully");

            } catch (const std::exception &e) {
                LOG_ERROR("Error updating covariance matrix: {}. Using fallback method.", e.what());
                fallbackUpdateCovariance();
            }
        }

        // Fallback methods for setting data
        void fallbackSetData(const MatrixXd &X) {
            LOG_WARN("Attempting fallback mechanism for setting data");

            try {
                x_data_ = MatrixXd::Zero(X.rows(), X.cols());
                x_data_ += X;
                updateCovariance();
                LOG_INFO("Fallback data setting successful");

            } catch (const std::exception &e) {
                LOG_ERROR("Fallback data setting failed: {}", e.what());
                x_data_ = MatrixXd::Constant(1, X.cols(), 0.0); // Minimal safe fallback
                ldlt_.compute(MatrixXd::Identity(1, 1));
                LOG_WARN("Set default minimal data matrix after fallback failure");
            }
        }

        // Set kernel parameters and update covariance matrix accordingly
        void setParameters(const VectorXd &params) {
            if (params.size() != kernel_.getParameterCount()) {
                LOG_ERROR("Invalid number of parameters provided for kernel. Expected {}, got {}",
                          kernel_.getParameterCount(), params.size());
                throw std::invalid_argument("Incorrect number of kernel parameters.");
            }

            kernel_.setParameters(params);
            updateCovariance();
            LOG_DEBUG("Kernel parameters successfully updated: {}", params);
        }

        // Perform regularized LDLT decomposition with adaptive regularization
        Eigen::LDLT<MatrixXd> performRegularizedLDLT(MatrixXd K) {
            Eigen::LDLT<MatrixXd> ldlt;
            T lambda = 0;
            const T max_lambda = 1e-3;
            const T lambda_factor = 10;

            // Retry with increasing regularization until decomposition succeeds
            while (lambda < max_lambda) {
                ldlt.compute(K + lambda * MatrixXd::Identity(K.rows(), K.cols()));
                if (ldlt.info() == Eigen::Success) {
                    LOG_DEBUG("LDLT decomposition successful with regularization lambda = {}", lambda);
                    return ldlt;
                }
                lambda = (lambda == 0) ? 1e-9 : lambda * lambda_factor;
            }

            if (ldlt.info() != Eigen::Success) {
                LOG_ERROR("LDLT decomposition failed even with regularization");
                throw std::runtime_error("LDLT decomposition failed despite regularization attempts");
            }

            return ldlt;
        }

        // Clean covariance matrix by ensuring positive semi-definiteness
        MatrixXd cleanCovarianceMatrix(const MatrixXd &cov) const {
            MatrixXd cleaned_cov = (cov + cov.transpose()) / 2.0;
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(cleaned_cov);
            VectorXd eigenvalues = eigen_solver.eigenvalues().cwiseMax(0.0);
            cleaned_cov =
                    eigen_solver.eigenvectors() * eigenvalues.asDiagonal() * eigen_solver.eigenvectors().transpose();
            LOG_DEBUG("Covariance matrix cleaned for positive semi-definiteness");
            return cleaned_cov;
        }

        // Fallback method for updating the covariance matrix
        void fallbackUpdateCovariance() {
            LOG_WARN("Fallback for covariance matrix update initiated");

            try {
                MatrixXd K = computeCovariance(x_data_, x_data_);
                K.diagonal().array() += noise_variance_;
                Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(K);
                VectorXd eigenvalues = eigen_solver.eigenvalues().cwiseMax(1e-9); // Ensure all eigenvalues are positive
                MatrixXd V = eigen_solver.eigenvectors();
                MatrixXd L = eigenvalues.cwiseSqrt().asDiagonal();
                K = V * L * L * V.transpose();

                ldlt_.compute(K);
                LOG_WARN("Covariance matrix fallback successful");

            } catch (const std::exception &e) {
                LOG_ERROR("Fallback covariance matrix update failed: {}", e.what());
                throw std::runtime_error("Failed to update covariance matrix using fallback.");
            }
        }

        // Robust optimization of hyperparameters in case of failure
        void robustOptimizeHyperparameters(const VectorXd &y, const VectorXd &lower_bounds,
                                           const VectorXd &upper_bounds) {
            const int max_attempts = 5;
            const T perturbation_factor = 0.1;

            VectorXd best_params = kernel_.getParameters();
            T best_likelihood = -std::numeric_limits<T>::infinity();

            for (int attempt = 0; attempt < max_attempts; ++attempt) {
                try {
                    VectorXd perturbed_lower = lower_bounds * (1 - perturbation_factor * attempt);
                    VectorXd perturbed_upper = upper_bounds * (1 + perturbation_factor * attempt);
                    VectorXd params = optimizer_.optimizeBounded(x_data_, y, kernel_, perturbed_lower, perturbed_upper);
                    setParameters(params);
                    T likelihood = computeLogLikelihood(y);

                    if (likelihood > best_likelihood) {
                        best_params = params;
                        best_likelihood = likelihood;
                    }

                    LOG_INFO("Robust optimization attempt {} succeeded", attempt + 1);
                    break;
                } catch (const std::exception &e) {
                    LOG_WARN("Robust optimization attempt {} failed: {}", attempt + 1, e.what());
                }
            }

            setParameters(best_params);
            LOG_INFO("Robust hyperparameter optimization completed with parameters: {}", best_params);
        }

        // Fallback method for posterior computation
        [[nodiscard]] std::pair<VectorXd, MatrixXd> fallbackPosterior(const MatrixXd &X_star, const MatrixXd &K_star,
                                                                      const MatrixXd &K_star_star) const {
            VectorXd mean = K_star.transpose() * (x_data_.transpose() * x_data_ +
                                                  noise_variance_ * MatrixXd::Identity(x_data_.rows(), x_data_.rows()))
                                                         .ldlt()
                                                         .solve(x_data_.transpose() * alpha_);
            MatrixXd cov = K_star_star - K_star.transpose() * K_star / (noise_variance_ * x_data_.rows());

            // Clean the covariance matrix
            cov = cleanCovarianceMatrix(cov);
            LOG_WARN("Fallback posterior computation successful");

            return {mean, cov};
        }

        // Compute the log-likelihood of the current model
        [[nodiscard]] T computeLogLikelihood(const VectorXd &y) const {
            if (y.size() != x_data_.rows()) {
                LOG_ERROR("Size mismatch between target vector and data matrix rows");
                throw std::invalid_argument("y size does not match number of data points.");
            }

            T n = static_cast<T>(y.size());
            T log_det_K = ldlt_.vectorD().array().log().sum();
            T quadratic = y.transpose() * alpha_;
            return -0.5 * (quadratic + log_det_K + n * std::log(2 * M_PI));
        }
    };
} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_HPP
