// File: optimization/gaussian/gp.hpp

#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_HPP

#include <Eigen/Dense>
#include <limits>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"
#include "optimization/hyperparameter_optimizer.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcess {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        explicit GaussianProcess(const Kernel &kernel, const double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(std::max(noise_variance, std::numeric_limits<double>::epsilon())),
            optimizer_() {
            noise_variance_ = config::get("optimization.gp.kernel.hyperparameters.noise_variance", noise_variance_);
            LOG_INFO("Initialized Gaussian Process with noise variance: {}", noise_variance_);
        }

        void setData(const MatrixXd &X) {
            if (X.rows() == 0 || X.cols() == 0) {
                LOG_ERROR("Attempted to set empty data!");
                throw std::invalid_argument("Input data X cannot be empty.");
            }
            X_ = X;
            updateCovariance();
            LOG_INFO("Set data with {} points of dimension {}", X_.rows(), X_.cols());
        }

        void optimizeHyperparameters(const VectorXd &y, const VectorXd &lower_bounds, const VectorXd &upper_bounds) {
            if (X_.rows() == 0 || y.size() == 0) {
                LOG_ERROR("Attempted hyperparameter optimization with no data");
                throw std::runtime_error("No data available for hyperparameter optimization.");
            }
            LOG_INFO("Starting hyperparameter optimization");

            VectorXd best_params = optimizer_.optimizeBounded(X_, y, kernel_, lower_bounds, upper_bounds);
            setParameters(best_params);
            updateCovariance();
            LOG_INFO("Hyperparameter optimization completed with parameters: {}", best_params);
        }

        [[nodiscard]] MatrixXd computeCovariance(const MatrixXd &X1, const MatrixXd &X2) const {
            return kernel_.computeGramMatrix(X1, X2);
        }

        [[nodiscard]] std::pair<VectorXd, MatrixXd> posterior(const MatrixXd &X_star) const {
            if (X_.rows() == 0) {
                LOG_ERROR("Attempted posterior computation with no data");
                throw std::runtime_error("No data set for Gaussian Process.");
            }

            MatrixXd K_star = computeCovariance(X_, X_star);
            MatrixXd K_star_star = computeCovariance(X_star, X_star);

            VectorXd mean = K_star.transpose() * alpha_;

            MatrixXd cov = K_star_star - K_star.transpose() * L_.triangularView<Eigen::Lower>().solve(K_star);

            // Ensure numerical stability of the covariance matrix
            cov = (cov + cov.transpose()) / 2.0;
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(cov);
            VectorXd eigenvalues = eigenSolver.eigenvalues().cwiseMax(0.0);
            cov = eigenSolver.eigenvectors() * eigenvalues.asDiagonal() * eigenSolver.eigenvectors().transpose();

            return {mean, cov};
        }

    protected:
        Kernel kernel_;
        double noise_variance_;
        MatrixXd X_;
        MatrixXd L_; // Cholesky decomposition of K
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency
        HyperparameterOptimizer<Kernel> optimizer_;

        void updateCovariance() {
            if (X_.rows() == 0) {
                LOG_WARN("Attempted to update covariance with no data");
                return;
            }

            MatrixXd K = computeCovariance(X_, X_);
            K.diagonal().array() += noise_variance_;

            // Ensure K is symmetric
            K = (K + K.transpose()) / 2.0;

            // Perform Cholesky decomposition with adaptive regularization
            Eigen::LLT<MatrixXd> llt;
            double lambda = 1e-9;
            do {
                llt.compute(K + lambda * MatrixXd::Identity(K.rows(), K.cols()));
                lambda *= 10;
            } while (llt.info() != Eigen::Success && lambda < 1e-3);

            if (llt.info() != Eigen::Success) {
                LOG_ERROR("Cholesky decomposition failed even with regularization");
                throw std::runtime_error(
                        "Cholesky decomposition failed. The kernel matrix may not be positive definite.");
            }
            L_ = llt.matrixL();

            LOG_DEBUG("Updated covariance matrix with regularization lambda = {}", lambda);
        }

        void setParameters(const VectorXd &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Expected 3 parameters for kernel.");
            }
            kernel_.setParameters(params(0), params(1), params(2));
            LOG_DEBUG("Updated kernel parameters: {}", params);
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_HPP

/*
#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_HPP

#include <Eigen/Dense>
#include <limits>
#include <random>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"
#include "optimization/hyperparameter_optimizer.hpp"

namespace optimization {
    template<typename Kernel = kernel::Matern52<>>
    class GaussianProcess {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        explicit GaussianProcess(const Kernel &kernel, const double noise_variance = 1e-6) :
            kernel_(kernel), noise_variance_(std::max(noise_variance, std::numeric_limits<double>::epsilon())),
            optimizer_(), rng_(std::random_device{}()) {
            noise_variance_ = config::get("optimization.gp.kernel.hyperparameters.noise_variance", noise_variance_);
            LOG_INFO("Initialized Gaussian Process with noise variance: {}", noise_variance_);
        }

        void setData(const MatrixXd &X) {
            if (X.rows() == 0 || X.cols() == 0) {
                LOG_ERROR("Attempted to set empty data!");
                throw std::invalid_argument("Input data X cannot be empty.");
            }
            X_ = X;
            updateCovariance();
            LOG_INFO("Set data with {} points of dimension {}", X_.rows(), X_.cols());
        }

        void optimizeHyperparameters(const VectorXd &y) {
            if (X_.rows() == 0 || y.size() == 0) {
                LOG_ERROR("Attempted hyperparameter optimization with no data");
                throw std::runtime_error("No data available for hyperparameter optimization.");
            }
            LOG_INFO("Starting hyperparameter optimization");

            // Perform cross-validation for robust hyperparameter optimization
            const int n_folds = std::min(5, static_cast<int>(X_.rows()));
            VectorXd best_params = crossValidateHyperparameters(X_, y, n_folds);

            setParameters(best_params);
            updateCovariance();
            LOG_INFO("Hyperparameter optimization completed with parameters: {}", best_params);
        }

        // Compute the kernel function k(X1, X2)
        [[nodiscard]] MatrixXd computeCovariance(const MatrixXd &X1, const MatrixXd &X2) const {
            return kernel_.computeGramMatrix(X1, X2);
        }

        // Compute the posterior distribution
        // Returns mean and covariance of f_star ~ N(mean, cov)
        [[nodiscard]] std::pair<VectorXd, MatrixXd> posterior(const MatrixXd &X_star) const {
            if (X_.rows() == 0) {
                LOG_ERROR("Attempted posterior computation with no data");
                throw std::runtime_error("No data set for Gaussian Process.");
            }

            // Compute K(X, X_star)
            MatrixXd K_star = computeCovariance(X_, X_star);

            // Compute K(X_star, X_star)
            MatrixXd K_star_star = computeCovariance(X_star, X_star);

            // Compute the mean: E[f_star] = K_star^T * K^(-1) * y = K_star^T * alpha
            VectorXd mean = K_star.transpose() * alpha_;

            // Compute the covariance: Var[f_star] = K_star_star - K_star^T * K^(-1) * K_star
            MatrixXd cov = K_star_star - K_star.transpose() * L_.triangularView<Eigen::Lower>().solve(K_star);

            // Ensure numerical stability of the covariance matrix
            cov = (cov + cov.transpose()) / 2.0; // Ensure symmetry
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(cov);
            VectorXd eigenvalues = eigenSolver.eigenvalues().cwiseMax(0.0); // Ensure non-negative eigenvalues
            cov = eigenSolver.eigenvectors() * eigenvalues.asDiagonal() * eigenSolver.eigenvectors().transpose();

            return {mean, cov};
        }

    protected:
        Kernel kernel_;
        double noise_variance_;
        MatrixXd X_;
        MatrixXd L_; // Cholesky decomposition of K
        VectorXd alpha_; // alpha = K^(-1) * y, precomputed for efficiency
        HyperparameterOptimizer<Kernel> optimizer_;
        mutable std::mt19937 rng_;

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

            // Perform Cholesky decomposition with adaptive regularization
            Eigen::LLT<MatrixXd> llt;
            double lambda = 1e-9;
            do {
                llt.compute(K + lambda * MatrixXd::Identity(K.rows(), K.cols()));
                lambda *= 10;
            } while (llt.info() != Eigen::Success && lambda < 1e-3);

            if (llt.info() != Eigen::Success) {
                LOG_ERROR("Cholesky decomposition failed even with regularization");
                throw std::runtime_error(
                        "Cholesky decomposition failed. The kernel matrix may not be positive definite.");
            }
            L_ = llt.matrixL();

            LOG_DEBUG("Updated covariance matrix with regularization lambda = {}", lambda);
        }

        void setParameters(const VectorXd &params) {
            if (params.size() != 3) {
                throw std::invalid_argument("Expected 3 parameters for kernel.");
            }
            kernel_.setParameters(params(0), params(1), params(2));
            LOG_DEBUG("Updated kernel parameters: {}", params);
        }

        VectorXd crossValidateHyperparameters(const MatrixXd &X, const VectorXd &y, int n_folds) {
            std::vector<VectorXd> fold_params;
            std::vector<double> fold_scores;

            // Create folds
            std::vector<int> indices(X.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng_);

            for (int fold = 0; fold < n_folds; ++fold) {
                // Split data into train and validation sets
                std::vector<int> train_indices, val_indices;
                for (int i = 0; i < X.rows(); ++i) {
                    if (i % n_folds == fold) {
                        val_indices.push_back(indices[i]);
                    } else {
                        train_indices.push_back(indices[i]);
                    }
                }

                MatrixXd X_train(train_indices.size(), X.cols());
                VectorXd y_train(train_indices.size());
                for (size_t i = 0; i < train_indices.size(); ++i) {
                    X_train.row(i) = X.row(train_indices[i]);
                    y_train(i) = y(train_indices[i]);
                }

                // Optimize hyperparameters on training set
                VectorXd fold_param = optimizer_.optimize(X_train, y_train, kernel_);
                fold_params.push_back(fold_param);

                // Compute score on validation set
                setParameters(fold_param);
                setData(X_train);
                updateCovariance();

                MatrixXd X_val(val_indices.size(), X.cols());
                VectorXd y_val(val_indices.size());
                for (size_t i = 0; i < val_indices.size(); ++i) {
                    X_val.row(i) = X.row(val_indices[i]);
                    y_val(i) = y(val_indices[i]);
                }

                auto [mean, _] = posterior(X_val);
                double mse = (mean - y_val).squaredNorm() / y_val.size();
                fold_scores.push_back(-mse); // Negative MSE as score (higher is better)
            }

            // Select best parameters
            auto best_it = std::max_element(fold_scores.begin(), fold_scores.end());
            int best_fold = std::distance(fold_scores.begin(), best_it);

            LOG_INFO("Cross-validation complete. Best fold: {}, Score: {}", best_fold, fold_scores[best_fold]);
            return fold_params[best_fold];
        }
    };
} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_HPP
*/
