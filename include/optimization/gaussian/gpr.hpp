// File: optimization/gaussian/gpr.hpp

#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <limits>
#include <stdexcept>
#include "optimization/gaussian/gp.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class GPR : public GaussianProcess<Kernel> {
    public:
        using MatrixXd = Eigen::MatrixXd;
        using VectorXd = Eigen::VectorXd;

        using GaussianProcess<Kernel>::GaussianProcess;

        void fit(const MatrixXd &X, const VectorXd &y) {
            if (X.rows() != y.size()) {
                LOG_ERROR("Mismatch between number of input points and target values");
                throw std::invalid_argument("Number of input points must match number of target values.");
            }
            this->setData(X);
            y_ = y;
            updateAlpha();
            LOG_INFO("Fitted GPR model with {} data points", X.rows());
        }

        // Predict the mean and variance for a new input point
        [[nodiscard]] std::pair<double, double> predict(const VectorXd &x) const {
            if (this->X_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                throw std::runtime_error("Model not fitted.");
            }

            MatrixXd X_star(1, x.size());
            X_star.row(0) = x.transpose();

            auto [mean, cov] = this->posterior(X_star);
            double variance = cov(0, 0);

            // Ensure variance is never negative due to numerical instabilities
            constexpr double epsilon = std::numeric_limits<double>::epsilon();
            variance = std::max(variance, epsilon);

            LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean(0), variance);
            return {mean(0), variance};
        }

        // Update the model with a new data point
        void update(const VectorXd &new_x, const double new_y, Eigen::Index max_points = 1000) {
            if (new_x.size() != this->X_.cols()) {
                LOG_ERROR("Dimension mismatch in update");
                throw std::invalid_argument("New input point dimension must match existing data.");
            }

            if (this->X_.rows() >= max_points) {
                // Use sliding window approach for online learning
                this->X_.bottomRows(max_points - 1) = this->X_.topRows(max_points - 1);
                y_.tail(max_points - 1) = y_.head(max_points - 1);
                this->X_.row(max_points - 1) = new_x.transpose();
                y_(max_points - 1) = new_y;
            } else {
                // Append new point if we haven't reached max_points
                this->X_.conservativeResize(this->X_.rows() + 1, Eigen::NoChange);
                this->X_.row(this->X_.rows() - 1) = new_x.transpose();
                y_.conservativeResize(y_.size() + 1);
                y_(y_.size() - 1) = new_y;
            }
            this->updateCovariance();
            updateAlpha();
            LOG_INFO("Updated model with new point. Total points: {}", this->X_.rows());
        }

        // Optimize kernel hyperparameters based on current data
        void optimizeHyperparameters() {
            GaussianProcess<Kernel>::optimizeHyperparameters(y_);
            updateAlpha();
        }

        // Calculate the average uncertainty across all data points
        [[nodiscard]] double getAverageUncertainty() const {
            if (this->X_.rows() == 0 || this->L_.rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                throw std::runtime_error("No data or covariance matrix not computed.");
            }

            // Ensure the Cholesky factorization L is valid
            if (this->L_.rows() != this->X_.rows()) {
                LOG_ERROR("Covariance matrix size mismatch.");
                throw std::runtime_error("Covariance matrix size mismatch.");
            }

            // Sum the diagonal elements of the precomputed covariance matrix for efficiency
            const auto diagonal = this->L_.diagonal(); // L is the Cholesky decomposition of the covariance matrix
            double total_variance =
                    (diagonal.array() * diagonal.array()).sum(); // Variance is the square of the diagonal elements
            return total_variance / static_cast<double>(this->X_.rows()); // Average variance (uncertainty)
        }

        // Optimized and Error-proof: Get the maximum observed uncertainty across all data points
        [[nodiscard]] double getMaxObservedUncertainty() const {
            if (this->X_.rows() == 0 || this->L_.rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                throw std::runtime_error("No data or covariance matrix not computed.");
            }

            // Ensure the Cholesky factorization L is valid
            if (this->L_.rows() != this->X_.rows()) {
                LOG_ERROR("Covariance matrix size mismatch.");
                throw std::runtime_error("Covariance matrix size mismatch.");
            }

            // Compute the maximum uncertainty (variance) using the diagonal of L
            const auto diagonal = this->L_.diagonal(); // Variance is the square of the diagonal elements of L
            return (diagonal.array() * diagonal.array()).maxCoeff(); // Return the maximum variance (uncertainty)
        }

    private:
        VectorXd y_; // Target values

        void updateAlpha() {
            // Compute alpha = K^(-1) * y using Cholesky decomposition
            // First solve L * v = y, then L^T * alpha = v
            VectorXd v = this->L_.template triangularView<Eigen::Lower>().solve(y_);
            this->alpha_ = this->L_.template triangularView<Eigen::Lower>().adjoint().solve(v);

            // Check for numerical instabilities
            if (!this->alpha_.allFinite()) {
                LOG_ERROR("Numerical instability detected in alpha computation");
                throw std::runtime_error(
                        "Numerical instability in alpha computation. The problem may be ill-conditioned.");
            }

            LOG_DEBUG("Updated alpha vector");
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
