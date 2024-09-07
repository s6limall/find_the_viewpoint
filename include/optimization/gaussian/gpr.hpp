// File: optimization/gaussian/gpr.hpp

#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <Eigen/Dense>
#include <cmath>
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

            VectorXd cleaned_y = detectAndHandleOutliers(y);

            this->setData(X);
            y_ = cleaned_y;
            updateAlpha();
            LOG_INFO("Fitted GPR model with {} data points", X.rows());
        }

        [[nodiscard]] std::pair<double, double> predict(const VectorXd &x) const {
            if (this->X_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                throw std::runtime_error("Model not fitted.");
            }

            MatrixXd X_star(1, x.size());
            X_star.row(0) = x.transpose();

            auto [mean, cov] = this->posterior(X_star);
            double variance = cov(0, 0);

            constexpr double epsilon = std::numeric_limits<double>::epsilon();
            variance = std::max(variance, epsilon);

            if (!std::isfinite(mean(0)) || !std::isfinite(variance)) {
                LOG_WARN("Non-finite value detected in prediction. Using fallback method.");
                return fallbackPredict(x);
            }

            LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean(0), variance);
            return {mean(0), variance};
        }

        void update(const VectorXd &new_x, const double new_y, Eigen::Index max_points = 1000) {
            if (new_x.size() != this->X_.cols()) {
                LOG_ERROR("Dimension mismatch in update");
                throw std::invalid_argument("New input point dimension must match existing data.");
            }

            if (this->X_.rows() >= max_points) {
                this->X_.bottomRows(max_points - 1) = this->X_.topRows(max_points - 1);
                y_.tail(max_points - 1) = y_.head(max_points - 1);
                this->X_.row(max_points - 1) = new_x.transpose();
                y_(max_points - 1) = new_y;
            } else {
                this->X_.conservativeResize(this->X_.rows() + 1, Eigen::NoChange);
                this->X_.row(this->X_.rows() - 1) = new_x.transpose();
                y_.conservativeResize(y_.size() + 1);
                y_(y_.size() - 1) = new_y;
            }
            this->updateCovariance();
            updateAlpha();
            LOG_INFO("Updated model with new point. Total points: {}", this->X_.rows());
        }

        void optimizeHyperparameters() {
            try {
                VectorXd lower_bounds(3), upper_bounds(3);
                lower_bounds << 1e-6, 1e-6, 1e-8;
                upper_bounds << 10.0, 10.0, 1.0;

                GaussianProcess<Kernel>::optimizeHyperparameters(y_, lower_bounds, upper_bounds);
                updateAlpha();
                LOG_INFO("Hyperparameters optimized successfully");
            } catch (const std::exception &e) {
                LOG_ERROR("Hyperparameter optimization failed: {}. Using current parameters.", e.what());
            }
        }

        double getMaxObservedUncertainty() const {
            if (this->X_.rows() == 0 || this->L_.rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                return std::numeric_limits<double>::infinity();
            }

            if (this->L_.rows() != this->X_.rows()) {
                LOG_ERROR("Covariance matrix size mismatch.");
                return std::numeric_limits<double>::infinity();
            }

            const auto diagonal = this->L_.diagonal();
            return (diagonal.array() * diagonal.array()).maxCoeff();
        }


    private:
        VectorXd y_;

        void updateAlpha() {
            constexpr double jitter = 1e-8;
            MatrixXd K = this->computeCovariance(this->X_, this->X_);
            K.diagonal().array() += jitter;

            Eigen::LDLT<MatrixXd> ldlt(K);
            this->alpha_ = ldlt.solve(y_);

            if (!this->alpha_.allFinite()) {
                LOG_WARN("Numerical instability detected in alpha computation. Attempting regularization.");
                regularizeAndRetry();
            }

            LOG_DEBUG("Updated alpha vector");
        }

        void regularizeAndRetry() {
            double lambda = 1e-6;
            const double max_lambda = 1.0;
            while (lambda < max_lambda) {
                try {
                    MatrixXd K = this->computeCovariance(this->X_, this->X_);
                    K.diagonal().array() += (this->noise_variance_ + lambda);
                    Eigen::LDLT<MatrixXd> ldlt(K);
                    this->alpha_ = ldlt.solve(y_);
                    if (this->alpha_.allFinite()) {
                        LOG_INFO("Regularization successful with lambda = {}", lambda);
                        return;
                    }
                } catch (const std::exception &e) {
                    LOG_WARN("Regularization attempt failed: {}", e.what());
                }
                lambda *= 10;
            }
            LOG_ERROR("Regularization failed. The problem may be ill-conditioned.");
            throw std::runtime_error("Failed to regularize the problem after multiple attempts.");
        }

        std::pair<double, double> fallbackPredict(const VectorXd &x) const {
            double min_dist = std::numeric_limits<double>::max();
            double nearest_y = 0.0;
            for (int i = 0; i < this->X_.rows(); ++i) {
                double dist = (this->X_.row(i).transpose() - x).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_y = y_(i);
                }
            }
            LOG_WARN("Using fallback prediction method. Nearest neighbor value: {}", nearest_y);
            return {nearest_y, getMaxObservedUncertainty()};
        }


        VectorXd detectAndHandleOutliers(const VectorXd &y) const {
            VectorXd sorted_y = y;
            std::sort(sorted_y.data(), sorted_y.data() + sorted_y.size());
            int n = sorted_y.size();
            double q1 = sorted_y(n / 4);
            double q3 = sorted_y(3 * n / 4);
            double iqr = q3 - q1;
            double lower_bound = q1 - 1.5 * iqr;
            double upper_bound = q3 + 1.5 * iqr;

            VectorXd cleaned_y = y;
            int outliers_count = 0;
            for (int i = 0; i < n; ++i) {
                if (y(i) < lower_bound || y(i) > upper_bound) {
                    cleaned_y(i) = std::clamp(y(i), lower_bound, upper_bound);
                    outliers_count++;
                }
            }

            LOG_INFO("Detected and handled {} outliers", outliers_count);
            return cleaned_y;
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP

/*
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

            // Detect and handle outliers
            VectorXd cleaned_y = detectAndHandleOutliers(y);

            this->setData(X);
            y_ = cleaned_y;
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

            if (std::isnan(mean(0)) || std::isnan(variance)) {
                LOG_WARN("NaN detected in prediction. Using fallback method.");
                return fallbackPredict(x);
            }

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
            try {
                GaussianProcess<Kernel>::optimizeHyperparameters(y_);
                updateAlpha();
                LOG_INFO("Hyperparameters optimized successfully");
            } catch (const std::exception &e) {
                LOG_ERROR("Hyperparameter optimization failed: {}. Using current parameters.", e.what());
            }
        }

        // Calculate the average uncertainty across all data points
        [[nodiscard]] double getAverageUncertainty() const {
            if (this->X_.rows() == 0 || this->L_.rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                throw std::runtime_error("No data or covariance matrix not computed.");
            }

            if (this->L_.rows() != this->X_.rows()) {
                LOG_ERROR("Covariance matrix size mismatch.");
                throw std::runtime_error("Covariance matrix size mismatch.");
            }

            const auto diagonal = this->L_.diagonal();
            double total_variance = (diagonal.array() * diagonal.array()).sum();
            return total_variance / static_cast<double>(this->X_.rows());
        }

        // Get the maximum observed uncertainty across all data points
        [[nodiscard]] double getMaxObservedUncertainty() const {
            if (this->X_.rows() == 0 || this->L_.rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                throw std::runtime_error("No data or covariance matrix not computed.");
            }

            if (this->L_.rows() != this->X_.rows()) {
                LOG_ERROR("Covariance matrix size mismatch.");
                throw std::runtime_error("Covariance matrix size mismatch.");
            }

            const auto diagonal = this->L_.diagonal();
            return (diagonal.array() * diagonal.array()).maxCoeff();
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
                LOG_WARN("Numerical instability detected in alpha computation. Attempting regularization.");
                regularizeAndRetry();
            }

            LOG_DEBUG("Updated alpha vector");
        }

        void regularizeAndRetry() {
            double lambda = 1e-6;
            const double max_lambda = 1.0;
            while (lambda < max_lambda) {
                try {
                    MatrixXd K = this->computeCovariance(this->X_, this->X_);
                    K.diagonal().array() += (this->noise_variance_ + lambda);
                    Eigen::LLT<MatrixXd> llt(K);
                    if (llt.info() == Eigen::Success) {
                        this->L_ = llt.matrixL();
                        VectorXd v = this->L_.template triangularView<Eigen::Lower>().solve(y_);
                        this->alpha_ = this->L_.template triangularView<Eigen::Lower>().adjoint().solve(v);
                        if (this->alpha_.allFinite()) {
                            LOG_INFO("Regularization successful with lambda = {}", lambda);
                            return;
                        }
                    }
                } catch (const std::exception &e) {
                    LOG_WARN("Regularization attempt failed: {}", e.what());
                }
                lambda *= 10;
            }
            LOG_ERROR("Regularization failed. The problem may be ill-conditioned.");
            throw std::runtime_error("Failed to regularize the problem after multiple attempts.");
        }

        std::pair<double, double> fallbackPredict(const VectorXd &x) const {
            // Simple fallback: use nearest neighbor prediction
            double min_dist = std::numeric_limits<double>::max();
            double nearest_y = 0.0;
            for (int i = 0; i < this->X_.rows(); ++i) {
                double dist = (this->X_.row(i).transpose() - x).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_y = y_(i);
                }
            }
            LOG_WARN("Using fallback prediction method. Nearest neighbor value: {}", nearest_y);
            return {nearest_y, this->getMaxObservedUncertainty()};
        }

        VectorXd detectAndHandleOutliers(const VectorXd &y) const {
            // Implement Interquartile Range (IQR) method for outlier detection
            VectorXd sorted_y = y;
            std::sort(sorted_y.data(), sorted_y.data() + sorted_y.size());
            int n = sorted_y.size();
            double q1 = sorted_y(n / 4);
            double q3 = sorted_y(3 * n / 4);
            double iqr = q3 - q1;
            double lower_bound = q1 - 1.5 * iqr;
            double upper_bound = q3 + 1.5 * iqr;

            VectorXd cleaned_y = y;
            int outliers_count = 0;
            for (int i = 0; i < n; ++i) {
                if (y(i) < lower_bound || y(i) > upper_bound) {
                    cleaned_y(i) = std::clamp(y(i), lower_bound, upper_bound);
                    outliers_count++;
                }
            }

            LOG_INFO("Detected and handled {} outliers", outliers_count);
            return cleaned_y;
        }

    public:
        // Method to get the log marginal likelihood
        double getLogMarginalLikelihood() const {
            if (this->X_.rows() == 0 || y_.size() == 0) {
                LOG_ERROR("Attempted to compute log marginal likelihood with no data");
                throw std::runtime_error("No data available for log marginal likelihood computation.");
            }

            // Compute log|K|
            double log_det_K = 2 * this->L_.diagonal().array().log().sum();

            // Compute y^T * K^-1 * y
            double quadratic_term = y_.dot(this->alpha_);

            // Compute log marginal likelihood
            double n = static_cast<double>(y_.size());
            double log_likelihood = -0.5 * quadratic_term - 0.5 * log_det_K - 0.5 * n * std::log(2 * M_PI);

            return log_likelihood;
        }

        // Method to compute the gradient of the log marginal likelihood w.r.t. hyperparameters
        VectorXd getLogMarginalLikelihoodGradient() const {
            if (this->X_.rows() == 0 || y_.size() == 0) {
                LOG_ERROR("Attempted to compute log marginal likelihood gradient with no data");
                throw std::runtime_error("No data available for log marginal likelihood gradient computation.");
            }

            int n_params = 3; // Assuming 3 hyperparameters: length_scale, variance, noise_variance
            VectorXd gradient(n_params);

            for (int i = 0; i < n_params; ++i) {
                MatrixXd K_grad = this->kernel_.computeGradientMatrix(this->X_, i);

                // Compute trace(K^-1 * dK/dθ)
                double trace_term = (this->L_.template triangularView<Eigen::Lower>().solve(K_grad)).trace();

                // Compute α^T * (dK/dθ) * α
                double quadratic_term = this->alpha_.dot(K_grad * this->alpha_);

                gradient(i) = 0.5 * (trace_term - quadratic_term);
            }

            return gradient;
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
*/
