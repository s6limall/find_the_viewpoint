// File: optimization/gaussian/gpr.hpp

#ifndef OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
#define OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "optimization/gaussian/gp.hpp"

namespace optimization {

    template<FloatingPoint T = double, IsKernel<T> KernelType = DefaultKernel<T>>
    class GPR : public GaussianProcess<T, KernelType> {
    public:
        using MatrixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorXd = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        using GaussianProcess<T, KernelType>::GaussianProcess;

        void fit(const MatrixXd &X, const VectorXd &y) {
            if (X.rows() != y.size()) {
                LOG_ERROR("Mismatch between number of input points and target values");
                throw std::invalid_argument("Number of input points must match number of target values.");
            }

            try {
                // Handle outliers before fitting the model
                const VectorXd cleaned_y = detectAndHandleOutliers(y);

                // Set the data for the Gaussian Process
                this->setData(X);

                // Ensure the target vector y_ is resized properly
                if (y_.size() != cleaned_y.size()) {
                    y_.resize(cleaned_y.size());
                }
                y_ = cleaned_y;

                // Update the alpha vector (precomputed K^-1 * y for efficiency)
                updateAlpha();
                LOG_INFO("Fitted GPR model with {} data points", X.rows());
            } catch (const std::exception &e) {
                LOG_ERROR("Error during fitting: {}. Attempting robust fit.", e.what());
                robustFit(X, y);
            }
        }


        [[nodiscard]] std::pair<T, T> predict(const VectorXd &x) const {
            if (this->x_data_.rows() == 0) {
                LOG_ERROR("Attempted prediction with unfitted model");
                return fallbackPredict(x);
            }

            try {
                MatrixXd X_star(1, x.size());
                X_star.row(0) = x.transpose();

                auto [mean, cov] = this->posterior(X_star);
                T variance = cov(0, 0);

                constexpr T epsilon = std::numeric_limits<T>::epsilon();
                variance = std::max(variance, epsilon);

                if (!std::isfinite(mean(0)) || !std::isfinite(variance)) {
                    LOG_WARN("Non-finite value detected in prediction. Using fallback method.");
                    return fallbackPredict(x);
                }

                LOG_DEBUG("Prediction at x: mean = {}, variance = {}", mean(0), variance);
                return {mean(0), variance};
            } catch (const std::exception &e) {
                LOG_ERROR("Error during prediction: {}. Using fallback method.", e.what());
                return fallbackPredict(x);
            }
        }

        void update(const VectorXd &new_x, const T new_y, Eigen::Index max_points = 1000) {
            if (new_x.size() != this->x_data_.cols()) {
                LOG_ERROR("Dimension mismatch in update");
                return; // Silently fail to update rather than throwing an exception
            }

            try {
                if (this->x_data_.rows() >= max_points) {
                    this->x_data_.bottomRows(max_points - 1) = this->x_data_.topRows(max_points - 1);
                    y_.tail(max_points - 1) = y_.head(max_points - 1);
                    this->x_data_.row(max_points - 1) = new_x.transpose();
                    y_(max_points - 1) = new_y;
                } else {
                    this->x_data_.conservativeResize(this->x_data_.rows() + 1, Eigen::NoChange);
                    this->x_data_.row(this->x_data_.rows() - 1) = new_x.transpose();
                    y_.conservativeResize(y_.size() + 1);
                    y_(y_.size() - 1) = new_y;
                }
                this->updateCovariance();
                updateAlpha();
                LOG_INFO("Updated model with new point. Total points: {}", this->x_data_.rows());
            } catch (const std::exception &e) {
                LOG_ERROR("Error in update: {}. Attempting robust update.", e.what());
                robustUpdate(new_x, new_y, max_points);
            }
        }

        void optimizeHyperparameters() {
            try {
                VectorXd lower_bounds(3), upper_bounds(3);
                lower_bounds << 1e-6, 1e-6, 1e-8;
                upper_bounds << 10.0, 10.0, 1.0;

                GaussianProcess<T, KernelType>::optimizeHyperparameters(y_, lower_bounds, upper_bounds);
                updateAlpha();
                LOG_INFO("Hyperparameters optimized successfully");
            } catch (const std::exception &e) {
                LOG_ERROR("Hyperparameter optimization failed: {}. Using robust optimization.", e.what());
                robustOptimizeHyperparameters();
            }
        }

        [[nodiscard]] T getMaxObservedUncertainty() const {
            if (this->x_data_.rows() == 0 || this->ldlt_.matrixL().rows() == 0) {
                LOG_ERROR("Attempted to get uncertainty with no data or covariance matrix not computed.");
                return std::numeric_limits<T>::infinity();
            }

            const auto diagonal = this->ldlt_.vectorD();
            return diagonal.maxCoeff();
        }

    private:
        VectorXd y_;

        void updateAlpha() {
            try {
                this->alpha_ = this->ldlt_.solve(y_);

                if (!this->alpha_.allFinite()) {
                    LOG_WARN("Numerical instability detected in alpha computation. Attempting regularization.");
                    regularizeAndRetry();
                }

                LOG_DEBUG("Updated alpha vector");
            } catch (const std::exception &e) {
                LOG_ERROR("Error in alpha computation: {}. Using robust fallback.", e.what());
                robustUpdateAlpha();
            }
        }

        void regularizeAndRetry() {
            T lambda = static_cast<T>(1e-6);
            constexpr T max_lambda = static_cast<T>(1.0);
            while (lambda < max_lambda) {
                try {
                    MatrixXd K = this->computeCovariance(this->x_data_, this->x_data_);
                    K.diagonal().array() += (this->noise_variance_ + lambda);
                    this->ldlt_.compute(K);
                    this->alpha_ = this->ldlt_.solve(y_);
                    if (this->alpha_.allFinite()) {
                        LOG_INFO("Regularization successful with lambda = {}", lambda);
                        return;
                    }
                } catch (const std::exception &e) {
                    LOG_WARN("Regularization attempt failed: {}", e.what());
                }
                lambda *= 10;
            }
            LOG_ERROR("Regularization failed. Using robust fallback method.");
            robustUpdateAlpha();
        }

        void robustUpdateAlpha() {
            // more stable but less accurate method TODO: replace with more robust method
            MatrixXd K = this->computeCovariance(this->x_data_, this->x_data_);
            K.diagonal().array() += this->noise_variance_;
            Eigen::JacobiSVD<MatrixXd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
            this->alpha_ = svd.solve(y_);
            LOG_WARN("Used SVD-based robust method for alpha computation");
        }

        [[nodiscard]] std::pair<T, T> fallbackPredict(const VectorXd &x) const {
            T min_dist = std::numeric_limits<T>::max();
            T nearest_y = 0;
            T mean_y = 0;

            if (this->x_data_.rows() > 0) {
                for (int i = 0; i < this->x_data_.rows(); ++i) {
                    const T dist = (this->x_data_.row(i).transpose() - x).norm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_y = y_(i);
                    }
                    mean_y += y_(i);
                }
                mean_y /= this->x_data_.rows();
            }

            T predicted_y = (this->x_data_.rows() > 0) ? (0.7 * nearest_y + 0.3 * mean_y) : 0;
            T uncertainty =
                    (this->x_data_.rows() > 0) ? getMaxObservedUncertainty() : std::numeric_limits<T>::infinity();

            LOG_WARN("Using fallback prediction method. Predicted value: {}, Uncertainty: {}", predicted_y,
                     uncertainty);
            return {predicted_y, uncertainty};
        }

        static VectorXd detectAndHandleOutliers(const VectorXd &y) {
            VectorXd sorted_y = y;
            std::sort(sorted_y.data(), sorted_y.data() + sorted_y.size());
            const size_t n = sorted_y.size();
            const T q1 = sorted_y(n / 4);
            const T q3 = sorted_y(3 * n / 4);
            const T iqr = q3 - q1;
            const T lower_bound = q1 - 1.5 * iqr;
            const T upper_bound = q3 + 1.5 * iqr;

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

        void robustFit(const MatrixXd &X, const VectorXd &y) {
            // TODO: Implement a more robust fitting method
            const VectorXd cleaned_y = detectAndHandleOutliers(y);
            this->setData(X);
            y_ = cleaned_y;

            // SVD based method for initial alpha computation
            MatrixXd K = this->computeCovariance(X, X);
            K.diagonal().array() += this->noise_variance_;
            Eigen::JacobiSVD<MatrixXd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
            this->alpha_ = svd.solve(y_);

            LOG_WARN("Used robust fitting method");
        }

        void robustUpdate(const VectorXd &new_x, const T new_y, Eigen::Index max_points) {
            // Implement a more robust update method
            if (this->x_data_.rows() >= max_points) {
                // Remove the point with the highest uncertainty
                VectorXd uncertainties = this->ldlt_.vectorD();
                Eigen::Index max_uncertainty_index;
                uncertainties.maxCoeff(&max_uncertainty_index);

                this->x_data_.row(max_uncertainty_index) = this->x_data_.row(this->x_data_.rows() - 1);
                y_(max_uncertainty_index) = y_(y_.size() - 1);

                this->x_data_.conservativeResize(this->x_data_.rows() - 1, Eigen::NoChange);
                y_.conservativeResize(y_.size() - 1);
            }

            // Add the new point
            this->x_data_.conservativeResize(this->x_data_.rows() + 1, Eigen::NoChange);
            this->x_data_.row(this->x_data_.rows() - 1) = new_x.transpose();
            y_.conservativeResize(y_.size() + 1);
            y_(y_.size() - 1) = new_y;

            // Update covariance and alpha using SVD for stability
            MatrixXd K = this->computeCovariance(this->x_data_, this->x_data_);
            K.diagonal().array() += this->noise_variance_;
            Eigen::JacobiSVD<MatrixXd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
            this->alpha_ = svd.solve(y_);

            LOG_WARN("Used robust update method");
        }

        void robustOptimizeHyperparameters() {
            // Implement a more robust hyperparameter optimization method
            const int max_attempts = 5;
            const T perturbation_factor = 0.1;

            VectorXd best_params = this->kernel_.getParameters();
            T best_likelihood = -std::numeric_limits<T>::infinity();

            for (int attempt = 0; attempt < max_attempts; ++attempt) {
                try {
                    VectorXd lower_bounds(3), upper_bounds(3);
                    lower_bounds << 1e-6, 1e-6, 1e-8;
                    upper_bounds << 10.0, 10.0, 1.0;

                    VectorXd perturbed_lower = lower_bounds * (1 - perturbation_factor * attempt);
                    VectorXd perturbed_upper = upper_bounds * (1 + perturbation_factor * attempt);

                    VectorXd params = this->optimizer_.optimizeBounded(this->x_data_, y_, this->kernel_,
                                                                       perturbed_lower, perturbed_upper);
                    this->setParameters(params);
                    T likelihood = this->computeLogLikelihood(y_);

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

            this->setParameters(best_params);
            updateAlpha();
            LOG_INFO("Robust hyperparameter optimization completed with parameters: {}", best_params);
        }
    };

} // namespace optimization

#endif // OPTIMIZATION_GAUSSIAN_PROCESS_REGRESSION_HPP
