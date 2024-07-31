// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace optimization {

    constexpr double M_1_SQRT2PI = 1.0 / std::sqrt(2.0 * M_PI);

    template<typename KernelType>
    class GaussianProcessRegression {
    public:
        explicit GaussianProcessRegression(const KernelType &kernel, double noise_variance = 1e-6,
                                           size_t window_size = 5);

        void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &target_values);
        std::pair<double, double> predict(const Eigen::VectorXd &query_point) const;
        void update(const Eigen::VectorXd &new_data_point, double new_target_value);
        Eigen::VectorXd getExpectedImprovement(const Eigen::MatrixXd &candidate_points, double best_value) const;
        bool checkConvergence(double tolerance = 1e-3) const;
        double computeLogMarginalLikelihood() const;

    private:
        void computeKernelMatrix();
        Eigen::VectorXd computeKernelVector(const Eigen::VectorXd &query_point) const;
        void updateKernelMatrix(const Eigen::VectorXd &new_data_point);
        static double computeExpectedImprovement(double mean, double variance, double best_value);

        KernelType kernel_;
        double noise_variance_;
        size_t window_size_;
        bool is_trained_;
        Eigen::MatrixXd training_data_;
        Eigen::VectorXd target_values_;
        Eigen::MatrixXd kernel_matrix_;
        Eigen::LLT<Eigen::MatrixXd> kernel_matrix_cholesky_;
        Eigen::VectorXd alpha_;
        std::deque<double> improvement_history_;
    };

    // Implementation

    template<typename KernelType>
    GaussianProcessRegression<KernelType>::GaussianProcessRegression(const KernelType &kernel, double noise_variance,
                                                                     size_t window_size) :
        kernel_(kernel), noise_variance_(noise_variance), window_size_(window_size), is_trained_(false) {}

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::fit(const Eigen::MatrixXd &training_data,
                                                    const Eigen::VectorXd &target_values) {
        if (training_data.rows() != target_values.size()) {
            LOG_ERROR("Number of rows in training data must match the size of target values.");
            throw std::invalid_argument("Number of rows in training data must match the size of target values.");
        }
        training_data_ = training_data;
        target_values_ = target_values;
        computeKernelMatrix();
        is_trained_ = true;
    }

    template<typename KernelType>
    std::pair<double, double> GaussianProcessRegression<KernelType>::predict(const Eigen::VectorXd &query_point) const {
        if (!is_trained_) {
            LOG_ERROR("Model has not been trained yet.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        const Eigen::VectorXd k_star = computeKernelVector(query_point);
        const Eigen::VectorXd v = kernel_matrix_cholesky_.matrixL().solve(k_star);

        double mean_prediction = k_star.dot(alpha_);
        double variance_prediction = kernel_.compute(query_point, query_point) - v.squaredNorm() + noise_variance_;

        variance_prediction = std::max(0.0, variance_prediction); // Ensure non-negative variance

        return {mean_prediction, variance_prediction};
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::update(const Eigen::VectorXd &new_data_point, double new_target_value) {
        if (!is_trained_) {
            LOG_ERROR("Model has not been trained yet.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        training_data_.conservativeResize(training_data_.rows() + 1, Eigen::NoChange);
        training_data_.row(training_data_.rows() - 1) = new_data_point;
        target_values_.conservativeResize(target_values_.size() + 1);
        target_values_(target_values_.size() - 1) = new_target_value;

        updateKernelMatrix(new_data_point);

        const double improvement = new_target_value - target_values_.maxCoeff();
        improvement_history_.push_back(improvement);
        if (improvement_history_.size() > window_size_) {
            improvement_history_.pop_front();
        }
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::getExpectedImprovement(const Eigen::MatrixXd &candidate_points,
                                                                  double best_value) const {
        const size_t num_candidates = candidate_points.rows();
        Eigen::VectorXd expected_improvements(num_candidates);

        // #pragma omp parallel for
        for (size_t i = 0; i < num_candidates; ++i) {
            auto [mean, variance] = predict(candidate_points.row(i));
            expected_improvements(i) = computeExpectedImprovement(mean, variance, best_value);
        }

        return expected_improvements;
    }

    template<typename KernelType>
    bool GaussianProcessRegression<KernelType>::checkConvergence(double tolerance) const {
        if (improvement_history_.size() < window_size_) {
            return false;
        }

        const double avg_improvement = std::accumulate(improvement_history_.begin(), improvement_history_.end(), 0.0) /
                                       improvement_history_.size();

        return avg_improvement < tolerance;
    }

    template<typename KernelType>
    double GaussianProcessRegression<KernelType>::computeLogMarginalLikelihood() const {
        if (!is_trained_) {
            LOG_ERROR("Model has not been trained yet.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        const double n = static_cast<double>(target_values_.size());
        const double log_det_K = 2 * kernel_matrix_cholesky_.matrixL().toDenseMatrix().diagonal().array().log().sum();
        const double data_fit = target_values_.dot(alpha_);

        return -0.5 * (data_fit + log_det_K + n * std::log(2 * M_PI));
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::computeKernelMatrix() {
        const int num_samples = training_data_.rows();
        kernel_matrix_.resize(num_samples, num_samples);

        // #pragma omp parallel for collapse(2)
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                kernel_matrix_(i, j) = kernel_.compute(training_data_.row(i), training_data_.row(j));
            }
        }

        kernel_matrix_.diagonal().array() += noise_variance_;

        kernel_matrix_cholesky_ = kernel_matrix_.llt();
        if (kernel_matrix_cholesky_.info() != Eigen::Success) {
            throw std::runtime_error("Cholesky decomposition failed. The kernel matrix may not be positive definite.");
        }

        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::computeKernelVector(const Eigen::VectorXd &query_point) const {
        const size_t num_samples = training_data_.rows();
        Eigen::VectorXd kernel_vector(num_samples);

        // #pragma omp parallel for
        for (int i = 0; i < num_samples; ++i) {
            kernel_vector(i) = kernel_.compute(training_data_.row(i), query_point);
        }

        return kernel_vector;
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::updateKernelMatrix(const Eigen::VectorXd &new_data_point) {
        const size_t num_samples = training_data_.rows();
        kernel_matrix_.conservativeResize(num_samples, num_samples);

#pragma omp parallel for
        for (int i = 0; i < num_samples; ++i) {
            kernel_matrix_(i, num_samples - 1) = kernel_.compute(training_data_.row(i), new_data_point);
            kernel_matrix_(num_samples - 1, i) = kernel_matrix_(i, num_samples - 1);
        }

        kernel_matrix_(num_samples - 1, num_samples - 1) += noise_variance_;

        kernel_matrix_cholesky_ = kernel_matrix_.llt();
        if (kernel_matrix_cholesky_.info() != Eigen::Success) {
            LOG_ERROR("Cholesky decomposition failed during update. The kernel matrix may not be positive definite.");
            throw std::runtime_error(
                    "Cholesky decomposition failed during update. The kernel matrix may not be positive definite.");
        }

        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
    }

    template<typename KernelType>
    double GaussianProcessRegression<KernelType>::computeExpectedImprovement(double mean, double variance,
                                                                             double best_value) {
        if (variance < 1e-10) {
            LOG_WARN("Variance = {} is close to zero. Expected improvement may not be reliable.", variance);
            return 0.0;
        }

        const double std_dev = std::sqrt(variance);
        const double z = (mean - best_value) / std_dev;
        const double cdf_z = 0.5 * std::erfc(-z * M_SQRT1_2);
        const double pdf_z = M_1_SQRT2PI * std::exp(-0.5 * z * z);

        return (mean - best_value) * cdf_z + std_dev * pdf_z;
    }

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
