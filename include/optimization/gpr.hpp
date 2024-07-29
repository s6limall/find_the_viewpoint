// File: optimization/gpr.hpp

#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

namespace optimization {

    constexpr double M_1_SQRT2PI = 1.0 / std::sqrt(2.0 * M_PI);

    template<typename KernelType>
    class GaussianProcessRegression {
    public:
        explicit GaussianProcessRegression(const KernelType &kernel, double jitter = 1e-6);

        void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &target_values);
        [[nodiscard]] std::pair<double, double> predict(const Eigen::VectorXd &query_point) const;
        void update(const Eigen::VectorXd &new_data_point, double new_target_value);
        [[nodiscard]] Eigen::VectorXd getExpectedImprovement(const Eigen::MatrixXd &candidate_points,
                                                             double best_value) const;

    private:
        void computeKernelMatrix();
        [[nodiscard]] Eigen::VectorXd computeKernelVector(const Eigen::VectorXd &query_point) const;
        void updateKernelMatrix(const Eigen::VectorXd &new_data_point);
        [[nodiscard]] static double computeExpectedImprovement(double mean, double variance, double best_value);

        KernelType kernel_;
        Eigen::MatrixXd training_data_;
        Eigen::VectorXd target_values_;
        Eigen::MatrixXd kernel_matrix_;
        Eigen::LLT<Eigen::MatrixXd> kernel_matrix_cholesky_;
        Eigen::VectorXd alpha_;
        bool is_trained_;
        double jitter_;
    };

    // Implementation

    template<typename KernelType>
    GaussianProcessRegression<KernelType>::GaussianProcessRegression(const KernelType &kernel, const double jitter) :
        kernel_(kernel), is_trained_(false), jitter_(jitter) {}

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::fit(const Eigen::MatrixXd &training_data,
                                                    const Eigen::VectorXd &target_values) {
        if (training_data.rows() != target_values.size()) {
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
            throw std::runtime_error("Model has not been trained yet.");
        }

        const Eigen::VectorXd kernel_vector = computeKernelVector(query_point);
        double mean_prediction = kernel_vector.dot(alpha_);
        double variance_prediction = kernel_.compute(query_point, query_point) -
                                     kernel_vector.dot(kernel_matrix_cholesky_.solve(kernel_vector));
        variance_prediction = std::max(0.0, variance_prediction); // Ensure non-negative variance

        return {mean_prediction, variance_prediction};
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::update(const Eigen::VectorXd &new_data_point,
                                                       const double new_target_value) {
        if (!is_trained_) {
            throw std::runtime_error("Model has not been trained yet.");
        }

        training_data_.conservativeResize(training_data_.rows() + 1, training_data_.cols());
        training_data_.row(training_data_.rows() - 1) = new_data_point;
        target_values_.conservativeResize(target_values_.size() + 1);
        target_values_(target_values_.size() - 1) = new_target_value;
        updateKernelMatrix(new_data_point);
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::computeKernelMatrix() {
        const int num_samples = training_data_.rows();
        kernel_matrix_ = Eigen::MatrixXd(num_samples, num_samples);
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                kernel_matrix_(i, j) = kernel_.compute(training_data_.row(i), training_data_.row(j));
            }
        }
        kernel_matrix_.diagonal().array() += jitter_; // Add jitter for numerical stability
        kernel_matrix_cholesky_ = kernel_matrix_.llt();
        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::computeKernelVector(const Eigen::VectorXd &query_point) const {
        const int num_samples = training_data_.rows();
        Eigen::VectorXd kernel_vector(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            kernel_vector(i) = kernel_.compute(training_data_.row(i), query_point);
        }
        return kernel_vector;
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::updateKernelMatrix(const Eigen::VectorXd &new_data_point) {
        const int num_samples = training_data_.rows();
        kernel_matrix_.conservativeResize(num_samples, num_samples);
        for (int i = 0; i < num_samples; ++i) {
            kernel_matrix_(i, num_samples - 1) = kernel_.compute(training_data_.row(i), new_data_point);
            kernel_matrix_(num_samples - 1, i) = kernel_matrix_(i, num_samples - 1);
        }
        kernel_matrix_.diagonal().array() += jitter_; // Add jitter for numerical stability
        kernel_matrix_cholesky_ = kernel_matrix_.llt();
        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::getExpectedImprovement(const Eigen::MatrixXd &candidate_points,
                                                                  const double best_value) const {
        const size_t num_candidates = candidate_points.rows();
        Eigen::VectorXd expected_improvements(num_candidates);
        for (int i = 0; i < num_candidates; ++i) {
            const auto [mean, variance] = predict(candidate_points.row(i));
            expected_improvements(i) = computeExpectedImprovement(mean, variance, best_value);
        }
        return expected_improvements;
    }

    template<typename KernelType>
    double GaussianProcessRegression<KernelType>::computeExpectedImprovement(const double mean, const double variance,
                                                                             const double best_value) {
        if (variance < 1e-9) {
            return 0.0;
        }
        const double std_dev = std::sqrt(variance);
        const double z = (mean - best_value) / std_dev;
        const double cdf_z = 0.5 * std::erfc(-z * M_SQRT1_2); // Standard normal CDF
        const double pdf_z = M_1_SQRT2PI * std::exp(-0.5 * z * z); // Standard normal PDF
        return (mean - best_value) * cdf_z + std_dev * pdf_z;
    }

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
