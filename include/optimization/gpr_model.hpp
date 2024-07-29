// File: optimization/gpr_model.hpp

#ifndef GPR_MODEL_HPP
#define GPR_MODEL_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    template<typename KernelType, typename T = double>
    class GPRModel {
    public:
        explicit GPRModel(const KernelType &kernel, T jitter = 1e-6) noexcept;

        void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &target_values);
        std::pair<T, T> predict(const Eigen::VectorXd &query_point) const;
        void update(const Eigen::VectorXd &new_data_point, T new_target_value);
        T computeExpectedImprovement(const Eigen::VectorXd &query_point, T current_best) const;

    private:
        void computeKernelMatrix();
        [[nodiscard]] Eigen::VectorXd computeKernelVector(const Eigen::VectorXd &query_point) const;

        KernelType kernel_;
        T jitter_;
        Eigen::MatrixXd training_data_;
        Eigen::VectorXd target_values_;
        Eigen::MatrixXd kernel_matrix_;
        Eigen::MatrixXd kernel_matrix_inverse_;
        bool is_trained_;
    };

    // Definitions

    template<typename KernelType, typename T>
    GPRModel<KernelType, T>::GPRModel(const KernelType &kernel, T jitter) noexcept :
        kernel_(kernel), jitter_(jitter), is_trained_(false) {}

    template<typename KernelType, typename T>
    void GPRModel<KernelType, T>::fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &target_values) {
        training_data_ = training_data;
        target_values_ = target_values;
        computeKernelMatrix();
        is_trained_ = true;
        LOG_INFO("GPR model fitted with training data.");
    }

    template<typename KernelType, typename T>
    std::pair<T, T> GPRModel<KernelType, T>::predict(const Eigen::VectorXd &query_point) const {
        if (!is_trained_) {
            LOG_ERROR("Model has not been trained yet.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        Eigen::VectorXd k = computeKernelVector(query_point);
        T mean_prediction = k.transpose() * kernel_matrix_inverse_ * target_values_;
        T variance_prediction = kernel_.compute(query_point, query_point) - k.transpose() * kernel_matrix_inverse_ * k;
        LOG_DEBUG("Prediction: Mean = {}, Variance = {}", mean_prediction, variance_prediction);
        return {mean_prediction, variance_prediction};
    }

    template<typename KernelType, typename T>
    void GPRModel<KernelType, T>::update(const Eigen::VectorXd &new_data_point, T new_target_value) {
        training_data_.conservativeResize(training_data_.rows() + 1, training_data_.cols());
        training_data_.row(training_data_.rows() - 1) = new_data_point;

        target_values_.conservativeResize(target_values_.size() + 1);
        target_values_(target_values_.size() - 1) = new_target_value;

        computeKernelMatrix();
        LOG_INFO("GPR model updated with new data point.");
    }

    template<typename KernelType, typename T>
    T GPRModel<KernelType, T>::computeExpectedImprovement(const Eigen::VectorXd &query_point, T current_best) const {
        auto [mean, variance] = predict(query_point);
        T stddev = std::sqrt(variance);
        T Z = (mean - current_best) / stddev;
        T ei = (mean - current_best) * std::erfc(-Z / std::sqrt(2)) +
               stddev * std::exp(-0.5 * Z * Z) / std::sqrt(2 * M_PI);
        LOG_DEBUG("Expected Improvement: {}", ei);
        return ei;
    }

    template<typename KernelType, typename T>
    void GPRModel<KernelType, T>::computeKernelMatrix() {
        int num_samples = training_data_.rows();
        kernel_matrix_.resize(num_samples, num_samples);
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                kernel_matrix_(i, j) = kernel_.compute(training_data_.row(i), training_data_.row(j));
            }
        }
        kernel_matrix_ += jitter_ * Eigen::MatrixXd::Identity(num_samples, num_samples);
        kernel_matrix_inverse_ = kernel_matrix_.llt().solve(Eigen::MatrixXd::Identity(num_samples, num_samples));
        LOG_INFO("Kernel matrix and its inverse computed.");
    }

    template<typename KernelType, typename T>
    Eigen::VectorXd GPRModel<KernelType, T>::computeKernelVector(const Eigen::VectorXd &query_point) const {
        int num_samples = training_data_.rows();
        Eigen::VectorXd k(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            k(i) = kernel_.compute(training_data_.row(i), query_point);
        }
        return k;
    }

} // namespace optimization

#endif // GPR_MODEL_HPP
