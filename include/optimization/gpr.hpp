#ifndef OPTIMIZATION_GPR_HPP
#define OPTIMIZATION_GPR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    constexpr double M_1_SQRT2PI = 1.0 / std::sqrt(2.0 * M_PI);

    template<typename KernelType>
    class GaussianProcessRegression {
    public:
        explicit GaussianProcessRegression(const KernelType &kernel, double noise_variance = 1e-6,
                                           size_t window_size = 5, size_t max_samples = 1000);

        void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &target_values);

        template<typename T,
                 typename = std::enable_if_t<Eigen::internal::traits<T>::RowsAtCompileTime != Eigen::Dynamic ||
                                             Eigen::internal::traits<T>::ColsAtCompileTime != Eigen::Dynamic>>
        auto predict(const T &query_points) const;

        void update(const Eigen::VectorXd &new_data_point, double new_target_value);
        Eigen::VectorXd getExpectedImprovement(const Eigen::MatrixXd &candidate_points, double best_value) const;
        bool checkConvergence(double tolerance = 1e-3) const;
        double computeLogMarginalLikelihood() const;
        void optimizeHyperparameters(int n_restarts = 10);

        [[nodiscard]] bool isTrained() const { return is_trained_; }

    private:
        KernelType kernel_;
        double noise_variance_;
        size_t window_size_;
        size_t max_samples_;
        bool is_trained_;
        Eigen::MatrixXd training_data_;
        Eigen::VectorXd target_values_;
        Eigen::MatrixXd kernel_matrix_;
        Eigen::LLT<Eigen::MatrixXd> kernel_matrix_cholesky_;
        Eigen::VectorXd alpha_;
        std::deque<double> improvement_history_;
        std::mt19937 rng_;

        void updateModel();
        void computeKernelMatrix();
        Eigen::VectorXd computeKernelVector(const Eigen::VectorXd &query_point) const;
        void updateKernelMatrix(const Eigen::VectorXd &new_data_point);
        static double computeExpectedImprovement(double mean, double variance, double best_value);
        void subsampleData();
        Eigen::Vector3d sampleRandomParameters() const;
        std::pair<double, double> predictAtPoint(const Eigen::VectorXd &query_point) const;
        std::pair<Eigen::VectorXd, Eigen::VectorXd> predictAtPoints(const Eigen::MatrixXd &query_points) const;
    };

    // Implementation

    template<typename KernelType>
    GaussianProcessRegression<KernelType>::GaussianProcessRegression(const KernelType &kernel, double noise_variance,
                                                                     size_t window_size, size_t max_samples) :
        kernel_(kernel), noise_variance_(noise_variance), window_size_(window_size), max_samples_(max_samples),
        is_trained_(false), rng_(std::random_device{}()) {
        LOG_DEBUG("GaussianProcessRegression initialized with noise_variance: {}, window_size: {}, max_samples: {}",
                  noise_variance, window_size, max_samples);
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::fit(const Eigen::MatrixXd &training_data,
                                                    const Eigen::VectorXd &target_values) {
        if (training_data.rows() != target_values.size()) {
            LOG_ERROR("Number of rows in training data ({}) does not match the size of target values ({}).",
                      training_data.rows(), target_values.size());
            throw std::invalid_argument("Number of rows in training data must match the size of target values.");
        }

        training_data_ = training_data;
        target_values_ = target_values;

        LOG_DEBUG("Training data and target values set. Number of samples: {}", training_data_.rows());

        if (training_data_.rows() > max_samples_) {
            LOG_WARN("Number of training samples ({}) exceeds max_samples ({}). Subsampling data.",
                     training_data_.rows(), max_samples_);
            subsampleData();
        }

        updateModel();
        is_trained_ = true;
        LOG_INFO("GaussianProcessRegression model trained with {} samples.", training_data_.rows());
    }

    template<typename KernelType>
    template<typename T, typename>
    auto GaussianProcessRegression<KernelType>::predict(const T &query_points) const {
        if (!is_trained_) {
            LOG_ERROR("Predict called on an untrained model.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        if constexpr (T::ColsAtCompileTime == 1) {
            LOG_DEBUG("Predicting for a single query point.");
            return predictAtPoint(query_points);
        } else {
            LOG_DEBUG("Predicting for multiple query points.");
            return predictAtPoints(query_points);
        }
    }

    template<typename KernelType>
    std::pair<double, double>
    GaussianProcessRegression<KernelType>::predictAtPoint(const Eigen::VectorXd &query_point) const {
        const Eigen::VectorXd k_star = computeKernelVector(query_point);
        const Eigen::VectorXd v = kernel_matrix_cholesky_.solve(k_star);

        double mean_prediction = k_star.dot(alpha_);
        double variance_prediction = kernel_.compute(query_point, query_point) - v.squaredNorm() + noise_variance_;

        variance_prediction = std::max(0.0, variance_prediction); // Ensure non-negative variance

        LOG_DEBUG("Prediction for point: mean = {}, variance = {}", mean_prediction, variance_prediction);

        return {mean_prediction, variance_prediction};
    }

    template<typename KernelType>
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    GaussianProcessRegression<KernelType>::predictAtPoints(const Eigen::MatrixXd &query_points) const {
        const Eigen::MatrixXd K_star = kernel_.computeGramMatrix(training_data_, query_points);
        const Eigen::VectorXd mean = K_star.transpose() * alpha_;

        const Eigen::MatrixXd v = kernel_matrix_cholesky_.solve(K_star);
        Eigen::VectorXd var = kernel_.computeGramMatrix(query_points).diagonal() - v.colwise().squaredNorm();
        var = var.array().max(0.0); // Ensure non-negative variance

        LOG_DEBUG("Prediction for {} points completed.", query_points.cols());

        return {mean, var};
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::update(const Eigen::VectorXd &new_data_point, double new_target_value) {
        if (!is_trained_) {
            LOG_ERROR("Update called on an untrained model.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        training_data_.conservativeResize(training_data_.rows() + 1, Eigen::NoChange);
        training_data_.row(training_data_.rows() - 1) = new_data_point;
        target_values_.conservativeResize(target_values_.size() + 1);
        target_values_(target_values_.size() - 1) = new_target_value;

        LOG_DEBUG("Added new data point and target value. Total samples: {}", training_data_.rows());

        updateKernelMatrix(new_data_point);

        const double improvement = new_target_value - target_values_.maxCoeff();
        improvement_history_.push_back(improvement);
        if (improvement_history_.size() > window_size_) {
            improvement_history_.pop_front();
        }

        LOG_INFO("Model updated with new data point. Current improvement history size: {}",
                 improvement_history_.size());
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::getExpectedImprovement(const Eigen::MatrixXd &candidate_points,
                                                                  double best_value) const {
        const size_t num_candidates = candidate_points.rows();
        Eigen::VectorXd expected_improvements(num_candidates);

        for (size_t i = 0; i < num_candidates; ++i) {
            auto [mean, variance] = predict(candidate_points.row(i));
            expected_improvements(i) = computeExpectedImprovement(mean, variance, best_value);
        }

        LOG_DEBUG("Computed expected improvement for {} candidate points.", num_candidates);

        return expected_improvements;
    }

    template<typename KernelType>
    bool GaussianProcessRegression<KernelType>::checkConvergence(double tolerance) const {
        if (improvement_history_.size() < window_size_) {
            return false;
        }

        const double avg_improvement = std::accumulate(improvement_history_.begin(), improvement_history_.end(), 0.0) /
                                       improvement_history_.size();

        bool converged = avg_improvement < tolerance;
        LOG_INFO("Convergence check: {}", converged);

        return converged;
    }

    template<typename KernelType>
    double GaussianProcessRegression<KernelType>::computeLogMarginalLikelihood() const {
        if (!is_trained_) {
            LOG_ERROR("computeLogMarginalLikelihood called on an untrained model.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        const double n = static_cast<double>(target_values_.size());
        const double log_det_K = 2 * kernel_matrix_cholesky_.matrixL().toDenseMatrix().diagonal().array().log().sum();
        const double data_fit = target_values_.dot(alpha_);

        double lml = -0.5 * (data_fit + log_det_K + n * std::log(2 * M_PI));
        LOG_DEBUG("Computed log marginal likelihood: {}", lml);

        return lml;
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::optimizeHyperparameters(int n_restarts) {
        if (!is_trained_) {
            LOG_ERROR("optimizeHyperparameters called on an untrained model.");
            throw std::runtime_error("Model has not been trained yet.");
        }

        Eigen::Vector3d best_params = kernel_.getParameters();
        double best_lml = computeLogMarginalLikelihood();

        for (int i = 0; i < n_restarts; ++i) {
            Eigen::Vector3d params = sampleRandomParameters();
            kernel_.setParameters(params(0), params(1), params(2));
            updateModel();

            double lml = computeLogMarginalLikelihood();
            if (lml > best_lml) {
                best_lml = lml;
                best_params = params;
            }

            LOG_DEBUG("Hyperparameter optimization iteration {}: lml = {}", i, lml);
        }

        kernel_.setParameters(best_params(0), best_params(1), best_params(2));
        updateModel();
        LOG_INFO("Hyperparameters optimized. Best log marginal likelihood: {}", best_lml);
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::updateModel() {
        computeKernelMatrix();
        kernel_matrix_cholesky_ = kernel_matrix_.llt();
        if (kernel_matrix_cholesky_.info() != Eigen::Success) {
            LOG_ERROR("Cholesky decomposition failed. The kernel matrix may not be positive definite.");
            throw std::runtime_error("Cholesky decomposition failed. The kernel matrix may not be positive definite.");
        }
        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
        LOG_DEBUG("Model updated. Kernel matrix and Cholesky decomposition recomputed.");
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::computeKernelMatrix() {
        const int num_samples = training_data_.rows();
        kernel_matrix_.resize(num_samples, num_samples);

        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_samples; ++j) {
                kernel_matrix_(i, j) = kernel_.compute(training_data_.row(i), training_data_.row(j));
            }
        }

        kernel_matrix_.diagonal().array() += noise_variance_;
        LOG_DEBUG("Kernel matrix computed with noise variance added.");
    }

    template<typename KernelType>
    Eigen::VectorXd
    GaussianProcessRegression<KernelType>::computeKernelVector(const Eigen::VectorXd &query_point) const {
        const size_t num_samples = training_data_.rows();
        Eigen::VectorXd kernel_vector(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            kernel_vector(i) = kernel_.compute(training_data_.row(i), query_point);
        }

        LOG_DEBUG("Kernel vector computed for query point.");

        return kernel_vector;
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::updateKernelMatrix(const Eigen::VectorXd &new_data_point) {
        const size_t num_samples = training_data_.rows();
        kernel_matrix_.conservativeResize(num_samples, num_samples);

        for (int i = 0; i < num_samples; ++i) {
            kernel_matrix_(i, num_samples - 1) = kernel_.compute(training_data_.row(i), new_data_point);
            kernel_matrix_(num_samples - 1, i) = kernel_matrix_(i, num_samples - 1);
        }

        kernel_matrix_(num_samples - 1, num_samples - 1) += noise_variance_;

        Eigen::VectorXd k_star = computeKernelVector(new_data_point);
        Eigen::VectorXd B = kernel_matrix_cholesky_.solve(k_star);

        double s = kernel_.compute(new_data_point, new_data_point) + noise_variance_ - k_star.dot(B);

        Eigen::MatrixXd updated_L(num_samples, num_samples);
        updated_L.topLeftCorner(num_samples - 1, num_samples - 1) = kernel_matrix_cholesky_.matrixL();
        updated_L.col(num_samples - 1).head(num_samples - 1) = B;
        updated_L.row(num_samples - 1).head(num_samples - 1) = B.transpose();
        updated_L(num_samples - 1, num_samples - 1) = std::sqrt(s);

        kernel_matrix_cholesky_ = Eigen::LLT<Eigen::MatrixXd>(updated_L);
        if (kernel_matrix_cholesky_.info() != Eigen::Success) {
            LOG_ERROR("Cholesky decomposition failed during kernel matrix update.");
            throw std::runtime_error("Cholesky decomposition failed during kernel matrix update.");
        }

        alpha_ = kernel_matrix_cholesky_.solve(target_values_);
        LOG_DEBUG("Kernel matrix and Cholesky decomposition updated with new data point.");
    }

    template<typename KernelType>
    double GaussianProcessRegression<KernelType>::computeExpectedImprovement(double mean, double variance,
                                                                             double best_value) {
        if (variance < 1e-10) {
            return 0.0;
        }

        const double std_dev = std::sqrt(variance);
        const double z = (mean - best_value) / std_dev;
        const double cdf_z = 0.5 * std::erfc(-z * M_SQRT1_2);
        const double pdf_z = M_1_SQRT2PI * std::exp(-0.5 * z * z);

        double ei = (mean - best_value) * cdf_z + std_dev * pdf_z;
        LOG_DEBUG("Computed expected improvement: mean = {}, variance = {}, best_value = {}, ei = {}", mean, variance,
                  best_value, ei);

        return ei;
    }

    template<typename KernelType>
    void GaussianProcessRegression<KernelType>::subsampleData() {
        std::vector<size_t> indices(training_data_.rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        indices.resize(max_samples_);

        Eigen::MatrixXd X_subsampled(max_samples_, training_data_.cols());
        Eigen::VectorXd y_subsampled(max_samples_);

        for (size_t i = 0; i < max_samples_; ++i) {
            X_subsampled.row(i) = training_data_.row(indices[i]);
            y_subsampled(i) = target_values_(indices[i]);
        }

        training_data_ = X_subsampled;
        target_values_ = y_subsampled;
        LOG_DEBUG("Training data subsampled. New size: {}", max_samples_);
    }

    template<typename KernelType>
    Eigen::Vector3d GaussianProcessRegression<KernelType>::sampleRandomParameters() const {
        std::uniform_real_distribution<double> unif(0.1, 10.0);
        Eigen::Vector3d params(unif(rng_), unif(rng_), 1e-6 * unif(rng_));
        LOG_DEBUG("Random parameters sampled: [{}, {}, {}]", params[0], params[1], params[2]);
        return params;
    }

} // namespace optimization

#endif // OPTIMIZATION_GPR_HPP
