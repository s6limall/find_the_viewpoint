/*
#include "optimization/cma_es_optimizer.hpp"

#include "common/logging/logger.hpp"

namespace optimization {

    CMAESOptimizer::CMAESOptimizer(int dimensions, int population_size, double lower_bound, double upper_bound) :
        dimensions_(dimensions),
        population_size_(population_size),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        mean_(Eigen::VectorXd::Zero(dimensions)),
        covariance_matrix_(Eigen::MatrixXd::Identity(dimensions, dimensions)),
        step_size_(0.5),
        random_engine_(std::random_device{}()),
        normal_distribution_(0.0, 1.0),
        max_step_size_(1.0),
        min_step_size_(1e-10),
        stagnation_threshold_(20),
        stagnation_count_(0),
        best_value_(std::numeric_limits<double>::infinity()) {
        initialize();
    }

    void CMAESOptimizer::initialize() {
        population_.resize(population_size_);
        fitness_.resize(population_size_);
    }

    void CMAESOptimizer::populate() {
        for (int i = 0; i < population_size_; ++i) {
            population_[i] = mean_ + step_size_ * sample_multivariate_normal();
            for (int j = 0; j < dimensions_; ++j) {
                population_[i](j) = std::clamp(population_[i](j), lower_bound_, upper_bound_);
            }
        }
    }

    Eigen::VectorXd CMAESOptimizer::sample_multivariate_normal() {
        Eigen::VectorXd z(dimensions_);
        for (int i = 0; i < dimensions_; ++i) {
            z(i) = normal_distribution_(random_engine_);
        }
        return covariance_matrix_.llt().matrixL() * z;
    }

    void CMAESOptimizer::evaluate_population(const std::vector<core::View> &view_space,
                                             const cv::Mat &target_image,
                                             std::function<double(const core::View &, const cv::Mat &)> evaluate) {
        for (int i = 0; i < population_size_; ++i) {
            core::View view = view_space[i]; // Assuming view_space[i] is core::View
            fitness_[i] = evaluate(view, target_image);
            if (fitness_[i] < best_value_) {
                best_value_ = fitness_[i];
                best_solution_ = population_[i];
                stagnation_count_ = 0;

                LOG_INFO("New best solution found with fitness: {}", best_value_);
            }
        }
    }

    void CMAESOptimizer::evolve() {
        Eigen::VectorXd weighted_sum = Eigen::VectorXd::Zero(dimensions_);
        double fitness_sum = 0;
        for (int i = 0; i < population_size_; ++i) {
            weighted_sum += fitness_[i] * population_[i];
            fitness_sum += fitness_[i];
        }
        mean_ = weighted_sum / fitness_sum;
    }

    void CMAESOptimizer::prune() {
        Eigen::MatrixXd cov_update = Eigen::MatrixXd::Zero(dimensions_, dimensions_);
        for (int i = 0; i < population_size_; ++i) {
            Eigen::VectorXd deviation = population_[i] - mean_;
            cov_update += deviation * deviation.transpose();
        }
        covariance_matrix_ = cov_update / population_size_;
    }

    void CMAESOptimizer::learn() {
        stagnation_count_++;
        if (stagnation_count_ > stagnation_threshold_) {
            restart();
            stagnation_count_ = 0;
        } else {
            double learning_rate = 1.0 / dimensions_;
            step_size_ = std::max(min_step_size_,
                                  step_size_ * std::exp(learning_rate * (fitness_[0] - fitness_.back())));
        }
    }

    void CMAESOptimizer::restart() {
        mean_ = best_solution_;
        covariance_matrix_ = Eigen::MatrixXd::Identity(dimensions_, dimensions_);
        step_size_ = max_step_size_ / 2;
        add_diversity();
    }

    void CMAESOptimizer::add_diversity() {
        for (int i = 0; i < dimensions_; ++i) {
            mean_(i) += normal_distribution_(random_engine_) * step_size_;
        }
    }

    CMAESOptimizer::Result CMAESOptimizer::optimize(const std::vector<core::View> &view_space,
                                                    const cv::Mat &target_image,
                                                    std::function<double(
                                                            const core::View &, const cv::Mat &)> evaluate) {
        for (int generation = 0; generation < 10000; ++generation) {
            // Fixed number of generations for simplicity
            populate();
            evaluate_population(view_space, target_image, evaluate);
            evolve();
            prune();
            learn();
        }

        Result result;
        result.best_score = best_value_;
        result.optimized_views.push_back(view_space[0]); // Convert best_solution_ to core::View
        return result;
    }

}
*/
