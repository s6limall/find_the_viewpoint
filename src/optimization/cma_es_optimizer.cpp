// File: optimization/cma_es_optimizer.cpp

#include "optimization/cma_es_optimizer.hpp"
#include "common/formatting/fmt_eigen.hpp"
#include "common/formatting/fmt_vector.hpp"
#include "core/view.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace optimization {
    CMAESOptimizer::CMAESOptimizer(int dimensions, int population_size, int max_iterations, double sigma,
                                   double tolerance) : dimensions_(dimensions),
                                                       population_size_(population_size),
                                                       max_iterations_(max_iterations),
                                                       min_sigma_(1e-10),
                                                       tolerance_(tolerance),
                                                       rng_(std::random_device{}()),
                                                       normal_dist_(0.0, sigma) {
        spdlog::info(
            "CMA-ES Optimizer initialized with dimensions: {}, population size: {}, max iterations: {}, sigma: {}, tolerance: {}",
            dimensions_, population_size_, max_iterations_, sigma, tolerance_);
    }

    void CMAESOptimizer::initialize(const std::vector<core::View> &initial_views) {
        if (dimensions_ == 0) {
            dimensions_ = detectDimensions(initial_views);
        }

        state_ = std::make_unique<CMAESState>(dimensions_, population_size_);
        state_->mean = Eigen::VectorXd::Zero(dimensions_);
        state_->covariance = Eigen::MatrixXd::Identity(dimensions_, dimensions_);
        state_->sigma = normal_dist_.stddev();

        spdlog::info("CMA-ES initialized with dimensions: {}, population size: {}, max iterations: {}, tolerance: {}",
                     dimensions_, population_size_, max_iterations_, tolerance_);
        spdlog::debug("Initial mean vector: {}", state_->mean.transpose());
        spdlog::debug("Initial covariance matrix:\n{}", state_->covariance);
        spdlog::debug("Initial sigma value: {}", state_->sigma);
    }

    OptimizationResult CMAESOptimizer::optimize(
        const std::vector<core::View> &initial_views,
        const cv::Mat &target_image,
        std::function<double(const core::View &, const cv::Mat &)> evaluate_callback) {
        spdlog::info("Starting CMA-ES optimization");

        if (!state_) {
            initialize(initial_views);
        }

        if (!initial_views.empty()) {
            state_->mean = Eigen::VectorXd::Zero(dimensions_);
            for (const auto &view: initial_views) {
                state_->mean += view.toVector();
            }
            state_->mean /= static_cast<double>(initial_views.size());
            spdlog::info("Initial mean set from initial views");
            spdlog::debug("Initial mean vector: {}", state_->mean.transpose());
        }

        generatePopulation();

        double best_score = std::numeric_limits<double>::max();
        double prev_best_score = best_score;
        core::View best_view;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            spdlog::info("Iteration {}", iter + 1);

            state_->scored_population.clear();
            for (const auto &individual: state_->population) {
                core::View view = createViewFromVector(individual);
                double score = getCachedScore(view);
                if (std::isnan(score)) {
                    score = evaluate_callback(view, target_image);
                    setCache(view, score);
                }

                state_->scored_population.emplace_back(score, individual);

                if (score < best_score) {
                    prev_best_score = best_score;
                    best_score = score;
                    best_view = view;
                    spdlog::info("New best score: {}", best_score);
                    spdlog::debug("Best view updated: {}", view.toVector().transpose());
                }
            }

            std::sort(state_->scored_population.begin(), state_->scored_population.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });

            if (hasConverged(state_->scored_population, best_score, prev_best_score)) {
                spdlog::info("Convergence criteria met at iteration {}", iter + 1);
                break;
            }

            Eigen::VectorXd new_mean = Eigen::VectorXd::Zero(dimensions_);
            for (int i = 0; i < population_size_ / 2; ++i) {
                new_mean += state_->scored_population[i].second;
            }
            new_mean /= static_cast<double>(population_size_ / 2);

            Eigen::MatrixXd new_covariance = Eigen::MatrixXd::Zero(dimensions_, dimensions_);
            for (int i = 0; i < population_size_ / 2; ++i) {
                Eigen::VectorXd deviation = state_->scored_population[i].second - new_mean;
                new_covariance += deviation * deviation.transpose();
            }
            new_covariance /= static_cast<double>(population_size_ / 2);
            new_covariance += Eigen::MatrixXd::Identity(dimensions_, dimensions_) * 1e-5;

            state_->mean = new_mean;
            state_->covariance = new_covariance;

            spdlog::debug("Updated mean vector: {}", state_->mean.transpose());
            spdlog::debug("Updated covariance matrix:\n{}", state_->covariance);

            adaptParameters();
            adjustPopulationSize();
            generatePopulation();
        }

        std::vector<core::View> optimized_views;
        for (const auto &individual: state_->population) {
            optimized_views.push_back(createViewFromVector(individual));
        }

        spdlog::info("CMA-ES optimization complete. Best score: {}", best_score);

        return {optimized_views, best_score, max_iterations_, true};
    }

    bool CMAESOptimizer::update(const core::View &view, double score) {
        auto serialized_view = serializeView(view);
        if (score_cache_.find(serialized_view) == score_cache_.end()) {
            score_cache_[serialized_view] = score;
            spdlog::debug("Updated score cache for view {}", serialized_view);
            return true;
        }
        spdlog::debug("Score cache already exists for view {}", serialized_view);
        return false;
    }

    core::View CMAESOptimizer::getNextView() {
        if (!state_) {
            throw std::runtime_error("Optimizer is not initialized.");
        }
        if (state_->population.empty()) {
            generatePopulation();
        }
        return createViewFromVector(state_->population.back());
    }

    double CMAESOptimizer::getCachedScore(const core::View &view) const {
        auto serialized_view = serializeView(view);
        auto it = score_cache_.find(serialized_view);
        if (it != score_cache_.end()) {
            spdlog::debug("Cached score found for view {}: {}", serialized_view, it->second);
            return it->second;
        }
        spdlog::debug("No cached score found for view {}", serialized_view);
        return std::nan("");
    }

    void CMAESOptimizer::setCache(const core::View &view, double score) {
        score_cache_[serializeView(view)] = score;
        spdlog::debug("Set score cache for view {}", serializeView(view));
    }

    void CMAESOptimizer::generatePopulation() {
        if (!state_) {
            throw std::runtime_error("Optimizer is not initialized.");
        }
        state_->population.clear();
        Eigen::LLT<Eigen::MatrixXd> llt(state_->covariance);
        if (llt.info() == Eigen::NumericalIssue) {
            throw std::runtime_error("Covariance matrix is not positive definite.");
        }

        Eigen::MatrixXd L = llt.matrixL();

        for (int i = 0; i < population_size_; ++i) {
            Eigen::VectorXd sample = state_->mean + state_->sigma * L * Eigen::VectorXd::NullaryExpr(
                                         dimensions_, [&](int) { return normal_dist_(rng_); });
            state_->population.push_back(sample);
        }
        spdlog::info("Generated new population of size {}", population_size_);
    }

    core::View CMAESOptimizer::createViewFromVector(const Eigen::VectorXd &vec) {
        core::View view;
        Eigen::Vector3f position = vec.head<3>().cast<float>();
        view.computePoseFromPositionAndObjectCenter(position, Eigen::Vector3f(0, 0, 0));
        return view;
    }

    void CMAESOptimizer::adaptParameters() {
        double sigma_decay_rate = 0.95;
        state_->sigma = std::max(state_->sigma * sigma_decay_rate, min_sigma_);
        spdlog::info("Adapted sigma to {}", state_->sigma);
    }

    bool CMAESOptimizer::hasConverged(const std::vector<std::pair<double, Eigen::VectorXd> > &population,
                                      double best_score, double prev_best_score) const {
        double mean_score = std::accumulate(population.begin(), population.end(), 0.0,
                                            [](double sum, const std::pair<double, Eigen::VectorXd> &elem) {
                                                return sum + elem.first;
                                            }) / population.size();

        double variance = std::accumulate(population.begin(), population.end(), 0.0,
                                          [&](double sum, const std::pair<double, Eigen::VectorXd> &elem) {
                                              return sum + std::pow(elem.first - mean_score, 2);
                                          }) / (population.size() - 1);

        spdlog::debug("Population mean score: {}", mean_score);
        spdlog::debug("Population score variance: {}", variance);


        bool score_improved = std::abs(best_score - prev_best_score) < tolerance_;

        spdlog::debug("Best score: {}, Previous best score: {}", best_score, prev_best_score);
        spdlog::debug("Score improvement check: {}", score_improved);

        return variance < tolerance_ || score_improved;
    }

    void CMAESOptimizer::adjustPopulationSize() {
        if (state_->sigma < min_sigma_ * 10) {
            population_size_ = std::max(population_size_ / 2, 10);
        } else if (state_->sigma > 0.1) {
            population_size_ = std::min(population_size_ * 2, 100);
        }
        spdlog::info("Adjusted population size to {}", population_size_);
    }

    std::string CMAESOptimizer::serializeView(const core::View &view) const {
        std::ostringstream oss;
        oss << view.toVector().transpose();
        return oss.str();
    }

    int CMAESOptimizer::detectDimensions(const std::vector<core::View> &initial_views) const {
        if (!initial_views.empty()) {
            return initial_views.front().toVector().size();
        }
        throw std::runtime_error("Initial views are empty. Cannot detect dimensions.");
    }
}
