// File: optimization/acquisition.hpp

#ifndef ACQUISITION_HPP
#define ACQUISITION_HPP

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <optional>
#include <random>
#include <string_view>
#include "common/logging/logger.hpp"
#include "common/metrics/metrics_collector.hpp"

namespace optimization {

    template<typename T = double>
    class Acquisition {
    public:
        using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using AcquisitionFunction = std::function<T(const VectorXt &, T, T)>;

        enum class Strategy { UCB, EI, PI, ADAPTIVE, ADAPTIVE_UCB, ADAPTIVE_EI, ADAPTIVE_PI, UCB_ALT };

        struct Config {
            Strategy strategy;
            T beta;
            T exploration_weight;
            T exploitation_weight;
            T momentum;
            T exploration_radius;
            T novelty_bonus;
            int iteration_count;

            explicit Config(const Strategy strategy = Strategy::ADAPTIVE, T beta = 2.0, T exploration_weight = 1.0,
                            T exploitation_weight = 1.0, T momentum = 0.1, const int iteration_count = 0) {
                this->strategy = stringToStrategy(
                        config::get("optimization.gp.acquisition.strategy", strategyToString(strategy)));
                this->beta = config::get("optimization.gp.acquisition.beta", beta);
                this->exploration_weight =
                        config::get("optimization.gp.acquisition.exploration_weight", exploration_weight);
                this->exploitation_weight =
                        config::get("optimization.gp.acquisition.exploitation_weight", exploitation_weight);
                this->momentum = config::get("optimization.gp.acquisition.momentum", momentum);
                this->exploration_radius = config::get("optimization.gp.acquisition.exploration_radius", 1.0);
                this->novelty_bonus = config::get("optimization.gp.acquisition.novelty_bonus", 1.2);
                this->iteration_count = config::get("optimization.gp.acquisition.iterations", iteration_count);
            }
        };

        std::set<VectorXt, std::function<bool(const VectorXt &, const VectorXt &)>> explored_points_{
                [](const VectorXt &a, const VectorXt &b) { return (a - b).norm() < 1e-6; }};

        explicit Acquisition(Config config = Config()) : config_(config), rng_(std::random_device{}()) {
            updateAcquisitionFunction();
            LOG_INFO("Acquisition function initialized with strategy: {}", strategyToString(config_.strategy));
        }

        Config getConfig() { return config_; }

        T compute(const VectorXt &x, T mean, T std_dev) {
            LOG_DEBUG("Computing acquisition function with mean: {}, std_dev: {}", mean, std_dev);

            if (!best_known_value_ || mean > best_known_value_.value()) {
                LOG_DEBUG("New best value found: {}", mean);
                best_known_value_ = mean;
                best_point_ = x;
                LOG_DEBUG("New best point found: {} with value: {}", best_point_, best_known_value_.value());
            }

            T ucb = config_.exploration_weight * (mean + config_.beta * std_dev);
            T ei = config_.exploitation_weight * computeEI(mean, std_dev);
            T pi = config_.exploitation_weight * computePI(mean, std_dev);

            T distance_to_best = best_point_.size() ? (x - best_point_).norm() : 0;
            T distance_factor = std::exp(-config_.momentum * distance_to_best);

            T acquisition_value = (ucb + ei + pi) * std::pow(distance_factor, 3);

            // Add a strong penalty for exploring too far from the best known point
            if (distance_to_best > config_.exploration_radius) {
                acquisition_value *= std::exp(-std::pow(distance_to_best - config_.exploration_radius, 2) /
                                              (2 * std::pow(config_.exploration_radius, 2)));
            }

            // Encourage exploration of unexplored areas
            if (!explored_points_.contains(x)) {
                acquisition_value *= config_.novelty_bonus;
            }

            explored_points_.insert(x);

            LOG_TRACE("Acquisition function result: {}", acquisition_value);

            metrics::recordMetrics(x, {{"mean", mean}, {"std_dev", std_dev}, {"acquisition_value", acquisition_value}});

            return acquisition_value;
        }

        void updateConfig(const Config &new_config) {
            config_ = new_config;
            updateAcquisitionFunction();
            LOG_INFO("Acquisition function updated with new configuration: {}", strategyToString(config_.strategy));
        }

        void updateFromConfig() {
            Config config(stringToStrategy(config::get("optimization.gp.acquisition.strategy", "ADAPTIVE")),
                          config::get("optimization.gp.acquisition.beta", 2.0),
                          config::get("optimization.gp.acquisition.exploration_weight", 1.0),
                          config::get("optimization.gp.acquisition.exploitation_weight", 1.0),
                          config::get("optimization.gp.acquisition.momentum", 0.1));
            config_ = config;
        }

        void incrementIteration() {
            ++config_.iteration_count;
            updateExplorationFactor();
            LOG_DEBUG("Iteration count incremented to {}", config_.iteration_count);
        }

        void updateBestPoint(const VectorXt &best_point) {
            best_point_ = best_point;
            LOG_DEBUG("Best point updated: {}", best_point_);
        }

    private:
        Config config_;
        AcquisitionFunction acquisition_func_;
        VectorXt best_point_;
        std::mt19937 rng_;
        std::optional<T> best_known_value_ = std::nullopt;

        void updateAcquisitionFunction() {
            static const std::unordered_map<Strategy, AcquisitionFunction> strategy_map = {
                    {Strategy::UCB, [this](const VectorXt &, T mean, T std_dev) { return computeUCB(mean, std_dev); }},
                    {Strategy::EI, [this](const VectorXt &, T mean, T std_dev) { return computeEI(mean, std_dev); }},
                    {Strategy::PI, [this](const VectorXt &, T mean, T std_dev) { return computePI(mean, std_dev); }},
                    {Strategy::ADAPTIVE,
                     [this](const VectorXt &x, T mean, T std_dev) {
                         return computeAdaptive(x, mean, std_dev, true, true, true);
                     }},
                    {Strategy::ADAPTIVE_UCB,
                     [this](const VectorXt &x, T mean, T std_dev) {
                         return computeAdaptive(x, mean, std_dev, true, false, false);
                     }},
                    {Strategy::ADAPTIVE_EI,
                     [this](const VectorXt &x, T mean, T std_dev) {
                         return computeAdaptive(x, mean, std_dev, false, true, false);
                     }},
                    {Strategy::ADAPTIVE_PI,
                     [this](const VectorXt &x, T mean, T std_dev) {
                         return computeAdaptive(x, mean, std_dev, false, false, true);
                     }},
                    {Strategy::UCB_ALT,
                     [this](const VectorXt &x, T mean, T std_dev) { return computeOriginalUCB(x, mean, std_dev); }}};

            if (strategy_map.contains(config_.strategy)) {
                acquisition_func_ = strategy_map.at(config_.strategy);
                LOG_DEBUG("Acquisition function strategy set to {}", strategyToString(config_.strategy));
            } else {
                LOG_ERROR("Unknown acquisition function strategy selected");
                throw std::invalid_argument("Unknown acquisition function strategy");
            }
        }

        T computeEI(T mean, T std_dev) const {
            if (!best_known_value_) {
                LOG_DEBUG("No best known value set, EI returns 0");
                return 0;
            }
            T z = (mean - *best_known_value_) / std_dev;
            T result = (mean - *best_known_value_) * normalCDF(z) + std_dev * normalPDF(z);
            LOG_TRACE("EI computed with mean: {}, std_dev: {}, result: {}", mean, std_dev, result);
            return result;
        }

        T computePI(T mean, T std_dev) const {
            if (!best_known_value_) {
                LOG_DEBUG("No best known value set, PI returns 0");
                return 0;
            }
            T result = normalCDF((mean - *best_known_value_) / std_dev);
            LOG_TRACE("PI computed with mean: {}, std_dev: {}, result: {}", mean, std_dev, result);
            return result;
        }

        T computeAdaptive(const VectorXt &x, T mean, T std_dev, bool use_ucb, bool use_ei, bool use_pi) const {
            T ucb = use_ucb ? computeUCB(mean, std_dev) : 0;
            T ei = use_ei ? computeEI(mean, std_dev) : 0;
            T pi = use_pi ? computePI(mean, std_dev) : 0;

            T distance_factor = best_point_.size() ? std::exp(-config_.momentum * (x - best_point_).norm()) : 1.0;
            T exploration_factor = getExplorationFactor();

            T result = (exploration_factor * (ucb + ei) + (1.0 - exploration_factor) * pi) * distance_factor;
            LOG_TRACE("Adaptive acquisition computed with mean: {}, std_dev: {}, result: {}", mean, std_dev, result);
            return result;
        }

        T computeUCB(T mean, T std_dev) const {
            T result = mean + config_.beta * std_dev;
            LOG_TRACE("UCB computed with mean: {}, std_dev: {}, result: {}", mean, std_dev, result);
            return result;
        }

        T computeOriginalUCB(const VectorXt &x, T mean, T std_dev) const {
            T ucb = computeUCB(mean, std_dev);

            T distance_factor = best_point_.size() ? std::exp(-config_.momentum * (x - best_point_).norm()) : 1.0;
            T exploration_factor = getExplorationFactor();

            T result = exploration_factor * ucb * distance_factor;
            LOG_TRACE("Original UCB acquisition computed with mean: {}, std_dev: {}, result: {}", mean, std_dev,
                      result);
            return result;
        }


        void updateExplorationFactor() {
            T factor = std::exp(-static_cast<T>(config_.iteration_count) / 100.0);
            config_.exploration_weight = std::max(T(0.1), factor);
            config_.exploitation_weight = 1.0 - config_.exploration_weight;
            LOG_DEBUG("Exploration factor updated to {}", config_.exploration_weight);
        }

        T getExplorationFactor() const {
            return config_.exploration_weight / (config_.exploration_weight + config_.exploitation_weight);
        }

        static T normalCDF(T x) {
            T result = 0.5 * (1 + std::erf(x / std::sqrt(2)));
            LOG_TRACE("normalCDF computed for x: {}, result: {}", x, result);
            return result;
        }

        static T normalPDF(T x) {
            T result = std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);
            LOG_TRACE("normalPDF computed for x: {}, result: {}", x, result);
            return result;
        }

        static const std::unordered_map<Strategy, std::string_view> &getStrategyMap() {
            static const std::unordered_map<Strategy, std::string_view> strategy_map = {
                    {Strategy::UCB, "UCB"},
                    {Strategy::EI, "EI"},
                    {Strategy::PI, "PI"},
                    {Strategy::ADAPTIVE, "ADAPTIVE"},
                    {Strategy::ADAPTIVE_UCB, "ADAPTIVE_UCB"},
                    {Strategy::ADAPTIVE_EI, "ADAPTIVE_EI"},
                    {Strategy::ADAPTIVE_PI, "ADAPTIVE_PI"},
                    {Strategy::UCB_ALT, "UCB_ALT"}};
            return strategy_map;
        }

        static std::string_view strategyToString(Strategy strategy) {
            const auto &strategy_map = getStrategyMap();
            if (auto it = strategy_map.find(strategy); it != strategy_map.end()) {
                return it->second;
            }

            LOG_WARN("Unknown strategy encountered: {}", static_cast<int>(strategy));
            return "UNKNOWN";
        }

        static Strategy stringToStrategy(std::string_view str) {
            const auto &strategy_map = getStrategyMap();

            // Reverse lookup
            auto it = std::find_if(strategy_map.begin(), strategy_map.end(),
                                   [str](const auto &pair) { return pair.second == str; });

            if (it != strategy_map.end()) {
                return it->first;
            }

            LOG_WARN("Unknown strategy string: {}, defaulting to ADAPTIVE", str);
            return Strategy::ADAPTIVE;
        }
    };
} // namespace optimization

#endif // ACQUISITION_HPP
