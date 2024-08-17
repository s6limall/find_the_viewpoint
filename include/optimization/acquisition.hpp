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

namespace optimization {

    template<typename T = double>
    class Acquisition {
    public:
        using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using AcquisitionFunc = std::function<T(const VectorXt &, T, T)>;

        enum class Strategy { UCB, EI, PI, ADAPTIVE };

        struct Config {
            Strategy strategy;
            T beta;
            T exploration_weight;
            T exploitation_weight;
            T momentum;
            int iteration_count;

            explicit Config(const Strategy strategy = Strategy::UCB, T beta = 2.0, T exploration_weight = 1.0,
                            T exploitation_weight = 1.0, T momentum = 0.1, const int iteration_count = 0) :
                strategy(strategy), beta(beta), exploration_weight(exploration_weight),
                exploitation_weight(exploitation_weight), momentum(momentum), iteration_count(iteration_count) {}
        };

        explicit Acquisition(Config config = Config()) : config_(config), rng_(std::random_device{}()) {
            updateAcquisitionFunction();
            LOG_INFO("Acquisition function initialized with strategy: {}", strategyToString(config_.strategy));
        }

        Config getConfig() { return config_; }

        T compute(const VectorXt &x, T mean, T std_dev) const {
            LOG_DEBUG("Computing acquisition function with mean: {}, std_dev: {}", mean, std_dev);
            T result = acquisition_func_(x, mean, std_dev);
            LOG_TRACE("Acquisition function result: {}", result);
            return result;
        }

        void updateConfig(const Config &new_config) {
            config_ = new_config;
            updateAcquisitionFunction();
            LOG_INFO("Acquisition function updated with new configuration: {}", strategyToString(config_.strategy));
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
        AcquisitionFunc acquisition_func_;
        VectorXt best_point_;
        std::mt19937 rng_;
        std::optional<T> best_known_value_ = std::nullopt;

        void updateAcquisitionFunction() {
            switch (config_.strategy) {
                case Strategy::UCB:
                    acquisition_func_ = [beta = config_.beta](const VectorXt &, T mean, T std_dev) {
                        LOG_TRACE("UCB strategy selected");
                        return mean + beta * std_dev;
                    };
                    break;
                case Strategy::EI:
                    acquisition_func_ = [this](const VectorXt &, T mean, T std_dev) {
                        LOG_TRACE("EI strategy selected");
                        return computeEI(mean, std_dev);
                    };
                    break;
                case Strategy::PI:
                    acquisition_func_ = [this](const VectorXt &, T mean, T std_dev) {
                        LOG_TRACE("PI strategy selected");
                        return computePI(mean, std_dev);
                    };
                    break;
                case Strategy::ADAPTIVE:
                    acquisition_func_ = [this](const VectorXt &x, T mean, T std_dev) {
                        LOG_TRACE("ADAPTIVE strategy selected");
                        return computeAdaptive(x, mean, std_dev);
                    };
                    break;
                default: {
                    LOG_ERROR("Unknown acquisition function strategy selected");
                    throw std::invalid_argument("Unknown acquisition function strategy");
                }
            }
            LOG_DEBUG("Acquisition function strategy set to {}", strategyToString(config_.strategy));
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

        T computeAdaptive(const VectorXt &x, T mean, T std_dev) const {
            T ucb = computeUCB(mean, std_dev);
            T ei = computeEI(mean, std_dev);
            T pi = computePI(mean, std_dev);

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

        static std::string_view strategyToString(Strategy strategy) {
            switch (strategy) {
                case Strategy::UCB:
                    return "UCB";
                case Strategy::EI:
                    return "EI";
                case Strategy::PI:
                    return "PI";
                case Strategy::ADAPTIVE:
                    return "ADAPTIVE";
                default:
                    return "UNKNOWN";
            }
        }
    };

} // namespace optimization

#endif // ACQUISITION_HPP
