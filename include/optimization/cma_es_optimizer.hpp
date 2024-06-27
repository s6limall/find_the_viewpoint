// File: optimization/cma_es_optimizer.hpp

#ifndef CMA_ES_OPTIMIZER_HPP
#define CMA_ES_OPTIMIZER_HPP

#include "optimization/optimizer.hpp"
#include <Eigen/Core>
#include <random>
#include <unordered_map>
#include <functional>
#include <memory>

namespace optimization {

    struct CMAESState {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double sigma;
        int iteration;
        std::vector<Eigen::VectorXd> population;
        std::vector<std::pair<double, Eigen::VectorXd> > scored_population;

        CMAESState(int dimensions, int population_size) :
            mean(Eigen::VectorXd::Zero(dimensions)),
            covariance(Eigen::MatrixXd::Identity(dimensions, dimensions)),
            sigma(0.5),
            iteration(0),
            population(population_size, Eigen::VectorXd::Zero(dimensions)) {
        }
    };

    class CMAESOptimizer : public Optimizer {
    public:
        explicit CMAESOptimizer(int dimensions = 0, int population_size = 10, int max_iterations = 50,
                                double sigma = 0.5,
                                double tolerance = 1e-6);

        void initialize(const std::vector<core::View> &initial_views = {});

        OptimizationResult optimize(const std::vector<core::View> &initial_views,
                                    const cv::Mat &target_image,
                                    std::function<double(const core::View &,
                                                         const cv::Mat &)> evaluate_callback) override;

        bool update(const core::View &view, double score) override;

        core::View getNextView() override;

        double getCachedScore(const core::View &view) const override;

        void setCache(const core::View &view, double score) override;

    private:
        int dimensions_;
        int population_size_;
        int max_iterations_;
        double min_sigma_;
        double tolerance_;

        std::mt19937 rng_;
        std::normal_distribution<double> normal_dist_;

        std::unique_ptr<CMAESState> state_;

        std::unordered_map<std::string, double> score_cache_;

        void generatePopulation();

        core::View createViewFromVector(const Eigen::VectorXd &vec);

        std::string serializeView(const core::View &view) const;

        void adaptParameters();

        bool hasConverged(const std::vector<std::pair<double, Eigen::VectorXd> > &population, double best_score,
                          double prev_best_score) const;

        void adjustPopulationSize();

        int detectDimensions(const std::vector<core::View> &initial_views) const;
    };

}

#endif // CMA_ES_OPTIMIZER_HPP



