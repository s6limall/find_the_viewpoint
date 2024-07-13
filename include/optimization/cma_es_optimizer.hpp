#ifndef CMA_ES_OPTIMIZER_HPP
#define CMA_ES_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <random>
#include <opencv2/core.hpp>
#include "core/view.hpp"  // Include the View class

namespace optimization {

    class CMAESOptimizer {
    public:
        struct Result {
            std::vector<core::View> optimized_views;
            double best_score;
        };

        CMAESOptimizer(int dimensions, int population_size, double lower_bound, double upper_bound);

        Result optimize(const std::vector<core::View> &view_space, const cv::Mat &target_image,
                        std::function<double(const core::View &, const cv::Mat &)> evaluate);

    private:
        int dimensions_;
        int population_size_;
        double lower_bound_;
        double upper_bound_;
        Eigen::VectorXd mean_;
        Eigen::MatrixXd covariance_matrix_;
        double step_size_;
        std::mt19937 random_engine_;
        std::normal_distribution<double> normal_distribution_;
        std::vector<Eigen::VectorXd> population_;
        std::vector<double> fitness_;
        Eigen::VectorXd best_solution_;
        double best_value_;
        int stagnation_threshold_;
        int stagnation_count_;
        double max_step_size_;
        double min_step_size_;

        void initialize();

        void populate();

        Eigen::VectorXd sample_multivariate_normal();

        void evaluate_population(const std::vector<core::View> &view_space, const cv::Mat &target_image,
                                 std::function<double(const core::View &, const cv::Mat &)> evaluate);

        void evolve();

        void prune();

        void learn();

        void restart();

        void add_diversity();
    };

}

#endif // CMA_ES_OPTIMIZER_HPP
