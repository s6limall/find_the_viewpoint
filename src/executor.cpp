#include "executor.hpp"

#include "optimization/gpr.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "viewpoint/evaluator.hpp"
#include "viewpoint/generator.hpp"
#include "viewpoint/octree.hpp"

using KernelType = optimization::kernel::Matern52Kernel<double>;

std::once_flag Executor::init_flag_;
double Executor::radius_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    const auto image_path = config::get("paths.target_image", Defaults::target_image_path);
    extractor_ = processing::image::FeatureExtractor::create<processing::image::AKAZEExtractor>();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    comparator_ = std::make_shared<processing::image::SSIMComparator>();
    // comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint estimation.");

        // 1. Generate initial samples using Fibonacci lattice sampler
        std::vector<double> lower_bounds = {0.0, 0.0, 0.0};
        std::vector<double> upper_bounds = {1.0, 1.0, 1.0};
        FibonacciLatticeSampler<double> sampler(lower_bounds, upper_bounds, radius_);

        Eigen::MatrixXd initial_samples =
                sampler.generate(config::get("sampling.count", 30)); // Generate initial samples
        for (int i = 0; i < initial_samples.cols(); ++i) {
            LOG_INFO("Sample {}: ({}, {}, {})", i, initial_samples(0, i), initial_samples(1, i), initial_samples(2, i));
        }

        // 2. Evaluate initial samples and prepare training data
        Eigen::MatrixXd X_train(initial_samples.cols(), 3);
        Eigen::VectorXd y_train(initial_samples.cols());
        std::vector<ViewPoint<double>> initial_viewpoints;


        for (size_t i = 0; i < initial_samples.cols(); ++i) {
            Eigen::Vector3d position = initial_samples.col(i);
            initial_viewpoints.emplace_back(position);
            Image<> viewpoint_image = Image<>::fromViewPoint(initial_viewpoints[i], extractor_);
            double score = comparator_->compare(target_, viewpoint_image);
            X_train.row(i) = initial_samples.col(i).transpose();
            y_train(i) = score;
            initial_viewpoints[i].setScore(score);
            LOG_INFO("Viewpoint ({}, {}, {}) - Score: {}", position(0), position(1), position(2), score);
        }

        // 3. Train GPR model with initial kernel parameters
        double initial_length_scale = 1.0;
        double initial_variance = 1.0;
        double initial_noise_variance = 1e-6;
        KernelType kernel(initial_length_scale, initial_variance, initial_noise_variance);
        optimization::GaussianProcessRegression<KernelType> gpr(kernel);
        gpr.fit(X_train, y_train);

        // 4. Create Octree
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        double size = 2 * radius_;
        double resolution = 0.1;
        viewpoint::Octree<double> octree(origin, size, resolution);

        for (const auto &viewpoint: initial_viewpoints) {
            octree.insert(viewpoint);
        }

        // 5. Iteratively refine the octree and generate new viewpoints until convergence
        const int max_iterations = 50;
        const int max_depth = 10;
        const double target_score_threshold = 0.9; // Define your target score threshold here
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (octree.checkConvergence()) {
                LOG_INFO("Convergence achieved after {} iterations.", iter);
                break;
            }

            octree.refine(max_depth, target_, comparator_, gpr);

            auto new_viewpoints = octree.sampleNewViewpoints(10, target_, comparator_, gpr);
            for (auto &viewpoint: new_viewpoints) {
                Image<> viewpoint_image = Image<>::fromViewPoint(viewpoint, extractor_);
                double score = comparator_->compare(target_, viewpoint_image);
                viewpoint.setScore(score);
                octree.insert(viewpoint);

                // Check if the target score threshold is achieved
                if (score >= target_score_threshold) {
                    LOG_INFO("Target score threshold achieved. Score: {}, Iteration: {}", score, iter);
                    return;
                }
            }
        }

        LOG_INFO("Viewpoint estimation completed.");
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}


///
/*
template<FloatingPoint T>
std::vector<ViewPoint<T>> Octree<T>::sampleNewViewpoints(
        size_t n, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
        const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) const {
    std::vector<ViewPoint<T>> new_viewpoints;
    std::priority_queue<std::pair<T, ViewPoint<T>>> pq;

    traverseTree([&](const Node &node) {
        for (const auto &point: node.points) {
            Eigen::VectorXd position = point.getPosition();
            auto [mean, variance] = gpr.predict(position);  // Explicit call to disambiguate
            T ucb = computeUCB(mean, variance, node.points.size());
            pq.emplace(ucb, point);
        }
    });

    new_viewpoints.reserve(std::min(n, pq.size()));
    for (size_t i = 0; i < n && !pq.empty(); ++i) {
        new_viewpoints.push_back(std::move(pq.top().second));
        pq.pop();
    }

    return new_viewpoints;
}

template<FloatingPoint T>
void Octree<T>::updateNodeStatistics(
        Node &node, const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
        const optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<T>> &gpr) {
    for (auto &point: node.points) {
        if (!point.hasScore()) {
            auto similarity_score = comparator->compare(target.getImage(), Image<T>::fromViewPoint(point).getImage());
            point.setScore(similarity_score);
        }
        if (!point.hasUncertainty()) {
            Eigen::VectorXd position = point.getPosition();
            auto [mean, variance] = gpr.predict(position);  // Explicit call to disambiguate
            point.setUncertainty(variance);
        }
    }
    node.similarity_gradient = computeSimilarityGradient(node, target, comparator);
}
*/
