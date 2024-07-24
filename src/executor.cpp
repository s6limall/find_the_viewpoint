// File: executor.cpp

#include "executor.hpp"
#include "core/perception.hpp"
#include "optimization/cmaes.hpp"
#include "optimization/pso.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/halton.hpp"
#include "sampling/transformer/builder.hpp"
#include "sampling/transformer/normalizer.hpp"
#include "sampling/transformer/spherical_shell.hpp"

std::once_flag Executor::init_flag_;
double Executor::radius_;
Image<> Executor::target_;
std::unique_ptr<processing::image::ImageComparator> Executor::comparator_;
std::unique_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::unique_ptr<processing::image::FeatureMatcher> Executor::matcher_;


void Executor::initialize() {
    extractor_ = FeatureExtractor::create<processing::image::AKAZEExtractor>();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    target_ = Image<>(common::io::image::readImage(Defaults::target_image_path), extractor_);
    radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    comparator_ = std::make_unique<processing::image::FeatureComparator>(std::move(extractor_), std::move(matcher_));
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {
        const ParticleSwarmOptimization::Parameters pso_params{20, 20, 0.7, 1.5, 1.5, radius_, 0.3}; // Example values
        ParticleSwarmOptimization pso(pso_params, target_, std::make_unique<processing::image::SSIMComparator>());

        const ViewPoint<> best_viewpoint = pso.optimize();
        const auto best_view = best_viewpoint.toView(Eigen::Vector3d::Zero());
        const Eigen::Matrix4d best_extrinsics = best_view.getPose();
        const cv::Mat best_rendered_view = core::Perception::render(best_extrinsics);

        LOG_INFO("Best viewpoint: {}", best_viewpoint.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw e;
    }
}

/*void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {
        core::Camera::Intrinsics intrinsics;
        intrinsics.setIntrinsics(640, 480, 0.95, 0.75);

        const auto processed_images = processing::image::Preprocessor<>::preprocess(target_.getImage());

        processing::vision::DistanceEstimator estimator;
        double distance = estimator.estimate(target_.getImage());

        std::vector<double> lower_bounds(3, 0.0), upper_bounds(3, 1.0);
        auto transformer =
                TransformerBuilder<>().add(std::make_shared<SphericalShellTransformer<>>(distance, 0.1)).build();

        HaltonSampler<> sampler(lower_bounds, upper_bounds);

        constexpr size_t num_samples = 100;
        Eigen::MatrixXd initial_viewpoints =
                sampler.generate(num_samples, [transformer](const Eigen::Matrix<double, Eigen::Dynamic, 1> &sample) {
                    return transformer->transform(sample);
                });

        std::vector<ViewPoint<>> viewpoint_objects;
        viewpoint_objects.reserve(num_samples);

        for (int i = 0; i < initial_viewpoints.rows(); ++i) {
            viewpoint_objects.emplace_back(ViewPoint<>::fromCartesian(
                    initial_viewpoints(i, 0), initial_viewpoints(i, 1), initial_viewpoints(i, 2)));
        }

        auto comparator = std::make_unique<processing::image::SSIMComparator>();
        Eigen::Vector3d object_center = Eigen::Vector3d::Zero();

        for (auto &viewpoint: viewpoint_objects) {
            auto view = viewpoint.toView(object_center);
            Eigen::Matrix4d extrinsics = view.getPose();
            cv::Mat rendered_view = core::Perception::render(extrinsics);

            double score = comparator->compare(target_.getImage(), rendered_view);
            viewpoint.setScore(score);
        }

        std::ranges::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                          [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        optimization::CMAES<>::Parameters cma_params{50, 3, 0.5, 1e-6, 100};
        optimization::CMAES<> cmaes(cma_params);

        // Initialize with the best initial samples
        std::vector<optimization::CMAES<>::Solution> initial_population;
        initial_population.reserve(viewpoint_objects.size());
        for (const auto &viewpoint: viewpoint_objects) {
            initial_population.push_back({viewpoint.getPosition(), viewpoint.getScore()});
        }
        cmaes.initialize(initial_population);

        const int max_iterations = 10;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            auto new_samples = cmaes.samplePopulation(transformer);
            std::vector<optimization::CMAES<double>::Solution> solutions;

            for (auto &sample: new_samples) {
                auto viewpoint = ViewPoint<>::fromCartesian(sample.x(0), sample.x(1), sample.x(2));
                auto view = viewpoint.toView(object_center);
                Eigen::Matrix4d extrinsics = view.getPose();
                cv::Mat rendered_view = core::Perception::render(extrinsics);

                double score = comparator->compare(target_.getImage(), rendered_view);
                sample.fitness = score;
                solutions.push_back(sample);
            }

            cmaes.updatePopulation(solutions);
            if (cmaes.terminationCriteria()) {
                break;
            }
        }

        std::ranges::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                          [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        auto best_viewpoint = viewpoint_objects.front();
        auto best_view = best_viewpoint.toView(object_center);
        Eigen::Matrix4d best_extrinsics = best_view.getPose();
        cv::Mat best_rendered_view = core::Perception::render(best_extrinsics);

        LOG_INFO("Best viewpoint: {}", best_viewpoint.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to initialize executor: {}", e.what());
        throw e;
    }
}*/


/*
void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);
    try {
        // Set up camera with default parameters
        core::Camera::Intrinsics intrinsics;
        intrinsics.setIntrinsics(640, 480, 0.95, 0.75); // Default values

        // Preprocess the target image
        const auto processed_images = processing::image::Preprocessor<>::preprocess(target_.getImage());

        processing::vision::DistanceEstimator estimator;

        // Estimate the distance to the object
        double distance = estimator.estimate(target_.getImage());

        // Initialize the sampler and transformer
        SphericalShellTransformer<> transformer(distance, 0.1);
        std::vector<double> lower_bounds(3, 0.0), upper_bounds(3, 1.0);

        HaltonSampler<> sampler(lower_bounds, upper_bounds);

        // Create a lambda function to use the transformer
        auto transformation = [&transformer](const Eigen::Matrix<double, Eigen::Dynamic, 1> &sample) {
            return transformer.transform(sample);
        };

        constexpr size_t num_samples = 100; // Increased sample count for better coverage
        Eigen::MatrixXd initial_viewpoints = sampler.generate(num_samples, transformation);

        std::vector<ViewPoint<>> viewpoint_objects;
        viewpoint_objects.reserve(num_samples); // Reserve space for efficiency

        for (int i = 0; i < initial_viewpoints.rows(); ++i) {
            viewpoint_objects.emplace_back(ViewPoint<>::fromCartesian(
                    initial_viewpoints(i, 0), initial_viewpoints(i, 1), initial_viewpoints(i, 2)));
        }


        // Comparator for evaluating views
        auto comparator = std::make_unique<processing::image::SSIMComparator>();

        // Define object center
        Eigen::Vector3d object_center = Eigen::Vector3d::Zero();

        // Evaluate initial viewpoints
        for (auto &viewpoint: viewpoint_objects) {
            auto view = viewpoint.toView(object_center);
            Eigen::Matrix4d extrinsics = view.getPose();
            cv::Mat rendered_view = core::Perception::render(extrinsics);

            double score = comparator->compare(target_.getImage(), rendered_view);
            viewpoint.setScore(score);
        }

        // Sort viewpoints by score
        std::ranges::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                          [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        // Iterative refinement
        const int max_iterations = 10;
        for (auto &viewpoint: viewpoint_objects) {
            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                auto view = viewpoint.toView(object_center);
                Eigen::Matrix4d extrinsics = view.getPose();
                cv::Mat rendered_view = core::Perception::render(extrinsics);

                double new_score = comparator->compare(target_.getImage(), rendered_view);
                if (new_score > viewpoint.getScore()) {
                    viewpoint.setScore(new_score);
                    // Update viewpoint based on optimization technique
                } else {
                    break; // Exit loop if no improvement
                }
            }
        }

        // Final sort by SSIM score
        std::ranges::sort(viewpoint_objects.begin(), viewpoint_objects.end(),
                  [](const auto &a, const auto &b) { return a.getScore() > b.getScore(); });

        // Display the best viewpoint
        auto best_viewpoint = viewpoint_objects.front();
        auto best_view = best_viewpoint.toView(object_center);
        Eigen::Matrix4d best_extrinsics = best_view.getPose();
        cv::Mat best_rendered_view = core::Perception::render(best_extrinsics);

        LOG_INFO("Best viewpoint: {}", best_viewpoint.getPosition());

        cv::imshow("Best Viewpoint", best_rendered_view);
        cv::waitKey(0);


    } catch (const std::exception &e) {
        LOG_ERROR("Failed to initialize executor: {}", e.what());
        throw e;
    }
}
*/
