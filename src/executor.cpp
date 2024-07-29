// File: executor.cpp

#include "executor.hpp"

#include "optimization/gpr.hpp"
#include "optimization/gpr_model.hpp"
#include "processing/image/occlusion_detector.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "viewpoint/evaluator.hpp"
#include "viewpoint/generator.hpp"

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
    comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint estimation.");
        // Generate initial viewpoints
        viewpoint::Generator<> generator(radius_, extractor_);
        auto initial_images = generator.provision(10);
        viewpoint::Evaluator<> evaluator(target_, comparator_);
        std::vector<double> initial_scores;
        for (const auto &image: initial_images) {
            initial_scores.push_back(evaluator.evaluate(image));
        }
        LOG_INFO("Initial viewpoints generated and evaluated.");

        // Initialize GPR Model
        optimization::kernel::Matern52Kernel<double> kernel(1.0, 1.0);
        optimization::GaussianProcessRegression<optimization::kernel::Matern52Kernel<double>> gpr_model(kernel);

        Eigen::MatrixXd training_data(initial_images.size(), 3);
        Eigen::VectorXd target_values(initial_scores.size());
        for (size_t i = 0; i < initial_images.size(); ++i) {
            training_data.row(i) = initial_images[i].getViewPoint().getPosition();
            target_values(i) = initial_scores[i];
        }
        gpr_model.fit(training_data, target_values);
        LOG_INFO("GPR model initialized and trained with initial data.");

        // Predict next viewpoint and update GPR model in a loop to observe learning
        constexpr int max_iterations = 10;
        for (int iter = 0; iter < max_iterations; ++iter) {
            LOG_INFO("Iteration {}: Starting prediction and model update.", iter);
            Eigen::MatrixXd candidate_points(initial_images.size(), 3);
            for (size_t i = 0; i < initial_images.size(); ++i) {
                candidate_points.row(i) = initial_images[i].getViewPoint().getPosition();
            }

            Eigen::VectorXd expected_improvements =
                    gpr_model.getExpectedImprovement(candidate_points, target_values.minCoeff());
            Eigen::Index best_candidate_index;
            expected_improvements.maxCoeff(&best_candidate_index);

            Eigen::VectorXd next_sample = candidate_points.row(best_candidate_index);

            const auto view =
                    ViewPoint<>(next_sample.x(), next_sample.y(), next_sample.z()).toView(Eigen::Vector3d::Zero());
            const Eigen::Matrix4d extrinsics = view.getPose();
            const cv::Mat rendered_view = core::Perception::render(extrinsics);
            const double score = comparator_->compare(target_.getImage(), rendered_view);

            gpr_model.update(next_sample, score);
            LOG_INFO("Iteration {}: Model updated with new data point. Score: {}", iter, score);
        }
        LOG_INFO("Viewpoint prediction and GPR model update complete.");
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}
