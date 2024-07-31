#include "executor.hpp"

#include "optimization/gpr.hpp"
#include "optimization/kernel/matern_52.hpp"
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
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint estimation.");
        LOG_INFO("Initialization complete.");

        // Generate initial viewpoints
        viewpoint::Generator<> generator(radius_, extractor_);
        auto initial_images = generator.provision(config::get("sampling.count", 20));
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

        // Initialize Octree
        viewpoint::Octree<> octree(Eigen::Matrix<double, 3, 1>::Zero(), radius_ * 2, radius_);
        for (const auto &image: initial_images) {
            octree.insert(image.getViewPoint());
        }
        octree.refine(3, target_, comparator_, gpr_model);

        // Tracking variables
        constexpr int max_iterations = 10;
        double convergence_threshold = 0.8;
        std::vector<double> best_scores;
        double global_best_score = -std::numeric_limits<double>::infinity();
        Eigen::Vector3d best_viewpoint;
        std::set<Eigen::Vector3d, viewpoint::EigenMatrixComparator> sampled_points;

        for (int iter = 0; iter < max_iterations; ++iter) {
            LOG_INFO("Iteration {}: Starting viewpoint sampling and model update.", iter);
            auto new_viewpoints = octree.sampleNewViewpoints(10, target_, comparator_, gpr_model);

            std::vector<Eigen::Vector3d> new_positions;
            std::vector<double> new_scores;

            for (const auto &viewpoint: new_viewpoints) {
                if (sampled_points.find(viewpoint.getPosition()) != sampled_points.end()) {
                    LOG_DEBUG("Skipping already sampled point ({}, {}, {})", viewpoint.getPosition().x(),
                              viewpoint.getPosition().y(), viewpoint.getPosition().z());
                    continue;
                }
                sampled_points.insert(viewpoint.getPosition());

                const auto view = viewpoint.toView(Eigen::Vector3d::Zero());
                const Eigen::Matrix4d extrinsics = view.getPose();
                const cv::Mat rendered_view = core::Perception::render(extrinsics);
                const double score = comparator_->compare(target_.getImage(), rendered_view);

                new_positions.push_back(viewpoint.getPosition());
                new_scores.push_back(score);

                gpr_model.update(viewpoint.getPosition(), score);
                LOG_DEBUG("Iteration {}: GPR model updated with new data point ({}, {}, {}). Score: {}", iter,
                          viewpoint.getPosition().x(), viewpoint.getPosition().y(), viewpoint.getPosition().z(), score);
            }

            // Update Octree with new viewpoints
            for (const auto &position: new_positions) {
                ViewPoint<> new_viewpoint(position.x(), position.y(), position.z());
                octree.insert(new_viewpoint);
            }

            // Refine Octree based on new data
            octree.refine(3, target_, comparator_, gpr_model);

            // Track best score
            double iteration_best_score = *std::max_element(new_scores.begin(), new_scores.end());
            if (iteration_best_score > global_best_score) {
                global_best_score = iteration_best_score;
                best_viewpoint = new_positions[std::distance(new_scores.begin(),
                                                             std::max_element(new_scores.begin(), new_scores.end()))];
            }
            best_scores.push_back(iteration_best_score);

            // Check for convergence
            if (octree.checkConvergence()) {
                LOG_INFO("Convergence achieved at iteration {}. Best score: {}", iter, global_best_score);
                break;
            }

            // Log progress
            LOG_INFO("Iteration {}: Best score: {}", iter, iteration_best_score);

            // Adaptive refinement: Focus on the region around the best viewpoint
            if (global_best_score > convergence_threshold) {
                octree = viewpoint::Octree<>(best_viewpoint, radius_ / std::pow(2, iter),
                                             radius_ / std::pow(2, iter + 1));
                for (const auto &image: initial_images) {
                    octree.insert(image.getViewPoint());
                }
                octree.refine(3, target_, comparator_, gpr_model);
            }
        }

        // Display final results
        LOG_INFO("Viewpoint prediction and GPR model update complete.");
        LOG_INFO("Global best score: {}", global_best_score);
        LOG_INFO("Best viewpoint: ({}, {}, {})", best_viewpoint.x(), best_viewpoint.y(), best_viewpoint.z());

        auto final_viewpoint = ViewPoint<>::fromPosition(best_viewpoint);
        core::Perception::render(final_viewpoint.toView().getPose());

        auto comparator_2 = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
        auto final_score =
                comparator_2->compare(target_.getImage(), Image<>::fromViewPoint(final_viewpoint).getImage());

        LOG_INFO("Final score (using SSIM): {}", global_best_score);
        LOG_INFO("Final score (using feature matching: {}", final_score);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}
