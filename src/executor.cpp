// File: executor.cpp

#include "executor.hpp"

#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"

using KernelType = optimization::kernel::Matern52<>;

Image<> Executor::target_;
std::once_flag Executor::init_flag_;
double Executor::radius_, Executor::target_score_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

void Executor::initialize() {
    LOG_INFO("Initializing executor.");
    extractor_ = processing::image::FeatureExtractor::create();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();
    std::tie(comparator_, target_score_) = processing::image::ImageComparator::create(extractor_, matcher_);

    const auto loadImage = [](const std::string &path) {
        LOG_DEBUG("Loaded image from path: {}", path);
        return Image<>(common::io::image::readImage(path), extractor_);
    };

    // simulator_->loadMesh(config::get("paths.mesh", ""));
    target_ = config::get("target_images.generate", false)
                      ? loadImage(TargetImageGenerator().getRandomTargetImagePath())
                      : loadImage(config::get("paths.target_image", "./target.png"));

    state::set("target_image", target_.getImage());
    LOG_DEBUG("Target image loaded successfully.");

    common::io::image::writeImage("target.png", target_.getImage());
    LOG_DEBUG("Target image saved successfully.");

    radius_ = config::get("estimation.distance.skip", true)
                      ? config::get("estimation.distance.initial_guess", 1.5)
                      : processing::vision::DistanceEstimator().estimate(target_.getImage());
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint optimization.");

        /*const auto mesh_path = state::get("paths.mesh", config::get("paths.mesh", "./3d_models/obj_000020.ply"));
        LOG_INFO("Loading mesh: {}", mesh_path);
        simulator_->loadMesh(mesh_path);
        LOG_INFO("Mesh loaded successfully.");*/

        const double size = 2 * radius_;
        const auto length_scale = config::get("optimization.gp.kernel.hyperparameters.length_scale", 0.5) * size;
        const auto variance = config::get("optimization.gp.kernel.hyperparameters.variance", 1.0);
        const auto noise_variance = config::get("optimization.gp.kernel.hyperparameters.noise_variance", 1e-6);

        std::optional<ViewPoint<>> global_best_viewpoint;
        double global_best_score = -std::numeric_limits<double>::infinity();

        size_t restart_count = 0;
        const size_t max_restarts = config::get("optimization.max_restarts", 5);

        do {
            ++restart_count;
            LOG_INFO("Starting optimization restart {} of {}", restart_count, max_restarts);

            // Reset GPR and other components for each restart
            optimization::kernel::Matern52<> kernel(length_scale, variance, noise_variance);
            optimization::GaussianProcessRegression<> gpr(kernel);

            FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
            const int sample_count = config::get("sampling.count", 20);
            Eigen::MatrixXd samples = sampler.generate(sample_count);

            ViewPoint<> best_initial_viewpoint;
            double best_initial_score = -std::numeric_limits<double>::infinity();
            Eigen::MatrixXd X_train(sample_count, 3);
            Eigen::VectorXd y_train(sample_count);

            for (int i = 0; i < sample_count; ++i) {
                Eigen::Vector3d position = samples.col(i);
                ViewPoint<> viewpoint(position);
                Image<> viewpoint_image = Image<>::fromViewPoint(viewpoint, extractor_);
                double score = comparator_->compare(target_, viewpoint_image);

                viewpoint.setScore(score);

                X_train.row(i) = position.transpose();
                y_train(i) = score;

                if (score > best_initial_score) {
                    best_initial_score = score;
                    best_initial_viewpoint = viewpoint;
                }

                LOG_INFO("Initial viewpoint {}: {} - Score: {}", i, viewpoint.toString(), score);
            }

            LOG_INFO("Best initial viewpoint: {} - Score: {}", best_initial_viewpoint.toString(), best_initial_score);

            gpr.fit(X_train, y_train);

            const auto min_size = config::get("octree.min_size_multiplier", 0.01) * size;
            const auto max_iterations = config::get("octree.max_iterations", 5);
            const auto tolerance = config::get("octree.tolerance", 0.1);

            // Create ViewpointOptimizer with new components
            optimization::ViewpointOptimizer<> optimizer(Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr,
                                                         radius_, tolerance);

            // Use the new optimize method
            optimizer.optimize(target_, comparator_, best_initial_viewpoint, target_score_);

            if (auto best_viewpoint = optimizer.getBestViewpoint()) {
                LOG_INFO("Restart {}: Best viewpoint: {} - Score: {}", restart_count, best_viewpoint->toString(),
                         best_viewpoint->getScore());

                // Update global best viewpoint if necessary
                if (best_viewpoint->getScore() > global_best_score) {
                    global_best_viewpoint = best_viewpoint;
                    global_best_score = best_viewpoint->getScore();
                }
            } else {
                LOG_WARN("No viewpoint found in restart {}", restart_count);
            }

        } while (restart_count < max_restarts &&
                 (!global_best_viewpoint || (global_best_viewpoint->getScore() - target_score_) < -0.05));

        if (global_best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: {} - Score: {}", global_best_viewpoint->toString(),
                     global_best_viewpoint->getScore());

            const auto object_name = config::get("object.name", "NA");
            const auto comparator_type = config::get("image.comparator.type", "NA");
            const auto output_path = fmt::format("{}_{}_diff.png", object_name, comparator_type);

            Image<> best_image = Image<>::fromViewPoint(*global_best_viewpoint, extractor_);
            common::utilities::Visualizer::diff(target_, best_image, output_path);

            // Final scoring
            const auto psnr = processing::image::PeakSNRComparator::compare(target_.getImage(), best_image.getImage());
            const auto ssim = processing::image::SSIMComparator().compare(target_.getImage(), best_image.getImage());
            const auto feature_score =
                    processing::image::FeatureComparator(extractor_, matcher_).compare(target_, best_image);
            const auto composite_score =
                    processing::image::CompositeComparator(extractor_, matcher_).compare(target_, best_image);

            LOG_INFO("Final scores - PSNR: {}, SSIM: {}, Feature: {}, Composite: {}", psnr, ssim, feature_score,
                     composite_score);

            metrics::recordMetrics(best_image.getViewPoint().value(), {
                                                                              {"psnr", psnr},
                                                                              {"ssim", ssim},
                                                                              {"feature", feature_score},
                                                                              {"composite", composite_score},
                                                                      });

        } else {
            LOG_WARN("No suitable viewpoint found after {} restarts", max_restarts);
        }

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}
