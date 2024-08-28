#include "executor.hpp"

#include "api/pose_callback.hpp"
#include "api/pose_publisher.hpp"
#include "common/utilities/camera.hpp"
#include "common/utilities/visualizer.hpp"
#include "misc/target_generator.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"
#include "optimization/viewpoint_optimizer.hpp"
#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"
#include "spatial/octree.hpp"

using KernelType = optimization::kernel::Matern52<>;

std::once_flag Executor::init_flag_;
double Executor::radius_, Executor::target_score_;
Image<> Executor::target_;
std::shared_ptr<processing::image::FeatureExtractor> Executor::extractor_;
std::shared_ptr<processing::image::FeatureMatcher> Executor::matcher_;
std::shared_ptr<processing::image::ImageComparator> Executor::comparator_;

std::shared_ptr<core::Simulator> Executor::simulator_ = std::make_shared<core::Simulator>();
std::string Executor::object_name_;
std::filesystem::path Executor::output_directory_;
std::filesystem::path Executor::models_directory_;

void Executor::initialize() {
    loadExtractor();
    loadComparator();
    matcher_ = processing::image::FeatureMatcher::create<processing::image::FLANNMatcher>();

    object_name_ = config::get("object.name", "obj_000001");
    output_directory_ = config::get("paths.output_directory", "target_images");
    models_directory_ = config::get("paths.models_directory", "3d_models");

    if (config::get("estimation.distance.skip", true)) {
        radius_ = config::get("estimation.distance.initial_guess", 1.5);
    } else {
        radius_ = processing::vision::DistanceEstimator().estimate(target_.getImage());
    }

    if (config::get("target_images.generate", false)) {
        generateTargetImages();
        const auto image_path = getRandomTargetImagePath();
        target_ = Image<>(common::io::image::readImage(getRandomTargetImagePath()), extractor_);
    } else {
        const auto image_path = config::get("paths.target_image", "./target.png");
        target_ = Image<>(common::io::image::readImage(image_path), extractor_);
    }
}

void Executor::generateTargetImages() {
    const bool generate_images = config::get("target_images.generate", false);
    if (!generate_images) {
        LOG_INFO("Target image generation skipped as per configuration.");
        return;
    }

    const std::filesystem::path model_path = models_directory_ / (object_name_ + ".ply");
    const std::filesystem::path output_dir = output_directory_ / object_name_;
    std::filesystem::create_directories(output_dir);

    simulator_->loadMesh(model_path.string());

    const int num_images = config::get("target_images.num_images", 5);

    for (int i = 0; i < num_images; ++i) {
        const Eigen::Matrix4d extrinsics = generateRandomExtrinsics();
        const std::string image_path = (output_dir / ("target_" + std::to_string(i + 1) + ".png")).string();

        cv::Mat rendered_image = simulator_->render(extrinsics, image_path);

        if (!rendered_image.empty()) {
            LOG_INFO("Generated target image: {}", image_path);
        } else {
            LOG_ERROR("Failed to generate target image: {}", image_path);
        }
    }
}

std::string Executor::getRandomTargetImagePath() {
    const std::filesystem::path output_dir = output_directory_ / object_name_;
    std::vector<std::string> image_paths;

    for (const auto &entry: std::filesystem::directory_iterator(output_dir)) {
        if (entry.path().extension() == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }

    if (!image_paths.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, image_paths.size() - 1);
        return image_paths[dis(gen)];
    }

    // If no generated images found, return a default path
    return (output_directory_ / object_name_ / "target_1.png").string();
}

Eigen::Matrix4d Executor::generateRandomExtrinsics() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    const auto tolerance = config::get("octree.tolerance", 0.1); // Get tolerance from config
    std::uniform_real_distribution<> dis_radius(radius_ - tolerance, radius_ + tolerance);

    Eigen::Vector3d position;

    // Generate a random point on the unit sphere in the upper hemisphere
    do {
        position = Eigen::Vector3d(dis(gen), dis(gen), dis(gen));
    } while (position.squaredNorm() > 1.0 || position.z() < 0.0);

    position.normalize();

    // Scale to a random radius within [radius_ - tolerance, radius_ + tolerance]
    double final_radius = dis_radius(gen);
    position *= final_radius;

    // Create the extrinsics matrix with the computed position
    Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();
    extrinsics.block<3, 1>(0, 3) = position;

    // Manually compute a simple rotation matrix to align the camera's view
    Eigen::Vector3d z_axis = -position.normalized();
    Eigen::Vector3d y_axis(0, 1, 0); // Arbitrary up direction

    if (std::abs(z_axis.dot(y_axis)) > 0.999) {
        y_axis = Eigen::Vector3d(1, 0, 0); // Change up direction if z is close to y
    }

    Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
    y_axis = z_axis.cross(x_axis).normalized();

    Eigen::Matrix3d rotation;
    rotation.col(0) = x_axis;
    rotation.col(1) = y_axis;
    rotation.col(2) = z_axis;

    extrinsics.block<3, 3>(0, 0) = rotation;

    return extrinsics;
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    try {
        LOG_INFO("Starting viewpoint optimization.");

        const double size = 2 * radius_;
        auto pose_callback = std::make_shared<PoseCallback>();
        auto pose_publisher = std::make_shared<PosePublisher>(pose_callback);

        pose_callback->registerCallback([](const ViewPoint<> &viewpoint) {
            LOG_INFO("Received new best viewpoint: {}", viewpoint.toString());
        });

        const auto initial_length_scale =
                config::get("optimization.gp.kernel.matern.initial_length_scale_multiplier", 0.5) * size;
        const auto initial_variance = config::get("optimization.gp.kernel.matern.initial_variance", 1.0);
        const auto initial_noise_variance = config::get("optimization.gp.kernel.matern.initial_noise_variance", 1e-6);

        std::optional<ViewPoint<>> global_best_viewpoint;
        double global_best_score = -std::numeric_limits<double>::infinity();

        size_t restart_count = 0;
        const size_t max_restarts = config::get("optimization.max_restarts", 5);

        do {
            ++restart_count;
            LOG_INFO("Starting optimization restart {} of {}", restart_count, max_restarts);

            // Reset GPR and other components for each restart
            optimization::kernel::Matern52<> kernel(initial_length_scale, initial_variance, initial_noise_variance);
            optimization::GaussianProcessRegression<> gpr(kernel);

            FibonacciLatticeSampler<> sampler({0, 0, 0}, {1, 1, 1}, radius_);
            const int initial_sample_count = config::get("sampling.count", 20);
            Eigen::MatrixXd initial_samples = sampler.generate(initial_sample_count);

            ViewPoint<> best_initial_viewpoint;
            double best_initial_score = -std::numeric_limits<double>::infinity();
            Eigen::MatrixXd X_train(initial_sample_count, 3);
            Eigen::VectorXd y_train(initial_sample_count);

            for (int i = 0; i < initial_sample_count; ++i) {
                Eigen::Vector3d position = initial_samples.col(i);
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
            optimization::ViewpointOptimizer<> optimizer(
                Eigen::Vector3d::Zero(), size, min_size, max_iterations, gpr, radius_, tolerance);

            // Use the new optimize method
            optimizer.optimize(target_, comparator_, best_initial_viewpoint, target_score_);

            auto best_viewpoint = optimizer.getBestViewpoint();

            if (best_viewpoint) {
                LOG_INFO("Restart {}: Best viewpoint: {} - Score: {}", restart_count,
                         best_viewpoint->toString(), best_viewpoint->getScore());

                // Update global best viewpoint if necessary
                if (best_viewpoint->getScore() > global_best_score) {
                    global_best_viewpoint = best_viewpoint;
                    global_best_score = best_viewpoint->getScore();
                }
            } else {
                LOG_WARN("No viewpoint found in restart {}", restart_count);
            }

        } while (restart_count < max_restarts &&
                 (!global_best_viewpoint ||
                  (global_best_viewpoint->getScore() - target_score_) < -0.02));

        if (global_best_viewpoint) {
            LOG_INFO("Optimization completed. Best viewpoint: {} - Score: {}",
                     global_best_viewpoint->toString(), global_best_viewpoint->getScore());

            Image<> best_image = Image<>::fromViewPoint(*global_best_viewpoint, extractor_);
            common::utilities::Visualizer::diff(target_, best_image);
        } else {
            LOG_WARN("No suitable viewpoint found after {} restarts", max_restarts);
        }

    } catch (const std::exception &e) {
        LOG_ERROR("Failed to execute optimization: {}", e.what());
        throw;
    }
}

void Executor::loadExtractor() {
    const auto detector_type = config::get("image.feature.detector.type", "SIFT");
    if (detector_type == "SIFT") {
        LOG_INFO("Using SIFT feature extractor.");
        extractor_ = std::make_shared<processing::image::SIFTExtractor>();
    } else if (detector_type == "AKAZE") {
        LOG_INFO("Using AKAZE feature extractor.");
        extractor_ = std::make_shared<processing::image::AKAZEExtractor>();
    } else if (detector_type == "ORB") {
        LOG_INFO("Using ORB feature extractor.");
        extractor_ = std::make_shared<processing::image::ORBExtractor>();
    } else {
        LOG_WARN("Invalid feature extractor type, defaulting to SIFT.");
        extractor_ = std::make_shared<processing::image::SIFTExtractor>();
    }
}

void Executor::loadComparator() {
    auto comparator_type = config::get("image.comparator.type", "SSIM");
    std::ranges::transform(comparator_type.begin(), comparator_type.end(), comparator_type.begin(), ::tolower);
    const std::string target_score_key = "image.comparator." + comparator_type + ".threshold";
    target_score_ = config::get(target_score_key, 0.80);
    if (comparator_type == "SSIM") {
        LOG_INFO("Using SSIM image comparator.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    } else if (comparator_type == "FEATURE") {
        LOG_INFO("Using feature-based image comparator.");
        comparator_ = std::make_shared<processing::image::FeatureComparator>(extractor_, matcher_);
    } else if (comparator_type == "COMPOSITE") {
        LOG_INFO("Using composite image comparator.");
        comparator_ = std::make_shared<processing::image::CompositeComparator>(extractor_, matcher_);
    } else {
        LOG_WARN("Invalid image comparator type, defaulting to SSIM.");
        comparator_ = std::make_shared<processing::image::SSIMComparator>();
    }
}
