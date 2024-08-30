// File: misc/target_generator.hpp

#ifndef TARGET_GENERATOR_HPP
#define TARGET_GENERATOR_HPP

#include <filesystem>
#include <random>
#include "core/vision/simulator.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"

class TargetImageGenerator {
public:
    explicit TargetImageGenerator(std::shared_ptr<core::Simulator> simulator) :
        simulator_(std::move(simulator)), radius_(0) {
        object_name_ = config::get("object.name", "obj_000001");
        output_directory_ = config::get("paths.output_directory", "target_images");
        models_directory_ = config::get("paths.models_directory", "3d_models");
        const auto image_path = config::get("paths.target_image", "./target.png");
        const cv::Mat target_image = common::io::image::readImage(image_path);
        radius_ = config::get("estimation.distance.skip", true)
                          ? config::get("estimation.distance.initial_guess", 1.5)
                          : processing::vision::DistanceEstimator().estimate(target_image);
    }

    void generateTargetImages() const {
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

    std::string getRandomTargetImagePath() const {
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

private:
    std::shared_ptr<core::Simulator> simulator_;
    std::string object_name_;
    std::filesystem::path output_directory_;
    std::filesystem::path models_directory_;
    double radius_;

    Eigen::Matrix4d generateRandomExtrinsics() const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        const auto tolerance = config::get("octree.tolerance", 0.1);
        std::uniform_real_distribution<> dis_radius(radius_ - tolerance, radius_ + tolerance);

        Eigen::Vector3d position;

        // Generate a random point on the unit sphere in the upper hemisphere
        do {
            position = Eigen::Vector3d(dis(gen), dis(gen), dis(gen));
        } while (position.squaredNorm() > 1.0 || position.z() < 0.0);

        position.normalize();

        // Scale to a random radius within [radius_ - tolerance, radius_ + tolerance]
        const double final_radius = dis_radius(gen);
        position *= final_radius;

        // Create the extrinsics matrix with the computed position
        Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();
        extrinsics.block<3, 1>(0, 3) = position;

        // Manually compute a simple rotation matrix to align the camera's view
        const Eigen::Vector3d z_axis = -position.normalized();
        Eigen::Vector3d y_axis(0, 1, 0); // Arbitrary up direction

        if (std::abs(z_axis.dot(y_axis)) > 0.999) {
            y_axis = Eigen::Vector3d(1, 0, 0); // Change up direction if z is close to y
        }

        const Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
        y_axis = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rotation;
        rotation.col(0) = x_axis;
        rotation.col(1) = y_axis;
        rotation.col(2) = z_axis;

        extrinsics.block<3, 3>(0, 0) = rotation;

        return extrinsics;
    }
};

#endif // TARGET_GENERATOR_HPP
