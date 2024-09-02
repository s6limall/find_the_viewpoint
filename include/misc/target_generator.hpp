// File: misc/target_generator.hpp

#ifndef TARGET_GENERATOR_HPP
#define TARGET_GENERATOR_HPP

#include <filesystem>
#include <random>
#include "core/vision/simulator.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"

class TargetImageGenerator {
public:
    explicit TargetImageGenerator() : radius_(0) {
        object_name_ = config::get("object.name", "obj_000001");
        output_directory_ = config::get("paths.output_directory", "target_images");
        models_directory_ = config::get("paths.models_directory", "3d_models");
        generateTargetImages();

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

        // simulator_->loadMesh(model_path.string());

        const int num_images = config::get("target_images.num_images", 5);

        for (int i = 0; i < num_images; ++i) {
            const Eigen::Matrix4d extrinsics = generateRandomExtrinsics();
            const std::string image_path = (output_dir / ("target_" + std::to_string(i + 1) + ".png")).string();

            cv::Mat rendered_image = core::Eye::render(extrinsics, image_path);

            if (!rendered_image.empty()) {
                LOG_INFO("Generated target image: {}", image_path);
            } else {
                LOG_ERROR("Failed to generate target image: {}", image_path);
            }
        }
    }

    [[nodiscard]] std::string getRandomTargetImagePath() const {
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
    // std::shared_ptr<core::Simulator> simulator_;
    std::string object_name_;
    std::filesystem::path output_directory_;
    std::filesystem::path models_directory_;
    double radius_;

    [[nodiscard]] static Eigen::Matrix4d generateRandomExtrinsics(const double radius = 2.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_azimuth(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<> dis_elevation(0.0, M_PI / 2.0); // Upper hemisphere only

        // Generate spherical coordinates
        const double azimuth = dis_azimuth(gen);
        const double elevation = dis_elevation(gen);

        // Convert spherical coordinates to Cartesian coordinates
        Eigen::Vector3d position;
        position.x() = std::sin(elevation) * std::cos(azimuth);
        position.y() = std::sin(elevation) * std::sin(azimuth);
        position.z() = std::cos(elevation);

        // Scale by the given radius
        position *= radius;

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
