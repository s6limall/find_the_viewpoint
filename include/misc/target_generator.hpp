// File: misc/target_generator.hpp

#ifndef TARGET_GENERATOR_HPP
#define TARGET_GENERATOR_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <stdexcept>
#include "core/vision/simulator.hpp"
#include "config/configuration.hpp"
#include "common/io/image.hpp"
#include "sampling/sampler/fibonacci.hpp"

class TargetImageGenerator {
public:
    TargetImageGenerator():simulator_(std::make_shared<core::Simulator>(std::make_optional(getModelPath().string()))) {
        loadConfiguration();
    }

    [[nodiscard]] std::optional<std::string> getOrGenerateTargetImage() const {
        const auto target_path = getOutputDirectory() / config_.target_filename;

        if (std::filesystem::exists(target_path)) {
            LOG_INFO("Target image found: {}", target_path.string());
            return target_path.string();
        }

        if (!config_.generate_images) {
            LOG_WARN("Target image not found and generation is disabled: {}", target_path.string());
            return std::nullopt;
        }

        LOG_INFO("Generating target image: {}", target_path.string());
        generateTargetImages();

        if (!std::filesystem::exists(target_path)) {
            throw std::runtime_error("Failed to generate target image: " + target_path.string());
        }

        return target_path.string();
    }

private:
    struct Config {
        std::filesystem::path models_directory{"./3d_models"};
        std::string object_name;
        std::filesystem::path output_directory{"target_images"};
        std::string target_filename{"target_1.png"};
        int num_images{1};
        double min_distance{1.0};
        double max_distance{3.0};
        bool generate_images{true};
    };

    void loadConfiguration() {
        config_.models_directory = config::get("paths.models_directory", config_.models_directory.string());
        config_.object_name = config::get("paths.object_name", config_.object_name);
        config_.output_directory = config::get("target_images.output_directory", config_.output_directory.string());
        config_.target_filename = config::get("target_images.filename", config_.target_filename);
        config_.num_images = std::stoi(config_.target_filename.substr(7, config_.target_filename.find('.') - 7));
        config_.min_distance = config::get("target_images.min_distance", config_.min_distance);
        config_.max_distance = config::get("target_images.max_distance", config_.max_distance);
        config_.generate_images = config::get("target_images.generate", config_.generate_images);
    }

    void generateTargetImages() const {
        std::filesystem::create_directories(getOutputDirectory());
        // simulator_->loadMesh(getModelPath().string());

        FibonacciLatticeSampler sampler({0, 0, 0}, {1, 1, 1}, 1.0);
        const auto viewpoints = sampler.generate(config_.num_images);

        for (int i = 0; i < config_.num_images; ++i) {
            const auto image_path = getOutputDirectory() / ("target_" + std::to_string(i + 1) + ".png");

            if (std::filesystem::exists(image_path)) {
                LOG_INFO("Image exists, skipping: {}", image_path.string());
                continue;
            }

            const auto extrinsics = generateExtrinsics(viewpoints.col(i), i);
            const auto rendered_image = simulator_->render(extrinsics, image_path.string());

            if (rendered_image.empty()) {
                LOG_ERROR("Failed to generate image: {}", image_path.string());
                continue;
            }

            LOG_INFO("Generated image: {}", image_path.string());

            if (image_path.filename() == config_.target_filename) break;
        }
    }

    [[nodiscard]] std::filesystem::path getModelPath() const {
        return config_.models_directory / (config_.object_name + ".ply");
    }

    [[nodiscard]] std::filesystem::path getOutputDirectory() const {
        return config_.output_directory / config_.object_name;
    }

    [[nodiscard]] Eigen::Matrix4d generateExtrinsics(const Eigen::Vector3d& viewpoint, int index) const {
        const double t = static_cast<double>(index) / (config_.num_images - 1);
        const double distance = std::lerp(config_.min_distance, config_.max_distance, t);
        const Eigen::Vector3d position = viewpoint.normalized() * distance;

        Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();
        extrinsics.block<3, 1>(0, 3) = position;
        extrinsics.block<3, 3>(0, 0) = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), -position).toRotationMatrix();

        return extrinsics;
    }

    std::shared_ptr<core::Simulator> simulator_;
    Config config_;
};

#endif //TARGET_GENERATOR_HPP
