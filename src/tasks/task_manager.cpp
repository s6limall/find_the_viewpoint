// File: tasks/task_manager.cpp

#include "tasks/task_manager.hpp"
#include "common/utilities/file_utils.hpp"
#include "common/utilities/matrix_utils.hpp"
#include "viewpoint/provider.hpp"
#include "core/simulator.hpp"
#include "optimization/cma_es_optimizer.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "processing/image_processor.hpp"
#include "viewpoint/generator.hpp"

namespace tasks {
    // Singleton instance access
    TaskManager &TaskManager::getInstance() {
        static TaskManager instance;
        return instance;
    }

    /*void TaskManager::execute(const std::string &object_name, int test_num) {
        spdlog::info("Executing task for object: {}, test number: {}", object_name, test_num);

        std::string model_directory = "../3d_models/";
        std::string view_space_file = "../view_space/5.txt";
        std::string view_space_images_directory = "../task1/viewspace_images/" + object_name + "/";
        std::string selected_views_directory = "../task1/selected_views/" + object_name + "/";

        common::utilities::FileUtils::ensureDirectoryExists(view_space_images_directory);
        common::utilities::FileUtils::ensureDirectoryExists(selected_views_directory);

        perception_simulator_ = initializePerception(model_directory + object_name + ".ply");

        auto target_views = loadOrGenerateViewpoints(view_space_file, true, 1, 3, 42);
        if (target_views.empty()) {
            spdlog::error("Failed to load target view for testing.");
            return;
        }

        cv::Mat target_image = cv::imread(view_space_images_directory + "rgb_0.png");
        if (target_image.empty()) {
            spdlog::error("Failed to load target image from path: {}", view_space_images_directory + "rgb_0.png");
            return;
        }

        auto view_space = loadOrGenerateViewpoints(view_space_file, false, 10, 3, 42);
        spdlog::info("Generated {} viewpoints", view_space.size());

        for (int test_id = 0; test_id < test_num; ++test_id) {
            spdlog::info("Running test {}", test_id);

            auto optimizer = createOptimizer();

            auto result = optimizer->optimize(view_space, target_image,
                                              [this](const core::View &view, const cv::Mat &target_image) {
                                                  return this->evaluateView(view, target_image);
                                              });

            spdlog::info("Best score after optimization: {}", result.best_score);

            std::ofstream fout(selected_views_directory + "test_" + std::to_string(test_id) + ".txt");
            fout << result.optimized_views.size() << std::endl;
            for (const auto &view: result.optimized_views) {
                fout << view.getPose() << std::endl;
            }
        }

        spdlog::info("Task execution completed for object: {}", object_name);
    }*/

    void TaskManager::execute(const std::string &object_name, int test_num) {
        spdlog::info("Executing task for object: {}, test number: {}", object_name, test_num);

        std::string model_directory = "../3d_models/";
        std::string view_space_file = "../view_space/5.txt";
        std::string view_space_images_directory = "../task1/viewspace_images/" + object_name + "/";
        std::string selected_views_directory = "../task1/selected_views/" + object_name + "/";

        common::utilities::FileUtils::ensureDirectoryExists(view_space_images_directory);
        common::utilities::FileUtils::ensureDirectoryExists(selected_views_directory);

        perception_simulator_ = initializePerception(model_directory + object_name + ".ply");

        cv::Mat target_image = cv::imread(view_space_images_directory + "rgb_2.png");
        if (target_image.empty()) {
            spdlog::error("Failed to load target image from path: {}", view_space_images_directory + "rgb_2.png");
            return;
        }

        cv::Mat camera_matrix = common::utilities::eigenToCvMat(perception_simulator_->getCamera()->getIntrinsics());

        auto generator = initializeGenerator(target_image, camera_matrix, 100, 3, 42);
        auto view_space = generator->provision();
        spdlog::info("Generated {} viewpoints", view_space.size());

        for (int test_id = 0; test_id < test_num; ++test_id) {
            spdlog::info("Running test {}", test_id);

            auto optimizer = createOptimizer();

            auto result = optimizer->optimize(view_space, target_image,
                                              [this](const core::View &view, const cv::Mat &target_image) {
                                                  return this->evaluateView(view, target_image);
                                              });

            spdlog::info("Best score after optimization: {}", result.best_score);

            std::ofstream fout(selected_views_directory + "test_" + std::to_string(test_id) + ".txt");
            fout << result.optimized_views.size() << std::endl;
            for (const auto &view: result.optimized_views) {
                fout << view.getPose() << std::endl;
            }
        }

        spdlog::info("Task execution completed for object: {}", object_name);
    }

    std::vector<core::View> TaskManager::loadOrGenerateViewpoints(const std::string &filepath, bool from_file,
                                                                  int num_samples, int dimensions, unsigned int seed) {
        const auto viewpoint_provider = viewpoint::Provider::createProvider(
                from_file, filepath, num_samples, dimensions);
        spdlog::debug("Generated viewpoints using {}", from_file ? "file" : "sampling");
        return viewpoint_provider->provision();
    }

    std::unique_ptr<optimization::Optimizer> TaskManager::createOptimizer() {
        spdlog::info("Creating CMA-ES optimizer");
        return std::make_unique<optimization::CMAESOptimizer>();
    }

    double TaskManager::evaluateView(const core::View &view, const cv::Mat &target_image) {
        std::string view_key = generateCacheKey(view);
        if (view_score_cache_.count(view_key)) {
            return view_score_cache_[view_key];
        }

        std::string temp_image_path = "../tmp/rendered_view.png";
        perception_simulator_->render(view.getPose(), temp_image_path);
        cv::Mat rendered_image = cv::imread(temp_image_path);

        if (rendered_image.empty()) {
            spdlog::error("Rendered image is empty for view: {}",
                          fmt::join(view.getPose().data(), view.getPose().data() + 16, ", "));
            return std::numeric_limits<double>::max();
        }

        auto [similarity, match_score] = processing::image::ImageProcessor::compareImages(rendered_image, target_image);
        double score = 1.0 - similarity; // Lower score -> better match

        view_score_cache_[view_key] = score;

        spdlog::debug("View evaluation score: {}, similarity: {}, match_score: {}", score, similarity, match_score);
        return score;
    }


    std::string TaskManager::generateCacheKey(const core::View &view) const {
        std::ostringstream oss;
        oss << view.getPose().transpose();
        return oss.str();
    }

    std::shared_ptr<core::Perception> TaskManager::initializePerception(const std::string &model_path) {
        spdlog::info("Initializing Perception with model path: {}", model_path);
        return std::make_shared<core::Perception>(model_path);
    }

    std::unique_ptr<viewpoint::Generator> TaskManager::initializeGenerator(
            const cv::Mat &target_image, const cv::Mat &camera_matrix, int num_samples, int dimensions,
            unsigned int seed) {
        auto generator = std::make_unique<viewpoint::Generator>(num_samples, dimensions);
        generator->setTargetImage(target_image);
        generator->setCameraMatrix(camera_matrix);
        return generator;
    }

}
