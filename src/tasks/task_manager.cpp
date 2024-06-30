// File: tasks/task_manager.cpp

#include "tasks/task_manager.hpp"
#include "common/utilities/file_utils.hpp"
#include "common/utilities/matrix_utils.hpp"
#include "common/logging/logger.hpp"
#include "viewpoint/provider.hpp"
#include "core/simulator.hpp"
#include "optimization/cma_es_optimizer.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "common/io/image.hpp"
#include "processing/image_processor.hpp"
#include "viewpoint/generator.hpp"

namespace tasks {

    std::shared_ptr<TaskManager> TaskManager::instance_ = nullptr;
    std::once_flag TaskManager::init_flag_;
    cv::Mat TaskManager::target_image_;
    core::Camera::CameraParameters TaskManager::camera_parameters_;

    std::shared_ptr<TaskManager> TaskManager::getInstance() {
        std::call_once(init_flag_, []() {
            instance_ = std::shared_ptr<TaskManager>(new TaskManager());
        });
        return instance_;
    }

    TaskManager::TaskManager() {
        LOG_INFO("Creating TaskManager...");
        prepare();
    }

    void TaskManager::prepare() {
        LOG_INFO("Preparing TaskManager for execution...");
        auto task_name = config::get("task.name", "task1");

        // Prepare perception simulator
        auto object_name = config::get("task.objects", "obj_000020");
        auto model_directory = config::get("paths.model_directory", "../3d_models");
        std::string model_path = fmt::format("{}/{}.ply", model_directory, object_name);
        perception_simulator_ = std::make_shared<core::Perception>(model_path);
        camera_parameters_ = perception_simulator_->getCamera()->getParameters();

        // prepare viewpoint provider
        const auto num_samples = config::get("viewpoints.count", 100);
        const auto dimensions = config::get("viewpoints.dimensions", 3);
        const auto from_file = config::get("viewpoints.from_file", false);
        const auto target_image_path = config::get("paths.target_image", "../task1/target_images/obj_000020");
        target_image_ = common::io::image::readImage(target_image_path);
        provider_ = viewpoint::Provider::create(from_file, num_samples, dimensions);
        provider_->setCameraParameters(camera_parameters_);
        provider_->setTargetImage(target_image_);

    }

    void TaskManager::execute(std::string object_name, size_t test_num) {
        LOG_INFO("Executing task for object: {}, test number: {}", object_name, test_num);

        auto task_name = config::get("task.name", "task1");
        std::string selected_views_directory = fmt::format("../{}/selected_views/{}/", task_name, object_name);

        auto view_space = provider_->provision();

        LOG_INFO("Generated {} viewpoints!", view_space.size());

        for (size_t count = 0; count < view_space.size(); count++) {
            const Eigen::Vector3f position = view_space[count].getPosition();
            LOG_DEBUG("Viewpoint/View #{}: ({}, {}, {})", count, position.x(), position.y(), position.z());
        }

        for (size_t test_id = 0; test_id < test_num; ++test_id) {
            LOG_INFO("Running test {}", test_id);

            auto optimizer = createOptimizer();

            const auto result = optimizer->optimize(view_space, target_image_,
                                                    [this](const core::View &view, const cv::Mat &target_image) {
                                                        return this->evaluateView(view, target_image_);
                                                    });

            LOG_INFO("Best score after optimization: {}", result.best_score);

            std::ofstream fout(selected_views_directory + "test_" + std::to_string(test_id) + ".txt");
            fout << result.optimized_views.size() << std::endl;
            for (const auto &view: result.optimized_views) {
                fout << view.getPose() << std::endl;
            }
        }

        LOG_INFO("Task execution completed for object: {}", object_name);
    }

    std::unique_ptr<optimization::Optimizer> TaskManager::createOptimizer() {
        LOG_INFO("Creating CMA-ES optimizer");
        return std::make_unique<optimization::CMAESOptimizer>();
    }

    double TaskManager::evaluateView(const core::View &view, const cv::Mat &target_image) {
        const std::string view_key = generateCacheKey(view);
        if (view_score_cache_.count(view_key)) {
            return view_score_cache_[view_key];
        }

        std::string temp_image_path = "../tmp/rendered_view.png";
        perception_simulator_->render(view.getPose(), temp_image_path);
        const cv::Mat rendered_image = common::io::image::readImage(temp_image_path);

        if (rendered_image.empty()) {
            LOG_ERROR("Rendered image is empty for view: {}",
                      fmt::join(view.getPose().data(), view.getPose().data() + 16, ", "));
            return std::numeric_limits<double>::max();
        }

        auto [similarity, match_score] = processing::image::ImageProcessor::compareImages(rendered_image, target_image);
        double score = 1.0 - similarity; // Lower score -> better match

        view_score_cache_[view_key] = score;

        LOG_DEBUG("View evaluation score: {}, similarity: {}, match_score: {}", score, similarity, match_score);
        return score;
    }

    std::string TaskManager::generateCacheKey(const core::View &view) const {
        std::ostringstream oss;
        oss << view.getPose().transpose();
        return oss.str();
    }

    std::unique_ptr<viewpoint::Provider> TaskManager::initializeGenerator(
            const cv::Mat &target_image, const core::Camera::CameraParameters &camera_parameters, int num_samples,
            int dimensions) {
        auto generator = viewpoint::Provider::create(false, num_samples, dimensions);
        generator->setTargetImage(target_image);
        generator->setCameraParameters(camera_parameters);
        return generator;
    }

}

/*void TaskManager::execute(const std::string &object_name, int test_num) {
        LOG_INFO("Executing task for object: {}, test number: {}", object_name, test_num);

        std::string model_directory = "../3d_models/";
        std::string view_space_file = "../view_space/5.txt";
        std::string view_space_images_directory = "../task1/viewspace_images/" + object_name + "/";
        std::string selected_views_directory = "../task1/selected_views/" + object_name + "/";

        common::utilities::FileUtils::ensureDirectoryExists(view_space_images_directory);
        common::utilities::FileUtils::ensureDirectoryExists(selected_views_directory);

        perception_simulator_ = initializePerception(model_directory + object_name + ".ply");

        auto target_views = loadOrGenerateViewpoints(view_space_file, true, 1, 3, 42);
        if (target_views.empty()) {
            LOG_ERROR("Failed to load target view for testing.");
            return;
        }

        cv::Mat target_image = cv::imread(view_space_images_directory + "rgb_0.png");
        if (target_image.empty()) {
            LOG_ERROR("Failed to load target image from path: {}", view_space_images_directory + "rgb_0.png");
            return;
        }

        auto view_space = loadOrGenerateViewpoints(view_space_file, false, 10, 3, 42);
        LOG_INFO("Generated {} viewpoints", view_space.size());

        for (int test_id = 0; test_id < test_num; ++test_id) {
            LOG_INFO("Running test {}", test_id);

            auto optimizer = createOptimizer();

            auto result = optimizer->optimize(view_space, target_image,
                                              [this](const core::View &view, const cv::Mat &target_image) {
                                                  return this->evaluateView(view, target_image);
                                              });

            LOG_INFO("Best score after optimization: {}", result.best_score);

            std::ofstream fout(selected_views_directory + "test_" + std::to_string(test_id) + ".txt");
            fout << result.optimized_views.size() << std::endl;
            for (const auto &view: result.optimized_views) {
                fout << view.getPose() << std::endl;
            }
        }

        LOG_INFO("Task execution completed for object: {}", object_name);
    }*/
