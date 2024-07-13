// // File: tasks/task_manager.cpp
//
// #include "tasks/task_manager.hpp"
// #include "common/utilities/matrix.hpp"
// #include "common/logging/logger.hpp"
// #include "viewpoint/provider.hpp"
// #include "core/simulator.hpp"
// #include "optimization/cma_es_optimizer.hpp"
// #include <opencv2/opencv.hpp>
// #include <fstream>
// #include "common/io/image.hpp"
// #include "processing/image_processor.hpp"
// #include "processing/vision/distance_estimator.hpp"
// #include "viewpoint/generator.hpp"
//
// namespace tasks {
//
//     std::shared_ptr<TaskManager> TaskManager::instance_ = nullptr;
//     std::once_flag TaskManager::init_flag_;
//     cv::Mat TaskManager::target_image_;
//     core::Camera::Intrinsics TaskManager::camera_intrinsics_;
//
//     std::shared_ptr<TaskManager> TaskManager::getInstance() {
//         std::call_once(init_flag_, []() {
//             instance_ = std::shared_ptr<TaskManager>(new TaskManager());
//         });
//         return instance_;
//     }
//
//     TaskManager::TaskManager() {
//         LOG_DEBUG("Creating TaskManager...");
//         prepare();
//     }
//
//     void TaskManager::prepare() {
//         LOG_TRACE("Preparing TaskManager for execution...");
//         auto task_name = config::get("task.name", "task1");
//
//         // Prepare perception simulator
//         auto object_name = config::get("task.objects", "obj_000020");
//         auto model_directory = config::get("paths.model_directory", "./3d_models");
//         std::string model_path = fmt::format("{}/{}.ply", model_directory, object_name);
//         // perception_simulator_ = std::make_shared<core::Perception>(model_path);
//         camera_intrinsics_ = core::Perception::getCamera()->getIntrinsics();
//
//         // prepare viewpoint provider
//         const auto num_samples = config::get("viewpoints.count", 100);
//         const auto dimensions = config::get("viewpoints.dimensions", 3);
//         const auto from_file = config::get("viewpoints.from_file", false);
//         const auto target_image_path = config::get("paths.target_image", "../task1/target_images/obj_000020");
//
//         target_image_ = common::io::image::readImage(target_image_path);
//         provider_ = viewpoint::Provider::create(from_file, num_samples, dimensions);
//         provider_->setCameraIntrinsics(camera_intrinsics_);
//         provider_->setTargetImage(target_image_);
//
//     }
//
//     void TaskManager::execute(std::string object_name, size_t test_num) {
//         LOG_INFO("Executing task for object: {}, test number: {}", object_name, test_num);
//
//         auto task_name = config::get("task.name", "task1");
//         std::string selected_views_directory = fmt::format("../{}/selected_views/{}/", task_name, object_name);
//
//         auto view_space = provider_->provision();
//
//         LOG_INFO("Generated {} viewpoints!", view_space.size());
//
//         for (size_t count = 0; count < view_space.size(); count++) {
//             const Eigen::Vector3d position = view_space[count].getPosition();
//             LOG_DEBUG("Viewpoint/View #{}: ({}, {}, {})", count, position.x(), position.y(), position.z());
//         }
//
//         for (size_t test_id = 0; test_id < test_num; ++test_id) {
//             LOG_INFO("Running test {}", test_id);
//
//             auto optimizer = createOptimizer();
//
//             const auto result = optimizer->optimize(view_space, target_image_,
//                                                     [this](const core::View &view, const cv::Mat &target_image) {
//                                                         return this->evaluateView(view, target_image);
//                                                     });
//
//             LOG_INFO("Best score after optimization: {}", result.best_score);
//
//             std::ofstream fout(selected_views_directory + "test_" + std::to_string(test_id) + ".txt");
//             fout << result.optimized_views.size() << std::endl;
//             for (const auto &view: result.optimized_views) {
//                 fout << view.getPose() << std::endl;
//             }
//         }
//
//         LOG_INFO("Task execution completed for object: {}", object_name);
//     }
//
//
//     std::unique_ptr<optimization::CMAESOptimizer> TaskManager::createOptimizer() {
//         LOG_INFO("Creating CMA-ES optimizer");
//         int dimensions = config::get("viewpoints.dimensions", 3);
//         int population_size = config::get("optimization.population_size", 500);
//         double lower_bound = config::get("optimization.lower_bound", -1.0);
//         double upper_bound = config::get("optimization.upper_bound", 1.0);
//         return std::make_unique<optimization::CMAESOptimizer>(dimensions, population_size, lower_bound, upper_bound);
//     }
//
//
//     double TaskManager::evaluateView(const core::View &view, const cv::Mat &target_image) {
//         const std::string view_key = generateCacheKey(view);
//         if (view_score_cache_.count(view_key)) {
//             return view_score_cache_[view_key];
//         }
//
//         const Eigen::Vector3d position = view.getPosition();
//         std::string temp_image_path = fmt::format("../tmp/render_{:.3f}_{:.3f}_{:.3f}.png", position.x(), position.y(),
//                                                   position.z());
//         core::Perception::render(view.getPose(), temp_image_path);
//         const cv::Mat rendered_image = common::io::image::readImage(temp_image_path);
//
//         processing::vision::DistanceEstimator distance_estimator(camera_intrinsics_.getFocalLengthX());
//         auto dist = distance_estimator.estimate(rendered_image);
//         double area = processing::vision::DistanceEstimator::calculateObjectSize(rendered_image);
//
//         // const Eigen::Vector3d position = view.getPosition();
//         LOG_DEBUG("Estimated distance for view ({}, {}, {}), distance: {}", position.x(), position.y(), position.z(),
//                   dist);
//
//         if (rendered_image.empty()) {
//             LOG_ERROR("Rendered image is empty for view: {}",
//                       fmt::join(view.getPose().data(), view.getPose().data() + 16, ", "));
//             return std::numeric_limits<double>::max();
//         }
//
//         auto [similarity, match_score] = processing::image::ImageProcessor::compareImages(rendered_image, target_image);
//         double score = 1.0 - match_score; // Lower score -> better match
//
//         view_score_cache_[view_key] = score;
//
//         LOG_DEBUG("View evaluation score: {} (Lower is better), similarity: {}, match_score: {}", score, similarity,
//                   match_score);
//         return score;
//     }
//
//     std::string TaskManager::generateCacheKey(const core::View &view) {
//         std::ostringstream oss;
//         oss << view.getPose().transpose();
//         return oss.str();
//     }
//
//     std::unique_ptr<viewpoint::Provider> TaskManager::initializeGenerator(
//             const cv::Mat &target_image, const core::Camera::Intrinsics &camera_intrinsics, int num_samples,
//             int dimensions) {
//         auto generator = viewpoint::Provider::create(false, num_samples, dimensions);
//         generator->setTargetImage(target_image);
//         generator->setCameraIntrinsics(camera_intrinsics);
//         return generator;
//     }
//
// }
