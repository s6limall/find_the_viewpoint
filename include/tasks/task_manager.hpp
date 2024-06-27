// File: tasks/task_manager.hpp

#ifndef TASK_MANAGER_HPP
#define TASK_MANAGER_HPP

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "core/view.hpp"
#include "core/perception.hpp"
#include "optimization/optimizer.hpp"
#include "viewpoint/generator.hpp"

namespace tasks {

    class TaskManager {
    public:
        // Returns the singleton instance of TaskManager.
        static TaskManager &getInstance();

        // Executes a specified task by performing operations on a 3D model.
        // - object_name: Name of the 3D object involved in the task.
        // - test_num: The number of tests to perform.
        void execute(const std::string &object_name, int test_num);

        TaskManager(const TaskManager &) = delete;

        TaskManager &operator=(const TaskManager &) = delete;

    private:
        TaskManager() = default;

        ~TaskManager() = default;

        std::shared_ptr<core::Perception> perception_simulator_;

        // Initializes the Perception system with a specified 3D model.
        std::shared_ptr<core::Perception> initializePerception(const std::string &model_path);

        // Loads viewpoints from a file or generates new viewpoints using sampling.
        std::vector<core::View> loadOrGenerateViewpoints(const std::string &filepath, bool from_file,
                                                         int num_samples, int dimensions, unsigned int seed);

        std::unique_ptr<optimization::Optimizer> createOptimizer();

        // Evaluates a view against the target image and caches the result.
        double evaluateView(const core::View &view, const cv::Mat &target_image);

        // Cache for storing evaluated views and their scores.
        std::unordered_map<std::string, double> view_score_cache_;

        // Generates a unique key for caching based on the view.
        std::string generateCacheKey(const core::View &view) const;

        std::unique_ptr<viewpoint::Generator> initializeGenerator(
                const cv::Mat &target_image, const cv::Mat &camera_matrix, int num_samples, int dimensions,
                unsigned int seed);
    };

}

#endif // TASK_MANAGER_HPP
