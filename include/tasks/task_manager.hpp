// File: tasks/task_manager.hpp

#ifndef TASK_MANAGER_HPP
#define TASK_MANAGER_HPP

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "core/view.hpp"
#include "core/perception.hpp"
#include "optimization/cma_es_optimizer.hpp"
#include "optimization/optimizer.hpp"
#include "viewpoint/generator.hpp"

namespace tasks {

    class TaskManager {
    public:
        // Returns the singleton instance of TaskManager.
        static std::shared_ptr<TaskManager> getInstance();

        // Executes a specified task by performing operations on a 3D model.
        // - object_name: Name of the 3D object involved in the task.
        // - test_num: The number of tests to perform.
        void execute(std::string object_name = "obj_000020", size_t test_num = 1);

        TaskManager(const TaskManager &) = delete;

        TaskManager &operator=(const TaskManager &) = delete;

        ~TaskManager() = default;

    private:
        TaskManager();

        static cv::Mat target_image_;
        static std::once_flag init_flag_;
        static std::shared_ptr<TaskManager> instance_;
        static core::Camera::Intrinsics camera_intrinsics_;
        std::unique_ptr<viewpoint::Provider> provider_;

        void prepare();

        // Loads viewpoints from a file or generates new viewpoints using sampling.
        std::vector<core::View> loadOrGenerateViewpoints(const std::string &filepath, bool from_file,
                                                         int num_samples, int dimensions, unsigned int seed);

        static std::unique_ptr<optimization::CMAESOptimizer> createOptimizer();

        // Evaluates a view against the target image and caches the result.
        double evaluateView(const core::View &view, const cv::Mat &target_image);

        // Cache for storing evaluated views and their scores.
        std::unordered_map<std::string, double> view_score_cache_;

        // Generates a unique key for caching based on the view.
        static std::string generateCacheKey(const core::View &view);

        static std::unique_ptr<viewpoint::Provider> initializeGenerator(
                const cv::Mat &target_image, const core::Camera::Intrinsics &camera_intrinsics, int num_samples,
                int dimensions);
    };

}

#endif // TASK_MANAGER_HPP

