// File: executor.hpp

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include "common/io/image.hpp"
#include "common/logging/logger.hpp"
#include "core/view.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "types/image.hpp"
#include "types/viewpoint.hpp"

class Executor {
public:
    static void execute();

    Executor(const Executor &) = delete;

    Executor &operator=(const Executor &) = delete;

private:
    static std::once_flag init_flag_;
    static double radius_, target_score_;
    static Image<> target_;
    static std::shared_ptr<processing::image::ImageComparator> comparator_;
    static std::shared_ptr<processing::image::FeatureExtractor> extractor_;
    static std::shared_ptr<processing::image::FeatureMatcher> matcher_;

    static std::shared_ptr<core::Simulator> simulator_;
    static std::string object_name_;
    static std::filesystem::path output_directory_;
    static std::filesystem::path models_directory_;

    Executor() = default;

    static void initialize();

    static void loadExtractor();
    static void loadComparator();

    static void generateTargetImages();
    static std::string getRandomTargetImagePath();
    static Eigen::Matrix4d generateRandomExtrinsics();

    struct Defaults {
        static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
        static constexpr std::string_view target_image_path = "../../task1/target_images/obj_000020/target_2.png";
    };
};

#endif // EXECUTOR_HPP
