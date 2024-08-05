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
#include "processing/image/occlusion_detector.hpp"
#include "types/image.hpp"
#include "types/viewpoint.hpp"

class Executor {
public:
    static void execute();

    Executor(const Executor &) = delete;

    Executor &operator=(const Executor &) = delete;

private:
    using ViewPoints = std::vector<ViewPoint<>>;
    using Images = std::vector<Image<>>;

    static std::once_flag init_flag_;
    static double radius_, target_score_;
    static Image<> target_;
    static std::shared_ptr<processing::image::ImageComparator> comparator_;
    static std::shared_ptr<processing::image::FeatureExtractor> extractor_;
    static std::shared_ptr<processing::image::FeatureMatcher> matcher_;

    Executor() = default;

    static void initialize();

    struct Defaults {
        static constexpr double fov_x = 0.95;
        static constexpr double fov_y = 0.75;
        static constexpr int width = 640;
        static constexpr int height = 480;
        static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
        static constexpr std::string_view target_image_path = "../../task1/target_images/obj_000020/target_2.png";
        static constexpr std::string_view comparator_type = "SSIM";
    };
};

#endif // EXECUTOR_HPP
