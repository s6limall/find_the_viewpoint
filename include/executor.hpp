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
#include "common/utilities/camera.hpp"
#include "common/utilities/visualizer.hpp"
#include "core/view.hpp"
#include "optimization/kernel/matern_52.hpp"
#include "optimization/octree.hpp"
#include "optimization/optimizer/gpr.hpp"
#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"
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

    Executor() = default;

    static void initialize();

    static void loadExtractor();
    static void loadComparator();

    struct Defaults {
        static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
        static constexpr std::string_view target_image_path = "./target_images/obj_000020/target_2.png";
    };
};

#endif // EXECUTOR_HPP
