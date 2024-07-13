// File: executor.hpp

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pcl/visualization/pcl_visualizer.h>
#include <mutex>
#include <memory>
#include <functional>

#include "common/logging/logger.hpp"
#include "common/utilities/visualizer.hpp"
#include "processing/vision/distance_estimator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
#include "processing/vision/pose_estimator.hpp"
#include "sampling/sampler/halton_sampler.hpp"
#include "sampling/transformer/spherical_transformer.hpp"
#include "clustering/dbscan.hpp"
#include "types/viewpoint.hpp"
#include "core/perception.hpp"
#include "core/view.hpp"
#include "processing/image/feature/extractor.hpp"

#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/matcher/bf_matcher.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "filtering/quadrant_filter.hpp"

class Executor {
public:
    static void execute();

    Executor(const Executor &) = delete;

    Executor &operator=(const Executor &) = delete;

private:
    static std::once_flag init_flag_;
    static cv::Mat target_image_;
    static std::shared_ptr<core::Camera> camera_;

    Executor() = default;

    static void initialize();

    static double estimateDistance();

    static std::vector<ViewPoint<double> > generateSamples(double estimated_distance);

    static std::vector<ViewPoint<double> > evaluateSamples(const processing::image::ImageComparator &comparator,
                                                           const std::vector<ViewPoint<double> > &samples,
                                                           double distance);

    static std::vector<Cluster<double> > clusterSamples(const std::vector<ViewPoint<double> > &evaluated_samples);


    struct Defaults {
        static constexpr double fov_x = 0.95;
        static constexpr double fov_y = 0.75;
        static constexpr int width = 640;
        static constexpr int height = 480;
        static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
    };
};

#endif // EXECUTOR_HPP
