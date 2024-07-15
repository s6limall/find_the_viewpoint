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
#include "processing/vision/estimation/distance_estimator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
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
#include "processing/vision/estimation/distance_estimator.hpp"
#include "types/image.hpp"
#include "viewpoint/evaluator.hpp"
#include "viewpoint/provider.hpp"

class Executor {
public:
    static void execute();

    Executor(const Executor &) = delete;

    Executor &operator=(const Executor &) = delete;

private:
    using ViewPoints = std::vector<ViewPoint<> >;

    static std::once_flag init_flag_;
    static Image<> target_;
    // static std::shared_ptr<core::Camera> camera_;
    static std::unique_ptr<processing::vision::DistanceEstimator> distance_estimator_;
    static std::unique_ptr<viewpoint::Provider<> > provider_;
    static std::unique_ptr<processing::image::ImageComparator> comparator_;
    static std::unique_ptr<viewpoint::Evaluator<> > evaluator_;
    static std::unique_ptr<processing::image::FeatureExtractor> extractor_;
    static double distance_;

    Executor() = default;

    static void initialize();

    static std::vector<Cluster<> > clusterSamples(const std::vector<Image<> > &evaluated_images);

    // static void poseEstimation(std::vector<Image<> > &images);
    static std::vector<ViewPoint<> > matchAndRansac(const std::vector<Image<> > &images,
                                                    const Image<> &target);

    static ViewPoint<> predictNextViewpoint(const std::vector<ViewPoint<> > &evaluated_points);

    static double objectiveFunction(const ViewPoint<> &point,
                                    const std::unique_ptr<processing::image::ImageComparator> &comparator,
                                    viewpoint::Evaluator<> &evaluator);


    struct Defaults {
        static constexpr double fov_x = 0.95;
        static constexpr double fov_y = 0.75;
        static constexpr int width = 640;
        static constexpr int height = 480;
        static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
    };
};

#endif // EXECUTOR_HPP
