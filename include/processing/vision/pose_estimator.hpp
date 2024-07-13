#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "core/camera.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"

namespace processing::vision {

    class PoseEstimator {
    public:
        explicit PoseEstimator(std::shared_ptr<core::Camera> camera);

        Eigen::Matrix4d estimatePose(const cv::Mat &image);

        [[nodiscard]] Eigen::Matrix4d getPose() const noexcept;

    private:
        std::shared_ptr<core::Camera> camera_;
        Eigen::Matrix4d pose_matrix_;

        std::vector<cv::Point2f> detectKeypoints(const cv::Mat &image);

        std::vector<Eigen::Vector3d> generateObjectPoints(int rows, int cols, float square_size);

    };


}

#endif // POSE_ESTIMATOR_HPP


/*
void Executor::initialize() {
target_image_ = common::io::image::readImage(config::get("paths.target_image", "target.png"));

// Create instances of feature extractor and matcher
auto extractor = processing::image::FeatureExtractor::create<processing::image::ORBExtractor>();
auto matcher = processing::image::FeatureMatcher::create<processing::image::BFMatcher>();

const auto sift_extractor = processing::image::FeatureExtractor::create<processing::image::SIFTExtractor>();
auto [keypoints, descriptors] = sift_extractor->extract(target_image_);

const auto camera = std::make_shared<core::Camera>();
const auto width = config::get("camera.width", Defaults::width);
const auto height = config::get("camera.height", Defaults::height);
const auto fov_x = config::get("camera.fov_x", Defaults::fov_x);
const auto fov_y = config::get("camera.fov_y", Defaults::fov_y);

LOG_INFO("Configuring camera with width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);
camera->setIntrinsics(width, height, fov_x, fov_y);

// Initialize PoseEstimator
pose_estimator_ = std::make_unique<processing::vision::PoseEstimator>(camera);

LOG_INFO("Executor initialized.");
}

*/
