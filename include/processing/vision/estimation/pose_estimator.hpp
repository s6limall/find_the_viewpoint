// File: processing/vision/estimation/pose_estimator.hpp

#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "processing/viewpoint/pnp.hpp"

namespace processing::vision {


    template<typename Scalar = double>
    class PoseEstimator {
    public:
        struct Config {
            std::unique_ptr<processing::image::FeatureExtractor> featureExtractor;
            std::unique_ptr<processing::image::FeatureMatcher> featureMatcher;
        };

        PoseEstimator(const core::Camera::Intrinsics &intrinsics, Config config) :
            pnpSolver_(intrinsics), config_(std::move(config)) {}

        std::optional<std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>>
        estimatePose(const cv::Mat &objectImage, const cv::Mat &sceneImage,
                     const std::vector<Eigen::Matrix<Scalar, 3, 1>> &objectPoints3D) const {

            auto [objectKeypoints, objectDescriptors] = config_.featureExtractor->extract(objectImage);
            auto [sceneKeypoints, sceneDescriptors] = config_.featureExtractor->extract(sceneImage);

            auto matches = config_.featureMatcher->match(objectDescriptors, sceneDescriptors);

            LOG_DEBUG("Found {} matches.", matches.size());

            std::vector<Eigen::Matrix<Scalar, 3, 1>> matchedObjectPoints;
            std::vector<Eigen::Matrix<Scalar, 2, 1>> matchedImagePoints;
            for (const auto &match: matches) {
                matchedObjectPoints.push_back(objectPoints3D[match.queryIdx]);
                matchedImagePoints.push_back(Eigen::Matrix<Scalar, 2, 1>(sceneKeypoints[match.trainIdx].pt.x,
                                                                         sceneKeypoints[match.trainIdx].pt.y));
            }

            LOG_DEBUG("Found {} matched object points.", matchedObjectPoints.size());
            return pnpSolver_.estimatePose(matchedObjectPoints, matchedImagePoints);
        }

    private:
        viewpoint::PnPSolver<Scalar> pnpSolver_;
        Config config_;
    };


} // namespace processing::vision

#endif // POSE_ESTIMATOR_HPP
