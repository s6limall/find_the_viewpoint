// File: executor.cpp

#include "executor.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "core/view.hpp"
#include "core/camera.hpp"
#include "common/logging/logger.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "viewpoint/generator.hpp"
#include "viewpoint/provider.hpp"

using Visualizer = common::utilities::Visualizer;

std::once_flag Executor::init_flag_;
Image<> Executor::target_;
std::unique_ptr<processing::vision::DistanceEstimator> Executor::distance_estimator_;
std::unique_ptr<viewpoint::Provider<> > Executor::provider_;
std::unique_ptr<viewpoint::Evaluator<> > Executor::evaluator_;
std::unique_ptr<processing::image::ImageComparator> Executor::comparator_;
double Executor::distance_;

void Executor::initialize() {
    // const auto target_image = common::io::image::readImage(config::get("paths.target_image", "target.png"));
    const auto target_image(common::io::image::readImage("../../task1/target_images/obj_000020/img.png"));
    const auto extractor = FeatureExtractor::create<processing::image::ORBExtractor>();
    target_ = Image<>(target_image, extractor);
    distance_estimator_ = std::make_unique<processing::vision::DistanceEstimator>();
    distance_ = distance_estimator_->estimate(target_.getImage());
    provider_ = std::make_unique<viewpoint::Generator<> >(distance_);
    comparator_ = std::make_unique<processing::image::SSIMComparator>(); // TODO: SSIM
    evaluator_ = std::make_unique<viewpoint::Evaluator<> >(target_);
}

void Executor::execute() {
    try {
        std::call_once(init_flag_, &Executor::initialize);

        const auto samples = provider_->provision();

        auto evaluated_samples = evaluator_->evaluate(comparator_, samples);

        QuadrantFilter<Image<> > quadrant_filter;
        const auto filtered_images = quadrant_filter.filter(evaluated_samples, [](const Image<> &image) {
            return image.getScore() > 0.0;
        });

        auto clusters = clusterSamples(filtered_images);

        std::vector<ViewPoint<> > points;
        for (const auto &cluster: clusters) {
            points.insert(points.end(), cluster.getPoints().begin(), cluster.getPoints().end());
            LOG_INFO("Points in cluster: {}, Average score = {}", cluster.size(), cluster.getAverageScore());
        }

        const Cluster<> &best_cluster = *std::max_element(clusters.begin(), clusters.end(),
                                                          [](const Cluster<> &a, const Cluster<> &b) {
                                                              return a.getAverageScore() < b.getAverageScore();
                                                          });

        LOG_INFO("Best cluster: {}, Average score = {}", best_cluster.size(), best_cluster.getAverageScore());

        // Convert best cluster points to images
        const auto extractor = FeatureExtractor::create<processing::image::ORBExtractor>();
        const auto best_points = best_cluster.getPoints();
        std::vector<Image<> > best_images;
        best_images.reserve(best_points.size());
        for (const auto &point: best_points) {
            core::View view = point.toView();
            cv::Mat rendered_image = core::Perception::render(view.getPose());
            Image<> image(rendered_image, extractor);
            image.setViewPoint(point);
            image.setScore(point.getScore());
            best_images.push_back(image);
        }

        // poseEstimation(best_images);

        const auto filtered_points = matchFeaturesAndFilterRANSAC(best_images, target_);
        for (const auto &point: filtered_points) {
            LOG_INFO("Filtered Point: ({}, {}, {}), Score: {}", point.getPosition().x(), point.getPosition().y(),
                     point.getPosition().z(), point.getScore());
        }

        // Find the best viewpoint based on the number of inliers
        auto best_viewpoint_it = std::max_element(filtered_points.begin(), filtered_points.end(),
                                                  [](const ViewPoint<> &a, const ViewPoint<> &b) {
                                                      return a.getScore() < b.getScore();
                                                  });

        if (best_viewpoint_it != filtered_points.end()) {
            const auto &best_viewpoint = *best_viewpoint_it;
            LOG_INFO("Best Viewpoint: ({}, {}, {}), Score: {}",
                     best_viewpoint.getPosition().x(), best_viewpoint.getPosition().y(),
                     best_viewpoint.getPosition().z(), best_viewpoint.getScore());
        } else {
            LOG_WARN("No best viewpoint found.");
        }

        Visualizer::visualizeResults(best_cluster.getPoints(), distance_ * 0.9, distance_ * 1.1);
    } catch (const std::exception &e) {
        LOG_ERROR("An error occurred during execution: {}", e.what());
        throw;
    }
}

std::vector<Cluster<> > Executor::clusterSamples(const std::vector<Image<> > &evaluated_images) {
    // Extract ViewPoint from evaluated_images
    std::vector<ViewPoint<> > viewpoints;
    viewpoints.reserve(evaluated_images.size());

    for (const auto &image: evaluated_images) {
        if (image.hasViewPoint()) {
            viewpoints.push_back(image.getViewPoint());
        }
    }

    // Perform clustering
    auto metric = [](const ViewPoint<> &a, const ViewPoint<> &b) {
        return std::abs(a.getScore() - b.getScore());
    };

    clustering::DBSCAN<double> dbscan(5, metric); // min_points = 5
    auto clusters = dbscan.cluster(viewpoints);

    // Log cluster information
    for (const auto &cluster: clusters) {
        double average_score = cluster.getAverageScore();
        LOG_DEBUG("Cluster ID: {}, Number of Points: {}, Average Score: {}", cluster.getClusterId(), cluster.size(),
                  average_score);
        for (const auto &point: cluster.getPoints()) {
            ;
            auto [x, y, z] = point.toCartesian();
            LOG_DEBUG("Point: ({}, {}, {}), Score: {}", x, y, z, point.getScore());
        }
    }

    return clusters;
}

std::vector<ViewPoint<> > Executor::matchFeaturesAndFilterRANSAC(const std::vector<Image<> > &images,
                                                                 const Image<> &target) {
    processing::image::BFMatcher matcher;
    std::vector<ViewPoint<> > filtered_points;

    for (const auto &image: images) {
        std::vector<cv::Point2f> target_points, image_points;
        std::vector<cv::DMatch> good_matches;

        // Match features
        auto matches = matcher.match(target.getDescriptors(), image.getDescriptors());
        LOG_DEBUG("Number of matches found: {}", matches.size());

        // Convert keypoints to points for RANSAC
        for (const auto &match: matches) {
            target_points.push_back(target.getKeypoints()[match.queryIdx].pt);
            image_points.push_back(image.getKeypoints()[match.trainIdx].pt);
        }

        if (target_points.size() >= 4) {
            // Perform RANSAC to filter matches
            std::vector<uchar> inliers_mask;
            cv::Mat homography = cv::findHomography(target_points, image_points, cv::RANSAC, 3.0, inliers_mask);
            LOG_DEBUG("Homography matrix: {}", homography);

            for (size_t i = 0; i < inliers_mask.size(); ++i) {
                if (inliers_mask[i]) {
                    good_matches.push_back(matches[i]);
                }
            }
            LOG_DEBUG("Number of inliers after RANSAC: {}", good_matches.size());

            if (!good_matches.empty()) {
                ViewPoint<> point = image.getViewPoint();
                point.setScore(static_cast<double>(good_matches.size()) / matches.size());
                filtered_points.push_back(point);
            }
        } else {
            LOG_WARN("Not enough points for RANSAC. Matches size: {}", matches.size());
        }
    }

    return filtered_points;
}


ViewPoint<> Executor::predictNextViewpoint(const std::vector<ViewPoint<> > &evaluated_points) {
    // Implement Bayesian Optimization to predict the next best viewpoint
    // Placeholder implementation, needs actual Bayesian Optimization logic
    return evaluated_points.front(); // Returning the first evaluated point as a placeholder
}


/*void Executor::poseEstimation(std::vector<Image<> > &images) {
    auto matcher = processing::image::FeatureMatcher::create<processing::image::BFMatcher>();
    auto camera = core::Perception::getCamera();
    cv::Mat camera_matrix;
    cv::eigen2cv(camera->getIntrinsics().getMatrix(), camera_matrix);

    for (auto &image: images) {
        if (!image.hasViewPoint())
            continue;

        const auto &target_descriptors = target_.getDescriptors();
        const auto &image_descriptors = image.getDescriptors();
        const auto &target_keypoints = target_.getKeypoints();
        const auto &image_keypoints = image.getKeypoints();

        LOG_DEBUG("Matching features between target and captured image.");
        std::vector<cv::DMatch> matches = matcher->match(target_descriptors, image_descriptors);

        std::vector<cv::Point2f> target_points, image_points;
        std::vector<cv::Point3f> object_points; // 3D points in world space
        for (const auto &match: matches) {
            target_points.push_back(target_keypoints[match.queryIdx].pt);
            image_points.push_back(image_keypoints[match.trainIdx].pt);

            // Assuming the keypoints' indices match the 3D points in ViewPoint
            const auto &viewpoint = image.getViewPoint();
            object_points.emplace_back(viewpoint.getPosition().x(), viewpoint.getPosition().y(),
                                       viewpoint.getPosition().z());
        }

        LOG_DEBUG("Number of matches found: {}", matches.size());
        if (target_points.size() >= 4 && image_points.size() >= 4 && object_points.size() >= 4) {
            cv::Mat inliers_mask;
            std::vector<cv::Point2f> inliers_target_points, inliers_image_points;
            std::vector<cv::Point3f> inliers_object_points;

            cv::findHomography(target_points, image_points, cv::RANSAC, 3.0, inliers_mask);

            for (size_t i = 0; i < matches.size(); i++) {
                if (inliers_mask.at<uchar>(i)) {
                    inliers_target_points.push_back(target_points[i]);
                    inliers_image_points.push_back(image_points[i]);
                    inliers_object_points.push_back(object_points[i]);
                }
            }

            LOG_DEBUG("Number of inliers after RANSAC: {}", inliers_target_points.size());

            if (inliers_object_points.size() >= 4) {
                cv::Mat rvec, tvec;
                bool success = cv::solvePnPRansac(inliers_object_points, inliers_image_points, camera_matrix, cv::Mat(),
                                                  rvec, tvec, false, 100, 8.0, 0.99, inliers_mask);

                if (success) {
                    LOG_DEBUG("solvePnPRansac succeeded.");
                    cv::Mat rotation;
                    cv::Rodrigues(rvec, rotation);

                    Eigen::Matrix3d rotation_eigen;
                    Eigen::Vector3d translation_eigen;
                    cv::cv2eigen(rotation, rotation_eigen);
                    cv::cv2eigen(tvec, translation_eigen);

                    LOG_DEBUG("Rotation matrix:\n{}", rotation_eigen);
                    LOG_DEBUG("Translation vector: ({}, {}, {})", translation_eigen.x(), translation_eigen.y(),
                              translation_eigen.z());

                    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                    pose.block<3, 3>(0, 0) = rotation_eigen;
                    pose.block<3, 1>(0, 3) = translation_eigen;

                    core::View view = image.getViewPoint().toView();
                    view.setPose(core::Camera::Extrinsics::fromPose(pose));
                    image.setViewPoint(ViewPoint<>(translation_eigen.x(), translation_eigen.y(),
                                                   translation_eigen.z()));

                    LOG_INFO("Pose estimated for image with score {}: translation = ({}, {}, {}), rotation = \n{}",
                             image.getScore(), translation_eigen.x(), translation_eigen.y(), translation_eigen.z(),
                             rotation_eigen);
                } else {
                    LOG_WARN("solvePnPRansac failed to estimate pose for image with score {}", image.getScore());
                }
            } else {
                LOG_WARN(
                        "Insufficient inliers for pose estimation: inliers_target_points = {}, inliers_image_points = {}, inliers_object_points = {}",
                        inliers_target_points.size(), inliers_image_points.size(), inliers_object_points.size());
            }
        } else {
            LOG_WARN("Insufficient matches for pose estimation: target_points = {}, image_points = {}",
                     target_points.size(), image_points.size());
        }
    }
}*/
