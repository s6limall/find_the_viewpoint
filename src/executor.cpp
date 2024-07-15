// File: executor.cpp

#include "executor.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "common/logging/logger.hpp"
#include "optimization/bayesian_optimizer.hpp"
#include "optimization/gp/gaussian_process.hpp"
#include "optimization/gp/acquisition_function.hpp"
#include "processing/image/comparison/mse_comparator.hpp"
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
std::unique_ptr<processing::image::FeatureExtractor> Executor::extractor_;
double Executor::distance_;

void Executor::initialize() {
    // const auto target_image = common::io::image::readImage(config::get("paths.target_image", "target.png"));
    const auto target_image(common::io::image::readImage("../../task1/target_images/obj_000020/img.png"));
    extractor_ = FeatureExtractor::create<processing::image::SIFTExtractor>();
    target_ = Image<>(target_image, extractor_);
    distance_estimator_ = std::make_unique<processing::vision::DistanceEstimator>();
    distance_ = distance_estimator_->estimate(target_.getImage());
    provider_ = std::make_unique<viewpoint::Generator<> >(distance_);
    comparator_ = std::make_unique<processing::image::SSIMComparator>(); // TODO: SSIM
    evaluator_ = std::make_unique<viewpoint::Evaluator<> >(target_);
}


double Executor::objectiveFunction(const ViewPoint<> &point,
                                   const std::unique_ptr<processing::image::ImageComparator> &comparator,
                                   viewpoint::Evaluator<> &evaluator) {
    const std::vector points = {point};
    const auto evaluated_images = evaluator.evaluate(comparator, points);
    return evaluated_images.front().getScore();
}

void Executor::execute() {
    try {
        std::call_once(init_flag_, &Executor::initialize);

        const auto samples = provider_->provision();

        auto evaluated_samples = evaluator_->evaluate(comparator_, samples);

        QuadrantFilter<Image<> > quadrant_filter;
        const auto filtered_images = quadrant_filter.filter(evaluated_samples, [](const Image<> &image) {
            return image.getScore();
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
        const auto best_points = best_cluster.getPoints();
        std::vector<Image<> > best_images;
        best_images.reserve(best_points.size());
        for (const auto &point: best_points) {
            core::View view = point.toView();
            cv::Mat rendered_image = core::Perception::render(view.getPose());
            Image<> image(rendered_image, extractor_);
            image.setViewPoint(point);
            image.setScore(point.getScore());
            best_images.push_back(image);
        }

        const auto filtered_points = matchAndRansac(best_images, target_);
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

            core::Perception::render(best_viewpoint.toView().getPose());
            LOG_INFO("Best Image ({}) Score: {}", best_viewpoint.getPosition(), best_viewpoint.getScore());

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
            auto [x, y, z] = point.toCartesian();
            LOG_DEBUG("Point: ({}, {}, {}), Score: {}", x, y, z, point.getScore());
        }
    }

    return clusters;
}

std::vector<ViewPoint<> > Executor::matchAndRansac(const std::vector<Image<> > &images,
                                                   const Image<> &target) {
    processing::image::FLANNMatcher matcher;
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


// Unused
ViewPoint<> Executor::predictNextViewpoint(const std::vector<ViewPoint<> > &evaluated_points) {
    // Implement Bayesian Optimization to predict the next best viewpoint
    return evaluated_points.front(); // Returning the first evaluated point for now
}
