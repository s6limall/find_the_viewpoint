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

        // Initial sampling and evaluation
        auto samples = provider_->provision();
        auto evaluated_samples = evaluator_->evaluate(comparator_, samples);

        QuadrantFilter<Image<> > quadrant_filter;
        auto filtered_images = quadrant_filter.filter(evaluated_samples, [](const Image<> &image) {
            return image.getScore();
        });

        auto clusters = clusterSamples(filtered_images);

        // Find the best cluster
        const auto &best_cluster = *std::max_element(clusters.begin(), clusters.end(),
                                                     [](const Cluster<> &a, const Cluster<> &b) {
                                                         return a.getAverageScore() < b.getAverageScore();
                                                     });

        LOG_INFO("Best cluster: {}, Average score = {}", best_cluster.size(), best_cluster.getAverageScore());

        // Convert best cluster points to images
        const auto &best_points = best_cluster.getPoints();
        std::vector<Image<> > best_images;
        best_images.reserve(best_points.size());

        std::transform(best_points.begin(), best_points.end(), std::back_inserter(best_images),
                       [](const ViewPoint<> &point) {
                           core::View view = point.toView();
                           cv::Mat rendered_image = core::Perception::render(view.getPose());
                           Image<> image(rendered_image, extractor_);
                           image.setViewPoint(point);
                           image.setScore(point.getScore());
                           return image;
                       });

        // Match and RANSAC on the initial best images
        auto filtered_points = matchAndRansac(best_images);

        // Continue generating new points until we reach a minimum score
        const double min_score_threshold = 0.5; // Example threshold value
        std::optional<ViewPoint<> > best_viewpoint;

        while (true) {
            auto it = std::find_if(filtered_points.begin(), filtered_points.end(),
                                   [min_score_threshold](const ViewPoint<> &point) {
                                       return point.getScore() >= min_score_threshold;
                                   });

            if (it != filtered_points.end()) {
                best_viewpoint = *it;
                break;
            }

            // Generate next sample viewpoint
            auto next_sample = provider_->next();
            core::View view = next_sample.toView();
            cv::Mat rendered_image = core::Perception::render(view.getPose());

            // Create Image object for the rendered image
            Image<> next_image(rendered_image, extractor_);
            next_image.setViewPoint(next_sample);

            // Perform RANSAC on the new image
            auto next_filtered_points = matchAndRansac({next_image});
            if (next_filtered_points.empty()) {
                break;
            }

            filtered_points.insert(filtered_points.end(), next_filtered_points.begin(), next_filtered_points.end());
        }

        if (best_viewpoint) {
            const auto &viewpoint = *best_viewpoint;
            LOG_INFO("Best Viewpoint: ({}, {}, {}), Score: {}",
                     viewpoint.getPosition().x(), viewpoint.getPosition().y(), viewpoint.getPosition().z(),
                     viewpoint.getScore());

            core::Perception::render(viewpoint.toView().getPose());
        } else {
            LOG_WARN("No best viewpoint found with a sufficient score.");
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

std::vector<ViewPoint<> > Executor::matchAndRansac(const std::vector<Image<> > &images) {
    processing::image::FLANNMatcher matcher;
    std::vector<ViewPoint<> > filtered_points;

    // Get target descriptors and keypoints once
    const auto &target_descriptors = target_.getDescriptors();
    const auto &target_keypoints = target_.getKeypoints();

    for (const auto &image: images) {
        // Match features
        auto matches = matcher.match(target_descriptors, image.getDescriptors());
        LOG_DEBUG("Number of matches found: {}", matches.size());

        // Convert keypoints to points for RANSAC if enough matches are found
        if (matches.size() < 4) {
            LOG_WARN("Not enough matches for RANSAC. Matches size: {}", matches.size());
            continue;
        }

        std::vector<cv::Point2f> target_points, image_points;
        target_points.reserve(matches.size());
        image_points.reserve(matches.size());

        std::transform(matches.begin(), matches.end(), std::back_inserter(target_points), [&](const auto &match) {
            return target_keypoints[match.queryIdx].pt;
        });

        std::transform(matches.begin(), matches.end(), std::back_inserter(image_points), [&](const auto &match) {
            return image.getKeypoints()[match.trainIdx].pt;
        });

        // Perform RANSAC to filter matches
        std::vector<uchar> inliers_mask;
        cv::Mat homography = cv::findHomography(target_points, image_points, cv::RANSAC, 3.0, inliers_mask);
        LOG_DEBUG("Homography matrix: {}", homography);

        // Collect good matches based on RANSAC inliers
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());
        for (size_t i = 0; i < inliers_mask.size(); ++i) {
            if (inliers_mask[i]) {
                good_matches.push_back(matches[i]);
            }
        }

        LOG_DEBUG("Number of inliers after RANSAC: {}", good_matches.size());

        // Store the viewpoint if there are good matches
        if (!good_matches.empty()) {
            ViewPoint<> point = image.getViewPoint();
            point.setScore(static_cast<double>(good_matches.size()) / matches.size());
            filtered_points.push_back(point);
        }
    }

    return filtered_points;
}


// Unused
ViewPoint<> Executor::predictNextViewpoint(const std::vector<ViewPoint<> > &evaluated_points) {
    // Implement Bayesian Optimization to predict the next best viewpoint
    return evaluated_points.front(); // Returning the first evaluated point for now
}
