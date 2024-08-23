#ifndef VIEWPOINT_OCTREE_HPP
#define VIEWPOINT_OCTREE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <vector>

#include "acquisition.hpp"
#include "cache/viewpoint_cache.hpp"
#include "common/logging/logger.hpp"
#include "optimization/optimizer/gpr.hpp"
#include "optimization/optimizer/levenberg_marquardt.hpp"
#include "optimization/radius_refiner.hpp"
#include "optimizer/lbfgs.hpp"
#include "processing/image/comparator.hpp"
#include "types/concepts.hpp"
#include "types/viewpoint.hpp"

// Metrics
#include <chrono>
#include <fmt/color.h>
#include <opencv2/viz.hpp>

namespace viewpoint {


    // TODO: Try Normalized Cross Correlation for shape matching

    template<FloatingPoint T = double>
    class Octree {
    public:
        struct Node {
            Eigen::Vector3<T> center;
            T size;
            std::vector<ViewPoint<T>> points;
            std::array<std::unique_ptr<Node>, 8> children;
            bool explored = false;
            T max_ucb = std::numeric_limits<T>::lowest();

            Node(const Eigen::Vector3<T> &center, T size) : center(center), size(size) {}
            [[nodiscard]] bool isLeaf() const noexcept { return children[0] == nullptr; } // node without children
        };

        struct NodeScore {
            T acquisition_value;
            T distance_penalty;
            Node *node;

            NodeScore(T acquisition, T dist_penalty, Node *n) :
                acquisition_value(acquisition), distance_penalty(dist_penalty), node(n) {}

            bool operator<(const NodeScore &other) const {
                // Higher values have higher priority
                return (acquisition_value - distance_penalty) < (other.acquisition_value - other.distance_penalty);
            }
        };

        Octree(const Eigen::Vector3<T> &center, T size, T min_size, const int max_iterations,
               optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr,
               std::optional<T> radius = std::nullopt, std::optional<T> tolerance = std::nullopt) :
            root_(std::make_unique<Node>(center, size)), min_size_(min_size), max_iterations_(max_iterations),
            gpr_(gpr), patience_(config::get("optimization.patience", 10)),
            improvement_threshold_(config::get("optimization.improvement_threshold", 1e-4)), radius_(radius),
            tolerance_(tolerance), recent_scores_(), target_score_(),
            cache_(typename cache::ViewpointCache<T>::CacheConfig{}),
            acquisition_(typename optimization::Acquisition<T>::Config{}),
            current_best_viewpoint_(ViewPoint<T>(center, 0)), initial_radius_(radius.value_or(size / 2)),
            current_radius_(initial_radius_) {}

        // Add this to your Octree class
        void setCurrentRadius(T radius) {
            current_radius_ = radius;
            LOG_INFO("Octree search radius updated to {}", current_radius_);
        }


        void optimize(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                      const ViewPoint<T> &initial_best, T target_score = T(0.95)) {
            LOG_INFO("Starting optimization with target score {}", target_score);
            target_score_ = target_score;
            best_viewpoint_ = initial_best;
            current_best_viewpoint_ = initial_best;
            T best_score = initial_best.getScore();
            stagnant_iterations_ = 0;
            recent_scores_.clear();

            // Configure acquisition function
            typename optimization::Acquisition<T>::Config acquisition_config(
                    optimization::Acquisition<T>::Strategy::ADAPTIVE, // Strategy
                    2.0, // Beta
                    1.0, // Exploration weight
                    0.5, // Exploitation weight
                    0.1 // Momentum
                    // iteration_count is omitted, so it will default to 0
            );

            // Update the acquisition function with the new configuration
            acquisition_.updateConfig(acquisition_config);
            const int max_stagnant_iterations = patience_;
            for (int i = 0; i < max_iterations_; ++i) {
                if (refine(target, comparator, i)) {
                    LOG_INFO("Refinement complete at iteration {}", i);
                    break;
                }

                if (!best_viewpoint_) {
                    LOG_ERROR("Lost best viewpoint during optimization");
                    return;
                }

                T current_best_score = best_viewpoint_->getScore();

                if (current_best_score > best_score) {
                    best_score = current_best_score;
                    stagnant_iterations_ = 0;
                } else {
                    ++stagnant_iterations_;
                }

                // Perform local refinement more frequently
                if (i % 5 == 0) {
                    localRefinement(target, comparator);
                }

                if (hasConverged(current_best_score, best_score, target_score, i) ||
                    stagnant_iterations_ >= max_stagnant_iterations) {
                    LOG_INFO("Optimization converged or stagnated at iteration {}", i);
                    break;
                }

                if (i % 10 == 0) {
                    gpr_.optimizeHyperparameters();
                }
            }

            if (!best_viewpoint_) {
                LOG_ERROR("No best viewpoint found after main optimization");
                return;
            }

            LOG_INFO("Main optimization complete. Best viewpoint before radius refinement: {}",
                     best_viewpoint_->toString());

            // Perform final radius refinement
            auto refined_result = finalRadiusRefinement(target);

            if (!refined_result) {
                LOG_ERROR("Radius refinement failed");
                return;
            }

            Image<T> refined_image = Image<>::fromViewPoint(refined_result->best_viewpoint);

            T refined_score = comparator->compare(target, refined_image);
            refined_result->best_viewpoint.setScore(refined_score);
            refined_image.setScore(refined_score);
            LOG_INFO("Refined viewpoint: {}, Score: {}", refined_image.getViewPoint()->toString(), refined_score);
            LOG_INFO("Optimization complete.");
            LOG_INFO("Initial best viewpoint: {}", initial_best.toString());
            LOG_INFO("Best viewpoint after main optimization: {}", best_viewpoint_->toString());
            LOG_INFO("Final best viewpoint after radius refinement: {}", refined_result->best_viewpoint.toString());
            LOG_INFO("Initial score: {:.6f}", initial_best.getScore());
            LOG_INFO("Total score improvement: {:.6f}", refined_score - initial_best.getScore());
            LOG_INFO("Radius refinement iterations: {}", refined_result->iterations);

            if (refined_score > best_viewpoint_->getScore()) {
                best_viewpoint_ = refined_result->best_viewpoint;
                LOG_INFO("Radius refinement improved the viewpoint");
            } else {
                LOG_INFO("Radius refinement did not improve the viewpoint. Keeping the original.");
            }
        }


        [[nodiscard]] std::optional<ViewPoint<T>> getBestViewpoint() const noexcept { return best_viewpoint_; }

        void visualizeSearchProgression() const {
            if (search_history_.empty()) {
                LOG_WARN("No search history available for visualization.");
                return;
            }

            try {
                // Prepare data for visualization
                std::vector<cv::Vec3f> points;
                std::vector<cv::Vec3b> colors;
                std::vector<float> sizes;

                auto start_time = search_history_.front().timestamp;
                auto end_time = search_history_.back().timestamp;
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

                T min_score = std::numeric_limits<T>::max();
                T max_score = std::numeric_limits<T>::lowest();

                for (const auto &step: search_history_) {
                    min_score = std::min(min_score, step.score);
                    max_score = std::max(max_score, step.score);
                }

                for (const auto &step: search_history_) {
                    const auto &pos = step.viewpoint.getPosition();
                    points.emplace_back(pos.x(), pos.y(), pos.z());

                    // Color mapping: blue (cold) to red (hot) based on score
                    float normalized_score = (step.score - min_score) / (max_score - min_score);
                    cv::Vec3b color(static_cast<uchar>(255 * normalized_score), // Red
                                    0, // Green
                                    static_cast<uchar>(255 * (1 - normalized_score)) // Blue
                    );
                    colors.push_back(color);

                    // Size mapping: larger points for higher scores
                    sizes.push_back(normalized_score * 5 + 1);
                }

                // Create 3D visualization window
                cv::viz::Viz3d window("Search Progression Visualization");
                window.setBackgroundColor(cv::viz::Color::white());

                // Add coordinate system
                window.showWidget("Coordinate System", cv::viz::WCoordinateSystem());

                // Add search points
                cv::viz::WCloud cloud(points, colors);
                cloud.setRenderingProperty(cv::viz::POINT_SIZE, 5);
                window.showWidget("Search Points", cloud);

                // Add trajectory of best points
                std::vector<cv::Vec3f> best_trajectory;
                T best_score = min_score;
                for (const auto &step: search_history_) {
                    if (step.score > best_score) {
                        best_score = step.score;
                        const auto &pos = step.viewpoint.getPosition();
                        best_trajectory.emplace_back(pos.x(), pos.y(), pos.z());
                    }
                }
                cv::viz::WPolyLine best_trajectory_widget(best_trajectory, cv::viz::Color::green());
                best_trajectory_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 3);
                window.showWidget("Best Trajectory", best_trajectory_widget);

                // Add text information
                auto addText = [&window](const std::string &text, const cv::Point &pos, const cv::viz::Color &color) {
                    cv::viz::WText widget(text, pos, 20, color);
                    window.showWidget(text, widget);
                };

                addText("Start", cv::Point(20, 20), cv::viz::Color::red());
                addText("End", cv::Point(20, 50), cv::viz::Color::green());
                addText(fmt::format("Duration: {:.2f} s", duration / 1000.0), cv::Point(20, 80),
                        cv::viz::Color::black());
                addText(fmt::format("Points explored: {}", search_history_.size()), cv::Point(20, 110),
                        cv::viz::Color::black());
                addText(fmt::format("Best score: {:.4f}", max_score), cv::Point(20, 140), cv::viz::Color::black());

                // Create color legend
                cv::Mat legend(1, 256, CV_8UC3);
                for (int i = 0; i < 256; ++i) {
                    legend.at<cv::Vec3b>(0, i) = cv::Vec3b(i, 0, 255 - i);
                }
                cv::resize(legend, legend, cv::Size(200, 30), 0, 0, cv::INTER_LINEAR);
                cv::viz::WImageOverlay legend_widget(legend, cv::Rect(20, 170, 200, 30));
                window.showWidget("Color Legend", legend_widget);

                addText("Score", cv::Point(20, 220), cv::viz::Color::black());
                addText(fmt::format("{:.2f}", min_score), cv::Point(20, 250), cv::viz::Color::black());
                addText(fmt::format("{:.2f}", max_score), cv::Point(200, 250), cv::viz::Color::black());

                // Interaction loop
                LOG_INFO("Visualization ready. Use mouse to interact, press 'q' to exit.");
                while (!window.wasStopped()) {
                    window.spinOnce(1, true);
                    int key = cv::waitKey(1);
                    if (key == 'q' || key == 'Q' || key == 27) { // 'q', 'Q', or ESC
                        break;
                    }
                }

                // Generate and save a static image
                cv::Mat screenshot = window.getScreenshot();
                cv::imwrite("search_progression.png", screenshot);
                LOG_INFO("Static visualization saved as 'search_progression.png'");

                // Generate a detailed report
                std::ofstream report("search_report.txt");
                if (report.is_open()) {
                    report << fmt::format("Search Progression Report\n");
                    report << fmt::format("========================\n\n");
                    report << fmt::format("Total points explored: {}\n", search_history_.size());
                    report << fmt::format("Duration: {:.2f} seconds\n", duration / 1000.0);
                    report << fmt::format("Best score achieved: {:.4f}\n\n", max_score);

                    report << fmt::format("Score progression:\n");
                    for (size_t i = 0; i < search_history_.size(); ++i) {
                        const auto &step = search_history_[i];
                        auto step_time =
                                std::chrono::duration_cast<std::chrono::milliseconds>(step.timestamp - start_time)
                                        .count() /
                                1000.0;
                        report << fmt::format(
                                "Step {:4d}: Time: {:6.2f}s, Score: {:.4f}, Position: ({:.2f}, {:.2f}, {:.2f})\n", i,
                                step_time, step.score, step.viewpoint.getPosition().x(),
                                step.viewpoint.getPosition().y(), step.viewpoint.getPosition().z());
                    }

                    report.close();
                    LOG_INFO("Detailed search report saved as 'search_report.txt'");
                } else {
                    LOG_WARN("Unable to open file for writing detailed report.");
                }

            } catch (const cv::Exception &e) {
                LOG_ERROR("OpenCV error during visualization: {}", e.what());
            } catch (const std::exception &e) {
                LOG_ERROR("Error during visualization: {}", e.what());
            }
        }

    private:
        std::unique_ptr<Node> root_;
        T min_size_;
        int max_iterations_;
        optimization::GaussianProcessRegression<optimization::kernel::Matern52<T>> &gpr_;
        std::optional<ViewPoint<T>> best_viewpoint_;
        mutable std::mt19937 rng_{std::random_device{}()};
        int patience_;
        const T improvement_threshold_;
        std::optional<T> radius_, tolerance_;
        std::deque<T> recent_scores_;
        int stagnant_iterations_ = 0;
        static constexpr int window_size_ = 5;
        T target_score_;
        cache::ViewpointCache<T> cache_;
        mutable optimization::Acquisition<T> acquisition_;
        optimization::LBFGSOptimizer<T> lbfgs_optimizer_;
        ViewPoint<T> current_best_viewpoint_;
        T initial_radius_;
        T current_radius_;

        bool isInValidSearchSpace(const ViewPoint<T> &point) const {
            auto [center_radius, center_polar, center_azimuthal] = current_best_viewpoint_.toSpherical();

            // Calculate relative position
            Eigen::Vector3<T> relative_position = point.getPosition() - current_best_viewpoint_.getPosition();
            ViewPoint<T> relative_point(relative_position);
            auto [point_radius, point_polar, point_azimuthal] = relative_point.toSpherical();

            // Check if the point is in the upper hemisphere
            if (point_polar > M_PI / 2) {
                return false;
            }

            // Check if the point is within the current search radius
            if (point_radius > current_radius_) {
                return false;
            }

            // Check if the point is within the allowed angular range
            T delta_theta = std::abs(point_polar - center_polar);
            T delta_phi = std::min(std::abs(point_azimuthal - center_azimuthal),
                                   2 * M_PI - std::abs(point_azimuthal - center_azimuthal));

            return delta_theta <= M_PI / 3 && delta_phi <= M_PI / 2;
        }


        void localRefinement(const Image<> &target,
                             const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            if (!best_viewpoint_)
                return;

            const int max_iterations = 20;
            const T learning_rate = 0.01;
            const T epsilon = 1e-5;

            Eigen::Vector3<T> current_position = best_viewpoint_->getPosition();
            T current_score = best_viewpoint_->getScore();

            for (int i = 0; i < max_iterations; ++i) {
                Eigen::Vector3<T> gradient;
                for (int j = 0; j < 3; ++j) {
                    Eigen::Vector3<T> perturbed_position = current_position;
                    perturbed_position[j] += epsilon;

                    ViewPoint<T> perturbed_viewpoint(perturbed_position);
                    Image<> perturbed_image = Image<>::fromViewPoint(perturbed_viewpoint);
                    T perturbed_score = comparator->compare(target, perturbed_image);

                    gradient[j] = (perturbed_score - current_score) / epsilon;
                }

                Eigen::Vector3<T> new_position = current_position + learning_rate * gradient;
                ViewPoint<T> new_viewpoint(new_position);
                Image<> new_image = Image<>::fromViewPoint(new_viewpoint);
                T new_score = comparator->compare(target, new_image);

                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    updateBestViewpoint(ViewPoint<T>(new_position, new_score));
                } else {
                    break; // Stop if no improvement
                }
            }
        }

        bool refine(const Image<> &target, const std::shared_ptr<processing::image::ImageComparator> &comparator,
                    int current_iteration) {
            updateAcquisitionFunction(current_iteration);

            std::priority_queue<NodeScore> pq;
            T distance_scale = root_->size / 10.0; // Scale factor for distance penalty
            pq.emplace(root_->max_ucb, 0.0, root_.get());

            int nodes_explored = 0;
            const int min_nodes_to_explore = 5;
            T best_score_this_refinement = best_viewpoint_->getScore();

            while (!pq.empty() && (nodes_explored < min_nodes_to_explore || current_iteration < max_iterations_ / 2)) {
                auto [acquisition_value, distance_penalty, node] = pq.top();
                pq.pop();

                if (node->size < min_size_)
                    continue;

                exploreNode(*node, target, comparator, current_iteration);
                nodes_explored++;

                // Check for convergence after exploring each node
                T current_best_score = best_viewpoint_->getScore();
                if (hasConverged(current_best_score, best_score_this_refinement, target_score_, current_iteration)) {
                    LOG_INFO("Convergence detected during refinement at iteration {}", current_iteration);
                    return true;
                }

                // Update best score for this refinement step
                if (current_best_score > best_score_this_refinement) {
                    best_score_this_refinement = current_best_score;
                    LOG_INFO("Best score updated to {} at iteration {} - {} nodes explored.",
                             best_score_this_refinement, current_iteration, nodes_explored);

                    // Perform local refinement
                    localRefinement(target, comparator);
                }


                if (!node->isLeaf()) {
                    for (auto &child: node->children) {
                        if (child) {
                            T dist_to_best = (child->center - best_viewpoint_->getPosition()).norm();
                            T child_distance_penalty = std::exp(-dist_to_best / distance_scale);
                            pq.emplace(child->max_ucb, child_distance_penalty, child.get());
                        }
                    }
                }
            }

            return nodes_explored == 0;
        }

        void exploreNode(Node &node, const Image<> &target,
                         const std::shared_ptr<processing::image::ImageComparator> &comparator, int current_iteration) {
            if (node.points.empty()) {
                node.points = samplePoints(node);
                if (isWithinNode(node, current_best_viewpoint_.getPosition())) {
                    addPointsAroundBest(node);
                }
            }

            node.max_ucb = std::numeric_limits<T>::lowest();
            for (auto &point: node.points) {
                if (isInValidSearchSpace(point)) {
                    evaluatePoint(point, target, comparator, current_iteration);
                    auto [mean, std_dev] = gpr_.predict(point.getPosition());
                    T acquisition_value = computeAcquisition(point.getPosition(), mean, std_dev, current_iteration);
                    node.max_ucb = std::max(node.max_ucb, acquisition_value);
                }
            }

            if (shouldSplit(node)) {
                splitNode(node);
            }

            node.explored = true;
        }

        std::vector<ViewPoint<T>> samplePoints(const Node &node) const {
            std::vector<ViewPoint<T>> points;
            std::uniform_real_distribution<T> dist(0, 1);

            auto [center_radius, center_polar, center_azimuthal] = current_best_viewpoint_.toSpherical();

            points.reserve(10);
            for (int i = 0; i < 10; ++i) {
                // Sample in spherical coordinates relative to current best viewpoint
                T r = current_radius_ * std::sqrt(dist(rng_));
                T theta = center_polar + (dist(rng_) - 0.5) * M_PI / 3; // Restrict to ±60° from center
                T phi = center_azimuthal + (dist(rng_) - 0.5) * M_PI / 2; // Restrict to ±90° from center

                // Ensure we stay in the upper hemisphere
                theta = std::max(T(0), std::min(theta, T(M_PI / 2)));

                ViewPoint<T> new_point = ViewPoint<T>::fromSpherical(r, theta, phi);

                // Convert the relative position to absolute position
                Eigen::Vector3<T> absolute_position = current_best_viewpoint_.getPosition() + new_point.getPosition();
                points.emplace_back(ViewPoint<T>(absolute_position));
            }

            return points;
        }


        void evaluatePoint(ViewPoint<T> &point, const Image<> &target,
                           const std::shared_ptr<processing::image::ImageComparator> &comparator,
                           int current_iteration) {
            if (!point.hasScore()) {
                auto cached_score = cache_.query(point.getPosition());
                if (cached_score) {
                    point.setScore(*cached_score);
                    LOG_DEBUG("Using cached score {} for position {}", *cached_score, point.getPosition());
                } else {
                    Image<> rendered_image = Image<>::fromViewPoint(point);
                    T score = comparator->compare(target, rendered_image);
                    point.setScore(score);
                    cache_.insert(point);
                    LOG_DEBUG("Computed new score {} for position {}", score, point.getPosition());
                }

                updateBestViewpoint(point);
                gpr_.update(point.getPosition(), point.getScore());
            } else {
                // Update the cache with the existing point
                cache_.update(point);
            }
        }

        void evaluatePoints(Node &node, const Image<> &target,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator,
                            int current_iteration) {
            node.max_ucb = std::numeric_limits<T>::lowest();

            for (auto &point: node.points) {
                if (!point.hasScore()) {
                    auto cached_score = cache_.query(point.getPosition());
                    if (cached_score) {
                        point.setScore(*cached_score);
                        LOG_DEBUG("Using cached score {} for position {}", *cached_score, point.getPosition());
                    } else {
                        Image<> rendered_image = Image<>::fromViewPoint(point);
                        T score = comparator->compare(target, rendered_image);
                        point.setScore(score);
                        cache_.insert(point);
                        LOG_DEBUG("Computed new score {} for position {}", score, point.getPosition());
                    }

                    updateBestViewpoint(point);
                    gpr_.update(point.getPosition(), point.getScore());
                } else {
                    // Update the cache with the existing point
                    cache_.update(point);
                }

                auto [mean, std_dev] = gpr_.predict(point.getPosition());

                // Update acquisition function state
                acquisition_.incrementIteration();
                if (best_viewpoint_) {
                    acquisition_.updateBestPoint(best_viewpoint_->getPosition());
                }

                // Compute acquisition value using the new acquisition function
                T acquisition_value = acquisition_.compute(point.getPosition(), mean, std_dev);

                node.max_ucb = std::max(node.max_ucb, acquisition_value);

                LOG_DEBUG("Point evaluation: position={}, mean={}, std_dev={}, acquisition_value={}",
                          point.getPosition(), mean, std_dev, acquisition_value);
            }
        }

        void addPointsAroundBest(Node &node) {
            constexpr int extra_points = 6;
            const T radius = node.size * 0.1;
            std::normal_distribution<T> dist(0, radius);

            auto [best_radius, best_polar, best_azimuthal] = current_best_viewpoint_.toSpherical();

            for (int i = 0; i < extra_points; ++i) {
                T r = radius * std::abs(dist(rng_));
                T theta = best_polar + dist(rng_) / best_radius;
                T phi = best_azimuthal + dist(rng_) / (best_radius * std::sin(best_polar));

                // Ensure we stay in the upper hemisphere
                theta = std::max(T(0), std::min(theta, T(M_PI / 2)));

                ViewPoint<T> relative_point = ViewPoint<T>::fromSpherical(r, theta, phi);
                Eigen::Vector3<T> absolute_position =
                        current_best_viewpoint_.getPosition() + relative_point.getPosition();
                ViewPoint<T> new_point(absolute_position);

                if (isInValidSearchSpace(new_point)) {
                    node.points.push_back(new_point);
                }
            }
        }

        void updateSearchRadius(T new_best_score) {
            T score_improvement = (new_best_score - T(0.5)) / T(0.5); // Assuming scores are in [0, 1]
            current_radius_ = initial_radius_ * std::exp(-5 * score_improvement);
            current_radius_ = std::max(current_radius_, min_size_ / 2); // Ensure radius doesn't get too small
            LOG_INFO("Updated search radius to {} based on new best score {}", current_radius_, new_best_score);
        }

        static bool isWithinNode(const Node &node, const Eigen::Vector3<T> &position) {
            return (position - node.center).cwiseAbs().maxCoeff() <= node.size / 2;
        }

        void updateBestViewpoint(const ViewPoint<T> &point) {
            if (point.getScore() > current_best_viewpoint_.getScore()) {
                current_best_viewpoint_ = point;
                updateSearchRadius(point.getScore());
                LOG_INFO("New best viewpoint: {} with score {}", point.toString(), point.getScore());
            }
        }

        [[nodiscard]] bool shouldSplit(const Node &node) const noexcept {
            return node.size > min_size_ * T(2) &&
                   (node.points.size() > 10 ||
                    (best_viewpoint_ && (node.center - best_viewpoint_->getPosition()).norm() < node.size));
        }

        void splitNode(Node &node) {
            T child_size = node.size / T(2);

            LOG_INFO("Splitting node at {} with size {}", node.center, node.size);
            for (int i = 0; i < 8; ++i) {
                Eigen::Vector3<T> offset((i & 1) ? child_size : -child_size, (i & 2) ? child_size : -child_size,
                                         (i & 4) ? child_size : -child_size);
                node.children[i] = std::make_unique<Node>(node.center + offset * T(0.5), child_size);
            }

            for (const auto &point: node.points) {
                int index = 0;
                for (int d = 0; d < 3; ++d) {
                    if (point.getPosition()[d] > node.center[d]) {
                        index |= (1 << d);
                    }
                }
                node.children[index]->points.push_back(point);
            }

            node.points.clear();
            node.points.shrink_to_fit();
        }

        T computeAcquisition(const Eigen::Vector3<T> &x, T mean, T std_dev, int current_iteration) const {
            acquisition_.incrementIteration();
            if (best_viewpoint_) {
                acquisition_.updateBestPoint(best_viewpoint_->getPosition());
            }
            return acquisition_.compute(x, mean, std_dev);
        }

        void updateAcquisitionFunction(int current_iteration) {
            T recent_improvement_rate = calculateRecentImprovementRate();
            T current_best_score = best_viewpoint_->getScore();

            auto current_config = acquisition_.getConfig();

            const T low_score_threshold = 0.3;
            const T high_score_threshold = 0.7;

            T base_exploration_weight = std::max(0.1, 1.0 - current_iteration / static_cast<double>(max_iterations_));

            if (current_best_score < low_score_threshold || recent_improvement_rate > 0.05) {
                current_config.exploration_weight = std::min(1.0, base_exploration_weight * 1.5);
            } else if (current_best_score > high_score_threshold) {
                current_config.exploration_weight = std::max(0.1, base_exploration_weight * 0.5);
            } else {
                current_config.exploration_weight = base_exploration_weight;
            }

            current_config.exploitation_weight = 1.0 - current_config.exploration_weight;
            current_config.momentum = std::max(0.05, std::min(0.5, recent_improvement_rate * 5));

            acquisition_.updateConfig(current_config);

            LOG_DEBUG("Updated acquisition function: exploration_weight={}, exploitation_weight={}, momentum={}",
                      current_config.exploration_weight, current_config.exploitation_weight, current_config.momentum);
        }

        T calculateRecentImprovementRate() const {
            const int window_size = std::min(10, static_cast<int>(recent_scores_.size()));
            if (window_size < 2)
                return 0.0; // Not enough data

            T oldest_score = recent_scores_[recent_scores_.size() - window_size];
            T newest_score = recent_scores_.back();

            return (newest_score - oldest_score) / static_cast<T>(window_size - 1);
        }

        // Local search - precise refinement
        void patternSearch(const Eigen::Vector3<T> &center, const Image<> &target,
                           const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            const T initial_step = min_size_ * 0.1;
            T step = initial_step;
            const int max_pattern_iterations = 20;
            const T tolerance = 1e-6;

            Eigen::Vector3<T> best_point = center;
            T best_score = best_viewpoint_->getScore();

            for (int iter = 0; iter < max_pattern_iterations; ++iter) {
                bool improved = false;
                for (int dim = 0; dim < 3; ++dim) {
                    for (int direction: std::array{-1, 1}) {
                        Eigen::Vector3<T> new_point = best_point;
                        new_point[dim] += direction * step;

                        auto cached_score = cache_.query(new_point);
                        T score;
                        if (cached_score) {
                            score = *cached_score;
                            LOG_DEBUG("Using cached score {} for position {}", score, new_point);
                        } else {
                            ViewPoint<T> new_viewpoint(new_point);
                            Image<> rendered_image = Image<>::fromViewPoint(new_viewpoint);
                            score = comparator->compare(target, rendered_image);
                            new_viewpoint.setScore(score);
                            cache_.update(new_viewpoint);
                            LOG_DEBUG("Computed new score {} for position {}", score, new_point);
                        }

                        updateBestViewpoint(ViewPoint<T>(new_point, score));
                        gpr_.update(new_point, score);

                        if (score > best_score) {
                            best_point = new_point;
                            best_score = score;
                            improved = true;
                        }
                    }
                }

                if (!improved) {
                    step *= 0.5;
                }

                if (step < tolerance) {
                    break;
                }
            }
        }

        Eigen::Vector3<T> projectToShell(const Eigen::Vector3<T> &point) const {
            if (!radius_ || !tolerance_)
                return point;

            Eigen::Vector3<T> direction = point - root_->center;
            T distance = direction.norm();
            T min_radius = *radius_ * (1 - *tolerance_);
            T max_radius = *radius_ * (1 + *tolerance_);

            if (distance < min_radius) {
                return root_->center + direction.normalized() * min_radius;
            } else if (distance > max_radius) {
                return root_->center + direction.normalized() * max_radius;
            }
            return point;
        }

        bool hasConverged(T current_score, T best_score, T target_score, int current_iteration) {
            // Check if we've reached or exceeded the target score
            if (current_score >= target_score) {
                LOG_INFO("Target score reached at iteration {}", current_iteration);
                return true;
            }

            LOG_INFO("Current score: {}, Best score: {}", current_score, best_score);

            // Calculate relative improvement
            T relative_improvement = (current_score - best_score) / best_score;

            if (relative_improvement > improvement_threshold_) {
                stagnant_iterations_ = 0;
            } else {
                stagnant_iterations_++;
            }

            // Early stopping based on stagnation
            if (stagnant_iterations_ >= patience_) {
                LOG_INFO("Early stopping triggered after {} stagnant iterations", patience_);
                return true;
            }

            // Moving average convergence check
            recent_scores_.push_back(current_score);
            if (recent_scores_.size() > window_size_) {
                recent_scores_.pop_front();
            }

            if (recent_scores_.size() == window_size_) {
                // T avg_score = std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0)) / window_size_;
                T avg_score = std::reduce(recent_scores_.begin(), recent_scores_.end(), T(0)) / window_size_;
                T score_variance =
                        std::accumulate(recent_scores_.begin(), recent_scores_.end(), T(0),
                                        [avg_score](T acc, T score) { return acc + std::pow(score - avg_score, 2); }) /
                        window_size_;

                if (score_variance < T(1e-6) && avg_score > target_score * T(0.95)) {
                    LOG_INFO("Convergence detected based on moving average at iteration {}", current_iteration);
                    return true;
                }
            }

            // Check confidence interval using GPR
            auto [mean, variance] = gpr_.predict(best_viewpoint_->getPosition());
            T confidence_interval = T(1.96) * std::sqrt(variance); // 95% confidence interval

            if (mean - confidence_interval > target_score) {
                LOG_INFO("High confidence in solution at iteration {}", current_iteration);
                return true;
            }

            return false;
        }

        void levenbergMarquardtRefinement(const Image<> &target,
                                          const std::shared_ptr<processing::image::ImageComparator> &comparator,
                                          optimization::LevenbergMarquardt<T, 3> &lm_optimizer) {
            auto error_func = [&](const Eigen::Vector3<T> &position) {
                auto cached_score = cache_.query(position);
                if (cached_score) {
                    LOG_DEBUG("Using cached score {} for position {}", *cached_score, position);
                    return static_cast<T>(1.0) - *cached_score;
                }
                ViewPoint<T> viewpoint(position);
                Image<> rendered_image = Image<>::fromViewPoint(viewpoint);
                T score = comparator->compare(target, rendered_image);
                cache_.update(ViewPoint<T>(position, score));
                LOG_DEBUG("Computed new score {} for position {}", score, position);
                return static_cast<T>(1.0) - score;
            };

            auto jacobian_func = [&](const Eigen::Vector3<T> &position) {
                const T h = static_cast<T>(1e-5);
                Eigen::Matrix<T, 1, 3> J;

                for (int i = 0; i < 3; ++i) {
                    Eigen::Vector3<T> perturbed_position = position;
                    perturbed_position[i] += h;
                    J(0, i) = (error_func(perturbed_position) - error_func(position)) / h;
                }

                return J;
            };

            auto result = lm_optimizer.optimize(best_viewpoint_->getPosition(), error_func, jacobian_func);

            if (result) {
                ViewPoint<T> new_viewpoint(result->position, static_cast<T>(1.0) - result->final_error);
                updateBestViewpoint(new_viewpoint);
                gpr_.update(result->position, new_viewpoint.getScore());
            }
        }

        [[nodiscard]] std::optional<typename RadiusRefiner<T>::RefineResult>
        finalRadiusRefinement(const Image<> &target) const {
            if (!best_viewpoint_) {
                LOG_WARN("No best viewpoint found for final radius refinement");
                return std::nullopt;
            }

            auto renderFunction = [](const ViewPoint<T> &vp) { return Image<>::fromViewPoint(vp); };

            auto refiner = RadiusRefiner<T>();
            auto result = refiner.refine(*best_viewpoint_, target, renderFunction);

            LOG_INFO("Final radius refinement complete.");

            return result;
        }

        struct SearchStep {
            ViewPoint<T> viewpoint;
            T score;
            std::chrono::steady_clock::time_point timestamp;
        };
        std::vector<SearchStep> search_history_;

        void recordSearchStep(const ViewPoint<T> &viewpoint, T score) {
            search_history_.push_back({viewpoint, score, std::chrono::steady_clock::now()});
        }


    };

} // namespace viewpoint

#endif // VIEWPOINT_OCTREE_HPP
