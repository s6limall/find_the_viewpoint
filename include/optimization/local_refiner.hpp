// File: optimization/local_refiner.hpp

#ifndef LOCAL_REFINER_HPP
#define LOCAL_REFINER_HPP

#include <Eigen/Dense>
#include <memory>
#include "common/logging/logger.hpp"
#include "common/metrics/metrics_collector.hpp"
#include "processing/image/comparator.hpp"
#include "types/viewpoint.hpp"

namespace optimization {

    template<FloatingPoint T = double>
    class LocalRefiner {
    public:
        explicit LocalRefiner(std::shared_ptr<processing::image::ImageComparator> comparator) :
            comparator_(std::move(comparator)),
            max_iterations_(config::get("optimization.local_search.max_iterations", 20)),
            initial_learning_rate_(config::get("optimization.local_search.learning_rate", 0.01)),
            epsilon_(config::get("optimization.local_search.epsilon", 1e-5)),
            momentum_(config::get("optimization.local_search.momentum", 0.9)),
            patience_(config::get("optimization.local_search.patience", 5)) {}

        ViewPoint<T> refine(const Image<> &target, const ViewPoint<T> &initial_viewpoint,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();
            Eigen::Vector3<T> velocity = Eigen::Vector3<T>::Zero();
            T current_score = initial_viewpoint.getScore();
            T best_score = current_score;
            Eigen::Vector3<T> best_position = current_position;
            int no_improvement_count = 0;
            T learning_rate = initial_learning_rate_;

            for (int i = 0; i < max_iterations_; ++i) {
                Eigen::Vector3<T> gradient = computeGradient(current_position, target, comparator);

                // Update velocity using momentum
                velocity = momentum_ * velocity + (1 - momentum_) * gradient;

                // Update position
                Eigen::Vector3<T> new_position = current_position + learning_rate * velocity;

                ViewPoint<T> new_viewpoint(new_position);
                auto new_image = Image<>::fromViewPoint(new_viewpoint);
                T new_score = comparator->compare(target, new_image);

                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    learning_rate *= 1.1; // Increase learning rate on improvement
                    no_improvement_count = 0;

                    if (new_score > best_score) {
                        best_score = new_score;
                        best_position = new_position;
                    }

                    LOG_DEBUG("Local refinement improved score to {} at iteration {}", current_score, i);
                } else {
                    learning_rate *= 0.5; // Decrease learning rate on no improvement
                    no_improvement_count++;

                    if (no_improvement_count >= patience_) {
                        LOG_DEBUG("Local refinement stopped due to no improvement for {} iterations", patience_);
                        break;
                    }
                }

                // Adaptive early stopping
                if (i > 0 && (best_score - initial_viewpoint.getScore()) / initial_viewpoint.getScore() < epsilon_) {
                    LOG_DEBUG("Local refinement converged at iteration {}", i);
                    break;
                }

                metrics::recordMetrics(new_viewpoint, {{"iteration", i},
                                                       {"position_x", new_position.x()},
                                                       {"position_y", new_position.y()},
                                                       {"position_z", new_position.z()},
                                                       {"score", new_score},
                                                       {"learning_rate", learning_rate}});
            }

            return ViewPoint<T>(best_position, best_score);
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;
        int max_iterations_;
        T initial_learning_rate_;
        T epsilon_;
        T momentum_;
        int patience_;

        Eigen::Vector3<T> computeGradient(const Eigen::Vector3<T> &position, const Image<> &target,
                                          const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> gradient;
            T h = epsilon_ * 10; // Step size for finite difference

            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3<T> pos_plus = position;
                Eigen::Vector3<T> pos_minus = position;
                pos_plus[i] += h;
                pos_minus[i] -= h;

                ViewPoint<T> vp_plus(pos_plus);
                ViewPoint<T> vp_minus(pos_minus);

                T score_plus = comparator->compare(target, Image<>::fromViewPoint(vp_plus));
                T score_minus = comparator->compare(target, Image<>::fromViewPoint(vp_minus));

                gradient[i] = (score_plus - score_minus) / (2 * h);
            }

            return gradient.normalized();
        }
    };

} // namespace optimization

#endif // LOCAL_REFINER_HPP
