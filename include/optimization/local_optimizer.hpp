// File: optimization/local_optimizer.hpp

#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include "acquisition.hpp"
#include "cache/viewpoint_cache.hpp"
#include "evaluation/viewpoint_evaluator.hpp"
#include "local_optimizer.hpp"
#include "radius_optimizer.hpp"
#include "sampling/viewpoint_sampler.hpp"
#include "spatial/octree.hpp"

namespace optimization {

    template<FloatingPoint T = double>
    class LocalOptimizer {
    public:
        explicit LocalOptimizer(const std::shared_ptr<processing::image::ImageComparator> &comparator) :
            comparator_(comparator) {}

        ViewPoint<T> optimize(const ViewPoint<T> &initial_viewpoint, const Image<> &target) {
            const int max_iterations = config::get("optimization.local_search.max_iterations", 50);
            const T initial_step_size = config::get("optimization.local_search.initial_step_size", 0.1);
            const T step_size_reduction_factor =
                    config::get("optimization.local_search.step_size_reduction_factor", 0.5);
            const T min_step_size = config::get("optimization.local_search.min_step_size", 1e-6);
            const T gradient_threshold = config::get("optimization.local_search.gradient_threshold", 1e-8);
            const T momentum_factor = config::get("optimization.local_search.momentum_factor", 0.9);
            const T c1 = config::get("optimization.local_search.armijo_c1", 1e-4);
            const T backtrack_factor = config::get("optimization.local_search.backtrack_factor", 0.5);

            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();
            T current_score = initial_viewpoint.getScore();
            T step_size = initial_step_size;

            Eigen::Vector3<T> momentum = Eigen::Vector3<T>::Zero();

            ViewPoint<T> best_viewpoint = initial_viewpoint;

            for (int i = 0; i < max_iterations; ++i) {
                Eigen::Vector3<T> gradient = computeGradient(current_position, target, step_size);

                if (gradient.norm() < gradient_threshold) {
                    LOG_INFO("Local optimization converged due to small gradient at iteration {}", i);
                    break;
                }

                momentum = momentum_factor * momentum + (1 - momentum_factor) * gradient;
                Eigen::Vector3<T> search_direction = momentum.normalized();

                T alpha = step_size;
                Eigen::Vector3<T> new_position;
                T new_score;

                while (true) {
                    new_position = current_position + alpha * search_direction;
                    ViewPoint<T> new_viewpoint(new_position);
                    Image<> new_image = Image<>::fromViewPoint(new_viewpoint);
                    new_score = comparator_->compare(target, new_image);

                    if (new_score > current_score + c1 * alpha * gradient.dot(search_direction)) {
                        break;
                    }

                    alpha *= backtrack_factor;
                    if (alpha < min_step_size) {
                        LOG_INFO("Local optimization stopped due to small step size at iteration {}", i);
                        return best_viewpoint;
                    }
                }

                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    best_viewpoint = ViewPoint<T>(new_position, new_score);
                    step_size /= step_size_reduction_factor;
                } else {
                    step_size *= step_size_reduction_factor;
                    if (step_size < min_step_size) {
                        LOG_INFO("Local optimization converged due to small step size at iteration {}", i);
                        break;
                    }
                }

                LOG_DEBUG("Local optimization iteration {}: score = {}, step_size = {}", i, current_score, step_size);
            }

            LOG_INFO("Local optimization complete. Final score: {}", current_score);
            return best_viewpoint;
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;

        Eigen::Vector3<T> computeGradient(const Eigen::Vector3<T> &position, const Image<> &target, T epsilon) {
            Eigen::Vector3<T> gradient;
            ViewPoint<T> center_viewpoint(position);
            Image<> center_image = Image<>::fromViewPoint(center_viewpoint);
            T center_score = comparator_->compare(target, center_image);

            for (int j = 0; j < 3; ++j) {
                Eigen::Vector3<T> perturbed_position = position;
                perturbed_position[j] += epsilon;

                ViewPoint<T> perturbed_viewpoint(perturbed_position);
                Image<> perturbed_image = Image<>::fromViewPoint(perturbed_viewpoint);
                T perturbed_score = comparator_->compare(target, perturbed_image);

                gradient[j] = (perturbed_score - center_score) / epsilon;
            }

            return gradient;
        }
    };

} // namespace optimization

#endif // LOCAL_SEARCH_HPP
