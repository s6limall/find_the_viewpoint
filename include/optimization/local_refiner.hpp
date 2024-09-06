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
            max_iterations_(config::get("optimization.local_search.max_iterations", 10)),
            learning_rate_(config::get("optimization.local_search.learning_rate", 0.01)),
            epsilon_(config::get("optimization.local_search.epsilon", 1e-5)) {}

        ViewPoint<T> refine(const Image<> &target, const ViewPoint<T> &initial_viewpoint,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();
            T current_score = initial_viewpoint.getScore();

            for (int i = 0; i < max_iterations_; ++i) {
                Eigen::Vector3<T> gradient;
                for (int j = 0; j < 3; ++j) {
                    auto perturbed_position = current_position;
                    perturbed_position[j] += epsilon_;

                    ViewPoint<T> perturbed_viewpoint(perturbed_position);
                    auto perturbed_image = Image<>::fromViewPoint(perturbed_viewpoint);
                    auto perturbed_score = comparator->compare(target, perturbed_image);

                    gradient[j] = (perturbed_score - current_score) / epsilon_;

                    // Store metrics for perturbed points during local refinement
                    metrics::recordMetrics(perturbed_viewpoint, {{"position_x", perturbed_position.x()},
                                                                 {"position_y", perturbed_position.y()},
                                                                 {"position_z", perturbed_position.z()},
                                                                 {"score", perturbed_score},
                                                                 {"refinement_iteration", i}});
                }

                auto new_position = current_position + learning_rate_ * gradient;
                ViewPoint<T> new_viewpoint(new_position);
                auto new_image = Image<>::fromViewPoint(new_viewpoint);
                auto new_score = comparator->compare(target, new_image);

                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    LOG_DEBUG("Local refinement improved score to {} at iteration {}", current_score, i);
                } else {
                    LOG_DEBUG("Local refinement stopped at iteration {} with no improvement", i);
                    break;
                }
            }

            return ViewPoint<T>(current_position, current_score);
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;
        int max_iterations_;
        T learning_rate_;
        T epsilon_;
    };

} // namespace optimization

#endif // LOCAL_REFINER_HPP
