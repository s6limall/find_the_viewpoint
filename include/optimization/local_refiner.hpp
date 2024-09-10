#ifndef LOCAL_REFINER_HPP
#define LOCAL_REFINER_HPP

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "optimization/optimizer/lbfgs.hpp"

namespace optimization {

    template<typename T = double, typename KernelType = DefaultKernel<T>>
    class LocalRefiner {
    public:
        explicit LocalRefiner(std::shared_ptr<processing::image::ImageComparator> comparator,
                              std::shared_ptr<GPR<T, KernelType>> gpr) :
            comparator_(std::move(comparator)), gpr_(std::move(gpr)),
            max_iterations_(config::get("optimization.local_search.max_iterations", 20)),
            initial_step_size_(config::get("optimization.local_search.initial_step_size", 0.05)),
            min_step_size_(config::get("optimization.local_search.min_step_size", 1e-6)),
            step_reduction_factor_(config::get("optimization.local_search.step_reduction_factor", 0.7)),
            improvement_threshold_(config::get("optimization.local_search.improvement_threshold", 1e-7)),
            lbfgs_optimizer_(typename LBFGSOptimizer<T>::Options()) {
            if (!comparator_ || !gpr_) {
                throw std::invalid_argument("Invalid comparator or GPR provided to LocalRefiner");
            }
        }

        ViewPoint<T> refine(const Image<> &target, const ViewPoint<T> &initial_viewpoint) {
            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();
            T radius = current_position.norm();
            T current_score = initial_viewpoint.getScore();
            T best_score = current_score;
            Eigen::Vector3<T> best_position = current_position;

            T step_size = initial_step_size_;
            int stagnant_iterations = 0;

            const std::array<Eigen::Vector3<T>, 6> directions = {
                    Eigen::Vector3<T>(1, 0, 0),  Eigen::Vector3<T>(-1, 0, 0), Eigen::Vector3<T>(0, 1, 0),
                    Eigen::Vector3<T>(0, -1, 0), Eigen::Vector3<T>(0, 0, 1),  Eigen::Vector3<T>(0, 0, -1)};

            try {
                for (int iteration = 0; iteration < max_iterations_; ++iteration) {
                    bool improved = false;

                    for (const auto &direction: directions) {
                        Eigen::Vector3<T> new_position =
                                (current_position + step_size * direction).normalized() * radius;

                        if (new_position.z() < 0)
                            continue; // Ensure upper hemisphere

                        T new_score = evaluateAndUpdate(new_position, target);

                        if (new_score > current_score) {
                            current_score = new_score;
                            current_position = new_position;
                            improved = true;

                            if (new_score > best_score) {
                                best_score = new_score;
                                best_position = new_position;
                            }

                            break; // Greedy: move to first improvement
                        }
                    }

                    if (improved) {
                        stagnant_iterations = 0;
                        step_size = std::min(step_size * 1.2, initial_step_size_);
                    } else {
                        stagnant_iterations++;
                        step_size *= step_reduction_factor_;
                    }

                    // Early stopping checks
                    if (step_size < min_step_size_ || stagnant_iterations > 3 ||
                        best_score - initial_viewpoint.getScore() < improvement_threshold_) {
                        LOG_INFO("Local refinement stopped after {} iterations. Reason: {}", iteration + 1,
                                 step_size < min_step_size_ ? "Small step size"
                                 : stagnant_iterations > 3  ? "Stagnation"
                                                            : "Insufficient improvement");
                        break;
                    }
                }

                // Final refinement using L-BFGS
                best_position = performLBFGSRefinement(target, best_position, radius);
                best_score = evaluateAndUpdate(best_position, target);

            } catch (const std::exception &e) {
                LOG_ERROR("Exception during local refinement: {}. Falling back to initial viewpoint.", e.what());
                return initial_viewpoint; // Fallback to initial viewpoint in case of any exception
            }

            return ViewPoint<T>(best_position, best_score);
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;
        std::shared_ptr<GPR<T, KernelType>> gpr_;
        int max_iterations_;
        T initial_step_size_;
        T min_step_size_;
        T step_reduction_factor_;
        T improvement_threshold_;
        LBFGSOptimizer<T> lbfgs_optimizer_;

        T evaluateAndUpdate(const Eigen::Vector3<T> &position, const Image<> &target) {
            ViewPoint<T> viewpoint(position);
            T score = comparator_->compare(target, Image<>::fromViewPoint(viewpoint));
            gpr_->update(position, score);
            return score;
        }

        Eigen::Vector3<T> performLBFGSRefinement(const Image<> &target, const Eigen::Vector3<T> &initial_position,
                                                 T radius) {
            auto objective_function = [&](const Eigen::Vector3<T> &position) -> T {
                return -evaluateAndUpdate(position.normalized() * radius, target);
            };

            auto gradient_function = [&](const Eigen::Vector3<T> &position) -> Eigen::Vector3<T> {
                return computeGradient(position.normalized() * radius, target);
            };

            Eigen::Vector3<T> optimized_position =
                    lbfgs_optimizer_.optimize(initial_position, objective_function, gradient_function);
            return optimized_position.normalized() * radius;
        }

        Eigen::Vector3<T> computeGradient(const Eigen::Vector3<T> &position, const Image<> &target) {
            const T h = 1e-5; // Step size for finite differences
            Eigen::Vector3<T> gradient;

            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3<T> pos_plus = position, pos_minus = position;
                pos_plus[i] += h;
                pos_minus[i] -= h;

                T score_plus = evaluateAndUpdate(pos_plus, target);
                T score_minus = evaluateAndUpdate(pos_minus, target);

                gradient[i] = (score_plus - score_minus) / (2 * h);
            }

            return gradient;
        }
    };

} // namespace optimization

#endif // LOCAL_REFINER_HPP
