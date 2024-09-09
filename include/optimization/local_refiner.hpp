// File: optimization/local_refiner.hpp

#ifndef LOCAL_REFINER_HPP
#define LOCAL_REFINER_HPP

#include "optimization/optimizer/lbfgs.hpp"

namespace optimization {

    template<FloatingPoint T = double, IsKernel<T> KernelType = DefaultKernel<T>>
    class LocalRefiner {
    public:
        explicit LocalRefiner(std::shared_ptr<processing::image::ImageComparator> comparator,
                              std::shared_ptr<GPR<T, KernelType>> gpr) :
            comparator_(std::move(comparator)), gpr_(std::move(gpr)),
            max_iterations_(config::get("optimization.local_search.max_iterations", 20)),
            epsilon_(config::get("optimization.local_search.epsilon", 1e-5)),
            patience_(config::get("optimization.local_search.patience", 5)),
            improvement_threshold_(config::get("optimization.local_search.improvement_threshold", 1e-4)),
            lbfgs_optimizer_(typename LBFGSOptimizer<T>::Options()) {}

        ViewPoint<T> refine(const Image<> &target, const ViewPoint<T> &initial_viewpoint,
                            const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();

            // Objective function for image comparison
            auto objective_function = [&](const Eigen::Vector3<T> &position) -> T {
                ViewPoint<T> viewpoint(position);
                return comparator->compare(target, Image<>::fromViewPoint(viewpoint));
            };

            // Gradient function using finite differences
            auto gradient_function = [&](const Eigen::Vector3<T> &position) -> Eigen::Vector3<T> {
                return computeGradient(position, target, comparator);
            };

            // Run L-BFGS with early stopping criteria
            Eigen::Vector3<T> optimized_position =
                    lbfgs_optimizer_.optimize(current_position, objective_function, gradient_function);

            // Final optimized viewpoint and score
            T optimized_score = objective_function(optimized_position);
            return ViewPoint<T>(optimized_position, optimized_score);
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;
        std::shared_ptr<GPR<T, KernelType>> gpr_;
        int max_iterations_;
        T epsilon_;
        int patience_;
        T improvement_threshold_;
        LBFGSOptimizer<T> lbfgs_optimizer_;

        // Gradient computation with finite differences
        Eigen::Vector3<T> computeGradient(const Eigen::Vector3<T> &position, const Image<> &target,
                                          const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> gradient;
            T h = epsilon_ * 10;

            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3<T> pos_plus = position;
                Eigen::Vector3<T> pos_minus = position;
                pos_plus[i] += h;
                pos_minus[i] -= h;

                T score_plus = comparator->compare(target, Image<>::fromViewPoint(ViewPoint<T>(pos_plus)));
                T score_minus = comparator->compare(target, Image<>::fromViewPoint(ViewPoint<T>(pos_minus)));

                gradient[i] = (score_plus - score_minus) / (2 * h);

                // Update GPR with new observations
                gpr_->update(pos_plus, score_plus);
                gpr_->update(pos_minus, score_minus);
            }

            return gradient.normalized();
        }

        // Add adaptive convergence checks
        bool hasConverged(const T current_score, const T best_score, int iteration, const int no_improvement_count) {
            // Stagnation check
            if (no_improvement_count >= patience_) {
                LOG_INFO("Local refinement stopped due to stagnation after {} iterations.", iteration);
                return true;
            }

            // Improvement check
            T relative_improvement = (current_score - best_score) / std::max(best_score, T(1e-8));
            if (relative_improvement < improvement_threshold_) {
                LOG_INFO("Local refinement stopped due to insufficient improvement after {} iterations.", iteration);
                return true;
            }

            return false;
        }

        // Early stopping and convergence criteria for gradient-based methods
        ViewPoint<T> gradientDescentRefinement(const Image<> &target, const ViewPoint<T> &initial_viewpoint,
                                               const std::shared_ptr<processing::image::ImageComparator> &comparator) {
            Eigen::Vector3<T> current_position = initial_viewpoint.getPosition();
            Eigen::Vector3<T> velocity = Eigen::Vector3<T>::Zero();
            T current_score = initial_viewpoint.getScore();
            T best_score = current_score;
            Eigen::Vector3<T> best_position = current_position;
            int no_improvement_count = 0;
            T learning_rate = 0.01; // Initial learning rate

            for (int i = 0; i < max_iterations_; ++i) {
                Eigen::Vector3<T> gradient = computeGradient(current_position, target, comparator);
                velocity = 0.9 * velocity + 0.1 * gradient; // Momentum

                // Update position and calculate new score
                Eigen::Vector3<T> new_position = current_position - learning_rate * velocity;
                T new_score = comparator->compare(target, Image<>::fromViewPoint(ViewPoint<T>(new_position)));

                // Update GPR with the new observation
                gpr_->update(new_position, new_score);

                // Track improvement
                if (new_score > current_score) {
                    current_position = new_position;
                    current_score = new_score;
                    learning_rate *= 1.1; // Increase learning rate on improvement
                    no_improvement_count = 0;

                    if (new_score > best_score) {
                        best_position = new_position;
                        best_score = new_score;
                    }
                } else {
                    learning_rate *= 0.5; // Decrease learning rate when no improvement
                    no_improvement_count++;
                }

                // Convergence check
                if (hasConverged(current_score, best_score, i, no_improvement_count))
                    break;
            }

            return ViewPoint<T>(best_position, best_score);
        }
    };

} // namespace optimization

#endif // LOCAL_REFINER_HPP
