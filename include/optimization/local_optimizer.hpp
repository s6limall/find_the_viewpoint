// File: optimization/local_optimizer.hpp

#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include "acquisition.hpp"
#include "cache/viewpoint_cache.hpp"
#include "evaluation/viewpoint_evaluator.hpp"
#include "local_optimizer.hpp"
#include "optimizer/levenberg_marquardt.hpp"
#include "radius_optimizer.hpp"
#include "sampling/viewpoint_sampler.hpp"
#include "spatial/octree.hpp"

namespace optimization {

    template<typename Scalar, int Dim = 3>
    class LocalOptimizer {
    public:
        using VectorType = typename LevenbergMarquardt<Scalar, Dim>::VectorType;
        using ResidualType = typename LevenbergMarquardt<Scalar, Dim>::ResidualType;
        using JacobianType = typename LevenbergMarquardt<Scalar, Dim>::JacobianType;

        explicit LocalOptimizer(const std::shared_ptr<processing::image::ImageComparator> &comparator) :
            comparator_(comparator) {}

        ViewPoint<Scalar> optimize(const ViewPoint<Scalar> &initial_viewpoint, const Image<> &target) {
            const auto residual_func = [this, &target](const VectorType &position) {
                return computeResiduals(position, target);
            };

            const auto jacobian_func = [this, &target](const VectorType &position) {
                return computeJacobian(position, target);
            };

            const auto result = lm_optimizer_.optimize(initial_viewpoint.getPosition(), residual_func, jacobian_func);

            if (result) {
                LOG_INFO("Local optimizement complete. Final error: {}, Iterations: {}, Function evals: {}, Jacobian "
                         "evals: {}",
                         result->final_error, result->iterations, result->function_evaluations,
                         result->jacobian_evaluations);
                return ViewPoint<Scalar>(result->position,
                                         -result->final_error); // Negative because we're maximizing similarity
            } else {
                LOG_WARN("Local optimizement failed. Returning initial viewpoint.");
                return initial_viewpoint;
            }
        }

    private:
        std::shared_ptr<processing::image::ImageComparator> comparator_;
        LevenbergMarquardt<Scalar, Dim> lm_optimizer_;

        ResidualType computeResiduals(const VectorType &position, const Image<> &target) const {
            ViewPoint<Scalar> viewpoint(position);
            const Image<> image = Image<>::fromViewPoint(viewpoint);
            Scalar similarity = comparator_->compare(target, image);

            // Convert similarity to residuals TODO: Adjust this
            ResidualType residuals(1);
            residuals << 1.0 - similarity; // Assuming similarity is in [0, 1]
            return residuals;
        }

        JacobianType computeJacobian(const VectorType &position, const Image<> &target) const {
            const Scalar h = std::sqrt(std::numeric_limits<Scalar>::epsilon());
            JacobianType J(1, Dim); // 1 row because we have 1 residual

            const ResidualType center_residual = computeResiduals(position, target);

            for (int i = 0; i < Dim; ++i) {
                VectorType perturbed_position = position;
                perturbed_position[i] += h;

                const ResidualType perturbed_residual = computeResiduals(perturbed_position, target);

                J(0, i) = (perturbed_residual[0] - center_residual[0]) / h;
            }

            return J;
        }
    };
} // namespace optimization

#endif // LOCAL_SEARCH_HPP
