// File: optimization/levenberg_marquardt.hpp

#ifndef LEVENBERG_MARQUARDT_HPP
#define LEVENBERG_MARQUARDT_HPP

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include "common/logging/logger.hpp"

namespace optimization {

    template<typename Scalar, int Dim>
    class LevenbergMarquardt {
    public:
        using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
        using MatrixType = Eigen::Matrix<Scalar, Dim, Dim>;
        using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Dim>; // Updated to handle non-square Jacobians
        using ErrorFunction = std::function<Scalar(const VectorType &)>;
        using JacobianFunction = std::function<JacobianType(const VectorType &)>;

        struct Options {
            Scalar initial_lambda = static_cast<Scalar>(1e-3);
            Scalar lambda_factor = static_cast<Scalar>(10.0);
            int max_iterations = 100;
            Scalar gradient_threshold = static_cast<Scalar>(1e-8);
            Scalar parameter_threshold = static_cast<Scalar>(1e-8);
            Scalar error_threshold = static_cast<Scalar>(1e-8);
            Scalar epsilon = static_cast<Scalar>(1e-6);
            bool use_geodesic_acceleration = true;

            // Experimental - not sure if this is useful at the moment
            Scalar relative_improvement_threshold = static_cast<Scalar>(1e-6);
            bool use_line_search = false;
        };

        struct Result {
            VectorType position;
            Scalar final_error;
            int iterations;
            bool converged;
        };

        explicit LevenbergMarquardt(const Options &options = Options{}) : options_{validateOptions(options)} {}

        std::optional<Result> optimize(const VectorType &initial_position, const ErrorFunction &error_func,
                                       const JacobianFunction &jacobian_func) const noexcept {
            LOG_DEBUG("Optimizing using Levenberg-Marquardt...");

            VectorType current_position = initial_position;
            Scalar lambda = options_.initial_lambda;
            Scalar prev_error = std::numeric_limits<Scalar>::max();

            Result result{current_position, static_cast<Scalar>(0.0), 0, false};

            for (int iter = 0; iter < options_.max_iterations; ++iter) {
                const Scalar current_error = error_func(current_position);
                const JacobianType jacobian = jacobian_func(current_position);

                if (std::abs(prev_error - current_error) < options_.error_threshold) {
                    result.converged = true;
                    break;
                }

                const MatrixType JTJ = jacobian.transpose() * jacobian;
                const VectorType JTe = jacobian.transpose() * VectorType::Constant(jacobian.rows(), current_error);

                const MatrixType augmented_JTJ = JTJ + lambda * MatrixType::Identity();
                const VectorType delta = augmented_JTJ.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-JTe);

                if (delta.norm() < options_.parameter_threshold || JTe.norm() < options_.gradient_threshold) {
                    result.converged = true;
                    break;
                }

                VectorType new_position = current_position + delta;

                if (options_.use_geodesic_acceleration) {
                    const VectorType acceleration =
                            computeGeodesicAcceleration(jacobian_func, jacobian, current_position);
                    new_position += 0.5 * delta.dot(delta) * acceleration;
                }

                Scalar step_size = 1.0;
                if (options_.use_line_search) {
                    step_size = lineSearch(error_func, current_position, delta, current_error);
                }

                new_position = current_position + step_size * delta;
                const Scalar new_error = error_func(new_position);

                const Scalar relative_improvement = (current_error - new_error) / current_error;
                if (relative_improvement < options_.relative_improvement_threshold) {
                    result.converged = true;
                    break;
                }

                if (new_error < current_error) {
                    current_position = new_position;
                    lambda = std::max(lambda / options_.lambda_factor, Scalar(1e-10));
                    prev_error = current_error;
                } else {
                    lambda = std::min(lambda * options_.lambda_factor, Scalar(1e10));
                }

                if (lambda > 1e9) {
                    LOG_WARN("Lambda became too large. Terminating optimization.");
                    break;
                }

                result.iterations = iter + 1;
            }

            result.position = current_position;
            result.final_error = error_func(current_position);

            LOG_DEBUG("Optimization completed in {} iterations. Final error: {}", result.iterations,
                      result.final_error);

            return result;
        }

    private:
        Options options_;

        static Options validateOptions(const Options &options) noexcept {
            Options validated = options;
            validated.initial_lambda = std::max(validated.initial_lambda, Scalar(1e-10));
            validated.lambda_factor = std::max(validated.lambda_factor, Scalar(1.1));
            validated.max_iterations = std::max(validated.max_iterations, 1);
            validated.gradient_threshold = std::max(validated.gradient_threshold, Scalar(1e-15));
            validated.parameter_threshold = std::max(validated.parameter_threshold, Scalar(1e-15));
            validated.error_threshold = std::max(validated.error_threshold, Scalar(1e-15));
            validated.epsilon = std::clamp(validated.epsilon, Scalar(1e-10), Scalar(1e-5));
            return validated;
        }

        Scalar lineSearch(const ErrorFunction &error_func, const VectorType &current_position, const VectorType &delta,
                          Scalar current_error) const {
            const Scalar c = 0.5;
            Scalar alpha = 1.0;
            const int max_line_search_iterations = 10;

            for (int i = 0; i < max_line_search_iterations; ++i) {
                const Scalar new_error = error_func(current_position + alpha * delta);
                if (new_error <= current_error - c * alpha * delta.squaredNorm()) {
                    return alpha;
                }
                alpha *= 0.5;
            }
            return alpha;
        }


        VectorType computeGeodesicAcceleration(const JacobianFunction &jacobian_func, const JacobianType &jacobian,
                                               const VectorType &position) const noexcept {
            const JacobianType jacobian_plus = jacobian_func(position + options_.epsilon * VectorType::Ones());
            const JacobianType jacobian_minus = jacobian_func(position - options_.epsilon * VectorType::Ones());

            const MatrixType hessian =
                    (jacobian_plus - jacobian_minus).template topRows<Dim>() / (2 * options_.epsilon);
            return -hessian * jacobian.transpose() * (jacobian * jacobian.transpose()).inverse();
        }
    };

} // namespace optimization

#endif // LEVENBERG_MARQUARDT_HPP
