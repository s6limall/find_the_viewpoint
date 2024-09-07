// File: optimization/levenberg_marquardt.hpp

#ifndef LEVENBERG_MARQUARDT_HPP
#define LEVENBERG_MARQUARDT_HPP

#include <Eigen/Dense>
#include <functional>
#include <optional>
#include "common/logging/logger.hpp"

namespace optimization {

    template<typename Scalar, int Dim>
    class LevenbergMarquardt {
    public:
        using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
        using MatrixType = Eigen::Matrix<Scalar, Dim, Dim>;
        using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Dim>;
        using ResidualType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using ResidualFunction = std::function<ResidualType(const VectorType &)>;
        using JacobianFunction = std::function<JacobianType(const VectorType &)>;

        struct Options {
            Scalar initial_trust_radius;
            Scalar max_trust_radius;
            int max_iterations;
            Scalar gradient_tolerance;
            Scalar function_tolerance;
            Scalar parameter_tolerance;
            bool use_dogleg;
            bool use_geodesic_acceleration;
            Scalar finite_diff_epsilon;
            bool verbose;

            Options() {
                initial_trust_radius = config::get("optimization.lm.initial_trust_radius", Scalar(1));
                max_trust_radius = config::get("optimization.lm.max_trust_radius", Scalar(1e3));
                max_iterations = config::get("optimization.lm.max_iterations", 50);
                gradient_tolerance = config::get("optimization.lm.gradient_tolerance", Scalar(1e-8));
                function_tolerance = config::get("optimization.lm.function_tolerance", Scalar(1e-8));
                parameter_tolerance = config::get("optimization.lm.parameter_tolerance", Scalar(1e-8));
                use_dogleg = config::get("optimization.lm.use_dogleg", true);
                use_geodesic_acceleration = config::get("optimization.lm.use_geodesic_acceleration", true);
                finite_diff_epsilon = config::get("optimization.lm.finite_diff_epsilon",
                                                  std::sqrt(std::numeric_limits<Scalar>::epsilon()));
                verbose = config::get("optimization.lm.verbose", false);
            }
        };

        struct Result {
            VectorType position;
            Scalar final_error;
            int iterations{};
            int function_evaluations{};
            int jacobian_evaluations{};
            bool converged{};
            std::string message;
        };

        explicit LevenbergMarquardt(const Options &options = Options{}) : options_(validateOptions(options)) {}

        std::optional<Result> optimize(const VectorType &initial_position, const ResidualFunction &residual_func,
                                       const JacobianFunction &jacobian_func) const {
            Result result{initial_position, std::numeric_limits<Scalar>::max(), 0, 0, 0, false, ""};
            VectorType current_position = initial_position;
            Scalar trust_radius = options_.initial_trust_radius;

            ResidualType residuals = residual_func(current_position);
            ++result.function_evaluations;
            Scalar current_error = 0.5 * residuals.squaredNorm();
            JacobianType jacobian = jacobian_func(current_position);
            ++result.jacobian_evaluations;

            for (int iter = 0; iter < options_.max_iterations; ++iter) {
                const MatrixType JTJ = jacobian.transpose() * jacobian;
                const VectorType JTr = jacobian.transpose() * residuals;

                if (JTr.norm() < options_.gradient_tolerance) {
                    result.converged = true;
                    result.message = "Converged: gradient norm below tolerance";
                    break;
                }

                VectorType step;
                if (options_.use_dogleg) {
                    step = computeDoglegStep(JTJ, JTr, trust_radius);
                } else {
                    const MatrixType augmented_JTJ = JTJ + (1.0 / trust_radius) * MatrixType::Identity();
                    step = -augmented_JTJ.ldlt().solve(JTr);
                }

                if (step.norm() <
                    options_.parameter_tolerance * (current_position.norm() + options_.parameter_tolerance)) {
                    result.converged = true;
                    result.message = "Converged: step size below tolerance";
                    break;
                }

                VectorType new_position = current_position + step;
                if (options_.use_geodesic_acceleration) {
                    const VectorType acceleration =
                            computeGeodesicAcceleration(jacobian_func, jacobian, current_position);
                    new_position += 0.5 * step.dot(step) * acceleration;
                }

                const ResidualType new_residuals = residual_func(new_position);
                ++result.function_evaluations;
                const Scalar new_error = 0.5 * new_residuals.squaredNorm();

                const Scalar actual_reduction = current_error - new_error;
                const Scalar predicted_reduction = -step.dot(JTr) - 0.5 * step.dot(JTJ * step);
                const Scalar relative_error = std::abs(actual_reduction - predicted_reduction) / predicted_reduction;

                if (relative_error > 0.9) {
                    const JacobianType new_jacobian = jacobian_func(new_position);
                    ++result.jacobian_evaluations;
                    jacobian = new_jacobian;
                } else {
                    jacobian = updateJacobianBroyden(jacobian, new_residuals - residuals, step);
                }

                const Scalar rho = actual_reduction / predicted_reduction;
                if (rho > 0.75 && step.norm() >= 0.9 * trust_radius) {
                    trust_radius = std::min(2.0 * trust_radius, options_.max_trust_radius);
                } else if (rho < 0.25) {
                    trust_radius *= 0.5;
                }

                if (rho > 0) {
                    current_position = new_position;
                    residuals = new_residuals;
                    current_error = new_error;
                }

                if (std::abs(actual_reduction) < options_.function_tolerance) {
                    result.converged = true;
                    result.message = "Converged: function value change below tolerance";
                    break;
                }

                result.iterations = iter + 1;
                if (options_.verbose) {
                    LOG_DEBUG("Iteration {}: error = {}, trust radius = {}", iter, current_error, trust_radius);
                }
            }

            result.position = current_position;
            result.final_error = current_error;
            if (!result.converged && result.message.empty()) {
                result.message = "Reached maximum iterations";
            }

            if (options_.verbose) {
                LOG_DEBUG("Optimization completed: {}", result.message);
                LOG_DEBUG("Final error: {}, Iterations: {}, Function evals: {}, Jacobian evals: {}", result.final_error,
                          result.iterations, result.function_evaluations, result.jacobian_evaluations);
            }

            return result;
        }

    private:
        Options options_;

        static Options validateOptions(const Options &options) {
            Options validated = options;
            validated.initial_trust_radius = std::max(validated.initial_trust_radius, Scalar(1e-10));
            validated.max_trust_radius = std::max(validated.max_trust_radius, validated.initial_trust_radius);
            validated.max_iterations = std::max(validated.max_iterations, 1);
            validated.gradient_tolerance = std::max(validated.gradient_tolerance, Scalar(1e-15));
            validated.function_tolerance = std::max(validated.function_tolerance, Scalar(1e-15));
            validated.parameter_tolerance = std::max(validated.parameter_tolerance, Scalar(1e-15));
            return validated;
        }

        VectorType computeDoglegStep(const MatrixType &JTJ, const VectorType &JTr, Scalar trust_radius) const {
            const VectorType grad_step = -JTr.dot(JTr) / JTr.dot(JTJ * JTr) * JTr;
            if (grad_step.norm() >= trust_radius) {
                return -trust_radius * JTr.normalized();
            }

            const VectorType gn_step = JTJ.ldlt().solve(-JTr);
            if (gn_step.norm() <= trust_radius) {
                return gn_step;
            }

            const Scalar a = grad_step.dot(gn_step - grad_step);
            const Scalar b = grad_step.squaredNorm() - trust_radius * trust_radius;
            const Scalar t = (-a + std::sqrt(a * a - b * (grad_step - gn_step).squaredNorm())) /
                             (grad_step - gn_step).squaredNorm();
            return grad_step + t * (gn_step - grad_step);
        }

        VectorType computeGeodesicAcceleration(const JacobianFunction &jacobian_func, const JacobianType &jacobian,
                                               const VectorType &position) const {
            const JacobianType jacobian_plus =
                    jacobian_func(position + options_.finite_diff_epsilon * VectorType::Ones());
            const JacobianType jacobian_minus =
                    jacobian_func(position - options_.finite_diff_epsilon * VectorType::Ones());
            const MatrixType hessian = (jacobian_plus - jacobian_minus) / (2 * options_.finite_diff_epsilon);
            return -hessian * jacobian.transpose() *
                   (jacobian * jacobian.transpose())
                           .ldlt()
                           .solve(jacobian *
                                  jacobian.transpose().ldlt().solve(jacobian * hessian.transpose().diagonal()));
        }

        JacobianType updateJacobianBroyden(const JacobianType &current_jacobian, const ResidualType &residual_diff,
                                           const VectorType &step) const {
            const Scalar denom = step.dot(step);
            if (denom < std::numeric_limits<Scalar>::epsilon()) {
                return current_jacobian;
            }
            return current_jacobian + (residual_diff - current_jacobian * step) * step.transpose() / denom;
        }
    };

} // namespace optimization

#endif // LEVENBERG_MARQUARDT_HPP
