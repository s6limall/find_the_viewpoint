// File: optimization/optimizer/lbfgs.hpp

#ifndef LBFGS_OPTIMIZER_HPP
#define LBFGS_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <limits>
#include "common/logging/logger.hpp"

namespace optimization {

template<typename T>
class LBFGSOptimizer {
public:
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using ObjectiveFunction = std::function<T(const VectorT&)>;
    using GradientFunction = std::function<VectorT(const VectorT&)>;

    struct Options {
        int max_iterations = 1000;
        T gradient_tolerance = 1e-6;
        T function_tolerance = 1e-6;
        T initial_step_size = 1.0;
        int memory_size = 10;
        T wolfe_c1 = 1e-4;
        T wolfe_c2 = 0.9;
        int max_line_search_iterations = 20;
        T max_step_size = 1e6;
        T min_step_size = 1e-20;
    };

    explicit LBFGSOptimizer(const Options& options = Options{}) : options_(options) {}

    VectorT optimize(const VectorT& initial_guess, const ObjectiveFunction& obj_func,
                     const GradientFunction& grad_func) {
        VectorT x = initial_guess;
        VectorT grad = grad_func(x);
        T fx = obj_func(x);

        std::vector<VectorT> s_history;
        std::vector<VectorT> y_history;

        T previous_fx = fx;

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
            if (!std::isfinite(fx) || grad.hasNaN()) {
                LOG_WARN("L-BFGS: Non-finite function value or gradient encountered. Stopping.");
                break;
            }

            if (grad.norm() < options_.gradient_tolerance) {
                LOG_INFO("L-BFGS: Gradient norm tolerance reached after {} iterations", iteration);
                break;
            }

            if (std::abs(fx - previous_fx) < options_.function_tolerance * std::max(std::abs(fx), T(1.0))) {
                LOG_INFO("L-BFGS: Function value improvement tolerance reached after {} iterations", iteration);
                break;
            }

            VectorT direction = computeSearchDirection(grad, s_history, y_history);

            if (direction.dot(grad) >= 0) {
                LOG_WARN("L-BFGS: Search direction is not a descent direction. Resetting to steepest descent.");
                direction = -grad;
            }

            T step_size = lineSearch(x, direction, fx, grad, obj_func, grad_func);

            if (step_size < options_.min_step_size) {
                LOG_WARN("L-BFGS: Step size too small. Stopping.");
                break;
            }

            VectorT x_new = x + step_size * direction;
            VectorT grad_new = grad_func(x_new);
            T fx_new = obj_func(x_new);

            VectorT s = x_new - x;
            VectorT y = grad_new - grad;

            T sy = s.dot(y);
            if (sy > 0) {
                updateHistory(s, y, s_history, y_history);
            } else {
                LOG_WARN("L-BFGS: Skipping update due to non-positive curvature");
            }

            x = x_new;
            grad = grad_new;
            previous_fx = fx;
            fx = fx_new;

            LOG_DEBUG("L-BFGS iteration {}: f(x) = {}, ||grad|| = {}, step_size = {}",
                      iteration, fx, grad.norm(), step_size);
        }

        return x;
    }

private:
    Options options_;

    VectorT computeSearchDirection(const VectorT& grad, const std::vector<VectorT>& s_history,
                                   const std::vector<VectorT>& y_history) {
        if (s_history.empty() || y_history.empty()) {
            return -grad;
        }

        VectorT q = -grad;
        std::vector<T> alpha(s_history.size());

        for (int i = static_cast<int>(s_history.size()) - 1; i >= 0; --i) {
            T denominator = y_history[i].dot(s_history[i]);
            if (std::abs(denominator) > std::numeric_limits<T>::epsilon()) {
                alpha[i] = s_history[i].dot(q) / denominator;
                q -= alpha[i] * y_history[i];
            } else {
                LOG_WARN("L-BFGS: Division by zero avoided in direction computation");
            }
        }

        T gamma = 1.0;
        T denominator = y_history.back().squaredNorm();
        if (denominator > std::numeric_limits<T>::epsilon()) {
            gamma = s_history.back().dot(y_history.back()) / denominator;
        }
        VectorT z = gamma * q;

        for (size_t i = 0; i < s_history.size(); ++i) {
            T denominator = y_history[i].dot(s_history[i]);
            if (std::abs(denominator) > std::numeric_limits<T>::epsilon()) {
                T beta = y_history[i].dot(z) / denominator;
                z += s_history[i] * (alpha[i] - beta);
            }
        }

        return z;
    }

    T lineSearch(const VectorT& x, const VectorT& direction, T fx, const VectorT& grad,
                 const ObjectiveFunction& obj_func, const GradientFunction& grad_func) {
        T step_size = options_.initial_step_size;
        T directional_derivative = grad.dot(direction);

        auto zoom = [&](T low, T high) -> T {
            for (int i = 0; i < options_.max_line_search_iterations; ++i) {
                T step = (low + high) / 2;
                T fx_new = obj_func(x + step * direction);

                if (fx_new > fx + options_.wolfe_c1 * step * directional_derivative || fx_new >= obj_func(x + low * direction)) {
                    high = step;
                } else {
                    VectorT grad_new = grad_func(x + step * direction);
                    T directional_derivative_new = grad_new.dot(direction);

                    if (std::abs(directional_derivative_new) <= -options_.wolfe_c2 * directional_derivative) {
                        return step;
                    }

                    if (directional_derivative_new * (high - low) >= 0) {
                        high = low;
                    }

                    low = step;
                }
            }
            return (low + high) / 2;
        };

        T previous_step_size = 0;
        T previous_fx = fx;

        for (int i = 0; i < options_.max_line_search_iterations; ++i) {
            T fx_new = obj_func(x + step_size * direction);

            if (fx_new > fx + options_.wolfe_c1 * step_size * directional_derivative || (i > 0 && fx_new >= previous_fx)) {
                return zoom(previous_step_size, step_size);
            }

            VectorT grad_new = grad_func(x + step_size * direction);
            T directional_derivative_new = grad_new.dot(direction);

            if (std::abs(directional_derivative_new) <= -options_.wolfe_c2 * directional_derivative) {
                return step_size;
            }

            if (directional_derivative_new >= 0) {
                return zoom(step_size, previous_step_size);
            }

            previous_step_size = step_size;
            previous_fx = fx_new;
            step_size = std::min(options_.max_step_size, step_size * 2);
        }

        LOG_WARN("L-BFGS: Line search did not converge");
        return step_size;
    }

    void updateHistory(const VectorT& s, const VectorT& y, std::vector<VectorT>& s_history,
                       std::vector<VectorT>& y_history) {
        if (s_history.size() == options_.memory_size) {
            s_history.erase(s_history.begin());
            y_history.erase(y_history.begin());
        }
        s_history.push_back(s);
        y_history.push_back(y);
    }
};

} // namespace optimization

#endif // LBFGS_OPTIMIZER_HPP