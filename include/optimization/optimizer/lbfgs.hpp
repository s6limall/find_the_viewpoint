// File: optimization/optimizer/lbfgs.hpp

#ifndef LBFGS_OPTIMIZER_HPP
#define LBFGS_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include "common/logging/logger.hpp"

namespace optimization {

    template<typename T>
    class LBFGSOptimizer {
    public:
        using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using ObjectiveFunction = std::function<T(const VectorT &)>;
        using GradientFunction = std::function<VectorT(const VectorT &)>;

        struct Options {
            int max_iterations = 100;
            T gradient_tolerance = 1e-5;
            T step_size = 1.0;
            int memory_size = 10;
            T wolfe_c1 = 1e-4;
            T wolfe_c2 = 0.9;
        };

        explicit LBFGSOptimizer(const Options &options = Options{}) : options_(options) {}

        VectorT optimize(const VectorT &initial_guess, const ObjectiveFunction &obj_func,
                         const GradientFunction &grad_func) {
            VectorT x = initial_guess;
            VectorT grad = grad_func(x);
            T fx = obj_func(x);

            std::vector<VectorT> s_history;
            std::vector<VectorT> y_history;

            VectorT direction;
            T step_size;

            for (int iter = 0; iter < options_.max_iterations; ++iter) {
                if (grad.norm() < options_.gradient_tolerance) {
                    LOG_INFO("L-BFGS converged after {} iterations", iter);
                    break;
                }

                direction = computeSearchDirection(grad, s_history, y_history);
                step_size = lineSearch(x, direction, fx, grad, obj_func, grad_func);

                VectorT x_new = x + step_size * direction;
                VectorT grad_new = grad_func(x_new);

                VectorT s = x_new - x;
                VectorT y = grad_new - grad;

                updateHistory(s, y, s_history, y_history);

                x = x_new;
                grad = grad_new;
                fx = obj_func(x);

                LOG_DEBUG("L-BFGS iteration {}: f(x) = {}, ||grad|| = {}", iter, fx, grad.norm());
            }

            return x;
        }

    private:
        Options options_;

        VectorT computeSearchDirection(const VectorT &grad, const std::vector<VectorT> &s_history,
                                       const std::vector<VectorT> &y_history) {
            VectorT q = -grad;
            std::vector<T> alpha(s_history.size());

            for (int i = s_history.size() - 1; i >= 0; --i) {
                alpha[i] = s_history[i].dot(q) / y_history[i].dot(s_history[i]);
                q -= alpha[i] * y_history[i];
            }

            T gamma = s_history.back().dot(y_history.back()) / y_history.back().squaredNorm();
            VectorT z = gamma * q;

            for (size_t i = 0; i < s_history.size(); ++i) {
                T beta = y_history[i].dot(z) / y_history[i].dot(s_history[i]);
                z += s_history[i] * (alpha[i] - beta);
            }

            return z;
        }

        T lineSearch(const VectorT &x, const VectorT &direction, T fx, const VectorT &grad,
                     const ObjectiveFunction &obj_func, const GradientFunction &grad_func) {
            T step_size = options_.step_size;
            T fx_new = obj_func(x + step_size * direction);
            VectorT grad_new = grad_func(x + step_size * direction);

            while (!wolfeConditions(fx, fx_new, grad, grad_new, direction, step_size)) {
                step_size *= 0.5;
                fx_new = obj_func(x + step_size * direction);
                grad_new = grad_func(x + step_size * direction);
            }

            return step_size;
        }

        bool wolfeConditions(T fx, T fx_new, const VectorT &grad, const VectorT &grad_new, const VectorT &direction,
                             T step_size) {
            T directional_derivative = grad.dot(direction);
            return (fx_new <= fx + options_.wolfe_c1 * step_size * directional_derivative) &&
                   (grad_new.dot(direction) >= options_.wolfe_c2 * directional_derivative);
        }

        void updateHistory(const VectorT &s, const VectorT &y, std::vector<VectorT> &s_history,
                           std::vector<VectorT> &y_history) {
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
