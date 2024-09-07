// File: optimization/hyperparameter_optimizer.hpp

#ifndef HYPERPARAMETER_OPTIMIZER_HPP
#define HYPERPARAMETER_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class HyperparameterOptimizer {
    public:
        using VectorXd = Eigen::VectorXd;
        using MatrixXd = Eigen::MatrixXd;

        explicit HyperparameterOptimizer(const int max_iterations = 100, const double convergence_tol = 1e-5,
                                         const int n_restarts = 5, const double param_lower_bound = 1e-6,
                                         const double param_upper_bound = 1e3) :
            max_iterations_(max_iterations), convergence_tol_(convergence_tol), n_restarts_(n_restarts),
            param_lower_bound_(param_lower_bound), param_upper_bound_(param_upper_bound), rng_(std::random_device{}()) {
            max_iterations_ =
                    config::get("optimization.gp.kernel.hyperparameters.optimization.max_iterations", max_iterations_);
            convergence_tol_ = config::get("optimization.gp.kernel.hyperparameters.optimization.convergence_tolerance",
                                           convergence_tol_);
            n_restarts_ = config::get("optimization.gp.kernel.hyperparameters.optimization.restarts", n_restarts_);
            param_lower_bound_ = config::get("optimization.gp.kernel.hyperparameters.optimization.param_lower_bound",
                                             param_lower_bound_);
            param_upper_bound_ = config::get("optimization.gp.kernel.hyperparameters.optimization.param_upper_bound",
                                             param_upper_bound_);
            LOG_INFO("HyperparameterOptimizer initialized with max_iterations={}, convergence_tol={}, n_restarts={}, "
                     "param_lower_bound={}, param_upper_bound={}",
                     max_iterations_, convergence_tol_, n_restarts_, param_lower_bound_, param_upper_bound_);
        }

        VectorXd optimize(const MatrixXd &X, const VectorXd &y, Kernel &kernel) {
            LOG_INFO("Starting hyperparameter optimization");
            if (X.rows() != y.rows() || X.rows() == 0) {
                LOG_ERROR("Invalid input dimensions: X rows: {}, y rows: {}", X.rows(), y.rows());
                return kernel.getParameters();
            }

            VectorXd best_params = kernel.getParameters();
            double best_nlml = std::numeric_limits<double>::infinity();
            double initial_nlml = computeNLML(X, y, kernel, best_params);
            LOG_INFO("Initial NLML: {}, Initial parameters: {}", initial_nlml, best_params.transpose());

            for (int restart = 0; restart < n_restarts_; ++restart) {
                LOG_INFO("Starting optimization restart {}/{}", restart + 1, n_restarts_);
                VectorXd initial_params =
                        (restart == 0) ? best_params : generateIntelligentInitialParams(X, y, kernel, restart);
                LOG_DEBUG("Initial parameters for restart {}: {}", restart + 1, initial_params.transpose());

                VectorXd optimized_params = optimizeFromInitial(X, y, kernel, initial_params);
                const double nlml = computeNLML(X, y, kernel, optimized_params);

                LOG_INFO("Restart {} completed. NLML: {}, Parameters: {}", restart + 1, nlml,
                         optimized_params.transpose());

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = optimized_params;
                    LOG_INFO("New best NLML found: {}", best_nlml);
                }
            }

            setKernelParameters(kernel, best_params);
            updateOptimizationStats(n_restarts_ * max_iterations_, best_nlml, initial_nlml, kernel.getParameters(),
                                    best_params);
            LOG_INFO("Hyperparameter optimization completed. Best NLML: {}, Best parameters: {}", best_nlml,
                     best_params.transpose());
            return best_params;
        }

        VectorXd optimizeBounded(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &lower_bounds,
                                 const VectorXd &upper_bounds) {
            LOG_INFO("Starting bounded hyperparameter optimization");
            if (X.rows() != y.rows() || X.rows() == 0) {
                LOG_ERROR("Invalid input dimensions: X rows: {}, y rows: {}", X.rows(), y.rows());
                return kernel.getParameters();
            }

            VectorXd best_params = kernel.getParameters();
            double best_nlml = std::numeric_limits<double>::infinity();
            double initial_nlml = computeNLML(X, y, kernel, best_params);
            LOG_INFO("Initial NLML: {}, Initial parameters: {}", initial_nlml, best_params);

            for (int restart = 0; restart < n_restarts_; ++restart) {
                LOG_INFO("Starting optimization restart {}/{}", restart + 1, n_restarts_);
                VectorXd initial_params =
                        (restart == 0) ? best_params : generateIntelligentInitialParams(X, y, kernel, restart);
                initial_params = initial_params.cwiseMax(lower_bounds).cwiseMin(upper_bounds);
                LOG_DEBUG("Initial parameters for restart {}: {}", restart + 1, initial_params);

                VectorXd optimized_params = optimizeFromInitial(X, y, kernel, initial_params);
                optimized_params = optimized_params.cwiseMax(lower_bounds).cwiseMin(upper_bounds);
                const double nlml = computeNLML(X, y, kernel, optimized_params);

                LOG_INFO("Restart {} completed. NLML: {}, Parameters: {}", restart + 1, nlml, optimized_params);

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = optimized_params;
                    LOG_INFO("New best NLML found: {}", best_nlml);
                }
            }

            setKernelParameters(kernel, best_params);
            updateOptimizationStats(n_restarts_ * max_iterations_, best_nlml, initial_nlml, kernel.getParameters(),
                                    best_params);
            LOG_INFO("Hyperparameter optimization completed. Best NLML: {}, Best parameters: {}", best_nlml,
                     best_params);
            return best_params;
        }

        struct OptimizationStats {
            int iterations;
            double final_nlml;
            double initial_nlml;
            VectorXd initial_params;
            VectorXd final_params;
        };

        OptimizationStats getLastOptimizationStats() const { return last_optimization_stats_; }

    private:
        int max_iterations_;
        double convergence_tol_;
        int n_restarts_;
        double param_lower_bound_;
        double param_upper_bound_;
        std::mt19937 rng_;
        OptimizationStats last_optimization_stats_;

        VectorXd optimizeFromInitial(const MatrixXd &X, const VectorXd &y, Kernel &kernel,
                                     const VectorXd &initial_params) {
            LOG_DEBUG("Starting optimization from initial parameters: {}", initial_params);
            VectorXd params = initial_params;
            VectorXd best_params = params;
            double best_nlml = std::numeric_limits<double>::infinity();

            std::vector<VectorXd> s, y_lbfgs;

            for (int iter = 0; iter < max_iterations_; ++iter) {
                VectorXd grad;
                double nlml = computeNLML(X, y, kernel, params, &grad);
                LOG_TRACE("Iteration {}: NLML = {}, Gradient norm = {}", iter, nlml, grad.norm());

                if (!std::isfinite(nlml)) {
                    LOG_WARN("Non-finite NLML encountered at iteration {}. Using fallback strategy.", iter);
                    return fallbackOptimization(X, y, kernel, best_params);
                }

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = params;
                    LOG_DEBUG("New best NLML at iteration {}: {}", iter, best_nlml);
                }

                if (grad.norm() < convergence_tol_) {
                    LOG_INFO("Convergence reached at iteration {} with gradient norm {}", iter, grad.norm());
                    break;
                }

                VectorXd direction = computeLBFGSDirection(grad, s, y_lbfgs);
                if (direction.norm() < std::numeric_limits<double>::epsilon()) {
                    LOG_WARN("Zero direction vector at iteration {}. Trying alternative optimization strategy.", iter);
                    return alternativeOptimizationStrategy(X, y, kernel, params);
                }

                double step_size = adaptiveLineSearch(X, y, kernel, params, direction, grad, nlml);
                LOG_TRACE("Step size for iteration {}: {}", iter, step_size);

                if (step_size <= std::numeric_limits<double>::epsilon()) {
                    LOG_WARN("Optimization stopped at iteration {}: step size is effectively zero", iter);
                    break;
                }

                VectorXd new_params = constrainParameters(params + step_size * direction);
                VectorXd s_k = new_params - params;
                VectorXd y_k = computeGradient(X, y, kernel, new_params) - grad;

                if (s_k.norm() > std::numeric_limits<double>::epsilon() &&
                    y_k.norm() > std::numeric_limits<double>::epsilon()) {
                    s.push_back(s_k);
                    y_lbfgs.push_back(y_k);

                    if (s.size() > 10) {
                        s.erase(s.begin());
                        y_lbfgs.erase(y_lbfgs.begin());
                    }
                    LOG_TRACE("Updated L-BFGS vectors. Current size: {}", s.size());
                }

                params = new_params;
                LOG_DEBUG("Updated parameters: {}", params);
            }

            return best_params;
        }

        VectorXd generateIntelligentInitialParams(const MatrixXd &X, const VectorXd &y, const Kernel &kernel,
                                                  int restart) {
            VectorXd current_params = kernel.getParameters();
            LOG_DEBUG("Generating intelligent initial parameters for restart {}", restart);

            if (restart % 3 == 0) {
                LOG_DEBUG("Using perturbed current parameters strategy");
                std::normal_distribution<double> perturbation(0.0, 0.1);
                for (int i = 0; i < current_params.size(); ++i) {
                    current_params(i) *= std::exp(perturbation(rng_));
                }
            } else if (restart % 3 == 1) {
                LOG_DEBUG("Using data statistics strategy");
                current_params(0) = X.colwise().mean().norm();
                current_params(1) = y.array().square().mean();
                current_params(2) = y.array().square().mean() * 0.01;
            } else {
                LOG_DEBUG("Using random sampling strategy");
                current_params = generateRandomParams();
            }

            VectorXd constrained_params = constrainParameters(current_params);
            LOG_DEBUG("Generated initial parameters: {}", constrained_params);
            return constrained_params;
        }

        VectorXd generateRandomParams() {
            std::uniform_real_distribution<double> dist(std::log(param_lower_bound_), std::log(param_upper_bound_));
            VectorXd params(3);
            for (int i = 0; i < 3; ++i) {
                params(i) = std::exp(dist(rng_));
            }
            LOG_TRACE("Generated random parameters: {}", params);
            return params;
        }

        VectorXd constrainParameters(const VectorXd &params) const {
            VectorXd constrained = params.cwiseMax(param_lower_bound_).cwiseMin(param_upper_bound_);
            LOG_TRACE("Constrained parameters: {} -> {}", params, constrained);
            return constrained;
        }

        void setKernelParameters(Kernel &kernel, const VectorXd &params) {
            try {
                kernel.setParameters(params(0), params(1), params(2));
                LOG_DEBUG("Kernel parameters set successfully: {}", params);
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to set kernel parameters: {}. Using constrained parameters.", e.what());
                VectorXd constrained_params = constrainParameters(params);
                kernel.setParameters(constrained_params(0), constrained_params(1), constrained_params(2));
                LOG_DEBUG("Constrained kernel parameters set: {}", constrained_params);
            }
        }

        double computeNLML(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                           VectorXd *grad = nullptr) {
            setKernelParameters(kernel, params);
            MatrixXd K = kernel.computeGramMatrix(X);
            Eigen::LLT<MatrixXd> llt(K);

            if (llt.info() != Eigen::Success) {
                LOG_WARN("Cholesky decomposition failed. Attempting to handle failure.");
                return handleCholeskyFailure(X, y, K, kernel, params, grad);
            }

            VectorXd alpha = llt.solve(y);
            double nlml = 0.5 * y.dot(alpha) + 0.5 * llt.matrixLLT().diagonal().array().log().sum() +
                          0.5 * X.rows() * std::log(2 * M_PI);

            if (grad) {
                computeNLMLGradient(X, y, K, alpha, kernel, params, grad);
            }

            LOG_TRACE("Computed NLML: {}", nlml);
            return nlml;
        }

        double handleCholeskyFailure(const MatrixXd &X, const VectorXd &y, MatrixXd &K, Kernel &kernel,
                                     const VectorXd &params, VectorXd *grad) {
            LOG_WARN("Handling Cholesky decomposition failure");
            double jitter = 1e-9;
            Eigen::LLT<MatrixXd> llt;

            while (jitter < 1e-3) {
                K.diagonal().array() += jitter;
                llt.compute(K);
                if (llt.info() == Eigen::Success) {
                    LOG_INFO("Cholesky decomposition succeeded with jitter = {}", jitter);
                    VectorXd alpha = llt.solve(y);
                    double nlml = 0.5 * y.dot(alpha) + 0.5 * llt.matrixLLT().diagonal().array().log().sum() +
                                  0.5 * X.rows() * std::log(2 * M_PI);

                    if (grad) {
                        computeNLMLGradient(X, y, K, alpha, kernel, params, grad);
                    }

                    return nlml;
                }
                jitter *= 10;
            }

            LOG_ERROR("Cholesky decomposition failed even with maximum jitter.");
            return std::numeric_limits<double>::infinity();
        }


        void computeNLMLGradient(const MatrixXd &X, const VectorXd &y, const MatrixXd &K, const VectorXd &alpha,
                                 Kernel &kernel, const VectorXd &params, VectorXd *grad) {
            MatrixXd K_inv = K.ldlt().solve(MatrixXd::Identity(X.rows(), X.rows()));
            MatrixXd alpha_alpha_t = alpha * alpha.transpose();
            MatrixXd factor = alpha_alpha_t - K_inv;

            grad->resize(params.size());
            for (int i = 0; i < params.size(); ++i) {
                MatrixXd K_grad = kernel.computeGradientMatrix(X, i);
                (*grad)(i) = 0.5 * (factor.array() * K_grad.array()).sum();
            }
            LOG_TRACE("Computed NLML gradient: {}", grad->transpose());
        }

        VectorXd computeGradient(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params) {
            VectorXd grad;
            computeNLML(X, y, kernel, params, &grad);
            return grad;
        }

        static VectorXd computeLBFGSDirection(const VectorXd &grad, const std::vector<VectorXd> &s,
                                              const std::vector<VectorXd> &y) {
            if (s.empty() || y.empty() || s.size() != y.size()) {
                LOG_DEBUG("Using negative gradient as L-BFGS direction");
                return -grad;
            }

            VectorXd q = -grad;
            std::vector<double> alpha(s.size());

            for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
                double rho_i = 1.0 / y[i].dot(s[i]);
                if (!std::isfinite(rho_i)) {
                    LOG_WARN("Non-finite rho_i encountered in L-BFGS. Skipping this correction pair.");
                    continue;
                }
                alpha[i] = rho_i * s[i].dot(q);
                q -= alpha[i] * y[i];
            }

            VectorXd z = q;
            if (!y.empty()) {
                double scale = y.back().dot(s.back()) / y.back().squaredNorm();
                z *= scale;
                LOG_TRACE("L-BFGS scaling factor: {}", scale);
            }

            for (size_t i = 0; i < s.size(); ++i) {
                double rho_i = 1.0 / y[i].dot(s[i]);
                if (!std::isfinite(rho_i)) {
                    continue;
                }
                double beta = rho_i * y[i].dot(z);
                z += s[i] * (alpha[i] - beta);
            }

            LOG_DEBUG("Computed L-BFGS direction. Direction norm: {}", z.norm());
            return z;
        }

        double adaptiveLineSearch(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                                  const VectorXd &direction, const VectorXd &grad, const double current_nlml) {
            LOG_DEBUG("Starting adaptive line search");
            double alpha = 1.0;
            const double c1 = 1e-4;
            const double c2 = 0.9;
            const double initial_alpha = 1.0;

            const double phi_0 = current_nlml;
            const double dphi_0 = grad.dot(direction);

            auto zoom = [&](double alpha_lo, double alpha_hi) -> double {
                LOG_TRACE("Entering zoom phase with alpha_lo = {} and alpha_hi = {}", alpha_lo, alpha_hi);
                for (int j = 0; j < 10; ++j) {
                    double alpha_j = 0.5 * (alpha_lo + alpha_hi);
                    VectorXd new_params = constrainParameters(params + alpha_j * direction);
                    double phi_j = computeNLML(X, y, kernel, new_params);

                    if (phi_j > phi_0 + c1 * alpha_j * dphi_0 ||
                        phi_j >= computeNLML(X, y, kernel, constrainParameters(params + alpha_lo * direction))) {
                        alpha_hi = alpha_j;
                    } else {
                        VectorXd grad_j;
                        computeNLML(X, y, kernel, new_params, &grad_j);
                        double dphi_j = grad_j.dot(direction);

                        if (std::abs(dphi_j) <= -c2 * dphi_0) {
                            LOG_TRACE("Zoom phase converged with alpha = {}", alpha_j);
                            return alpha_j;
                        }

                        if (dphi_j * (alpha_hi - alpha_lo) >= 0) {
                            alpha_hi = alpha_lo;
                        }

                        alpha_lo = alpha_j;
                    }
                    LOG_TRACE("Zoom iteration {}: alpha_lo = {}, alpha_hi = {}", j, alpha_lo, alpha_hi);
                }
                LOG_WARN("Zoom phase did not converge. Returning alpha_lo = {}", alpha_lo);
                return alpha_lo;
            };

            double phi_prev = phi_0;
            for (int i = 0; i < 10; ++i) {
                VectorXd new_params = constrainParameters(params + alpha * direction);
                double phi = computeNLML(X, y, kernel, new_params);

                LOG_TRACE("Line search iteration {}: alpha = {}, phi = {}", i, alpha, phi);

                if (phi > phi_0 + c1 * alpha * dphi_0 || (i > 0 && phi >= phi_prev)) {
                    LOG_DEBUG("Entering zoom phase from alpha = {}", alpha);
                    return zoom(alpha / 2, alpha);
                }

                VectorXd grad_new;
                computeNLML(X, y, kernel, new_params, &grad_new);
                double dphi = grad_new.dot(direction);

                if (std::abs(dphi) <= -c2 * dphi_0) {
                    LOG_DEBUG("Line search converged with alpha = {}", alpha);
                    return alpha;
                }

                if (dphi >= 0) {
                    LOG_DEBUG("Entering zoom phase (reverse direction) from alpha = {}", alpha);
                    return zoom(alpha, alpha / 2);
                }

                alpha *= 2;
                phi_prev = phi;
            }

            LOG_WARN("Line search did not converge. Using initial step size.");
            return initial_alpha;
        }

        VectorXd fallbackOptimization(const MatrixXd &X, const VectorXd &y, Kernel &kernel,
                                      const VectorXd &initial_params) {
            LOG_INFO("Entering fallback optimization strategy");
            VectorXd best_params = initial_params;
            double best_nlml = std::numeric_limits<double>::infinity();

            const int num_samples = 100;
            for (int i = 0; i < num_samples; ++i) {
                VectorXd params = generateRandomParams();
                double nlml = computeNLML(X, y, kernel, params);

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = params;
                    LOG_DEBUG("Fallback optimization: New best NLML = {} at iteration {}", best_nlml, i);
                }
            }

            LOG_INFO("Fallback optimization complete. Best NLML = {}", best_nlml);
            return best_params;
        }

        VectorXd alternativeOptimizationStrategy(const MatrixXd &X, const VectorXd &y, Kernel &kernel,
                                                 const VectorXd &initial_params) {
            LOG_INFO("Entering alternative optimization strategy");
            VectorXd best_params = initial_params;
            double best_nlml = computeNLML(X, y, kernel, best_params);

            const int num_iterations = 50;
            const double step_size = 0.1;

            for (int i = 0; i < num_iterations; ++i) {
                for (int j = 0; j < best_params.size(); ++j) {
                    VectorXd test_params = best_params;
                    test_params(j) *= (1 + step_size);
                    double nlml = computeNLML(X, y, kernel, test_params);

                    if (nlml < best_nlml) {
                        best_nlml = nlml;
                        best_params = test_params;
                        LOG_DEBUG("Alternative strategy: New best NLML = {} at iteration {}", best_nlml, i);
                    } else {
                        test_params(j) *= (1 - 2 * step_size) / (1 + step_size);
                        nlml = computeNLML(X, y, kernel, test_params);

                        if (nlml < best_nlml) {
                            best_nlml = nlml;
                            best_params = test_params;
                            LOG_DEBUG("Alternative strategy: New best NLML = {} at iteration {}", best_nlml, i);
                        }
                    }
                }
                LOG_TRACE("Alternative strategy iteration {}: Best NLML = {}", i, best_nlml);
            }

            LOG_INFO("Alternative optimization strategy complete. Best NLML = {}", best_nlml);
            return best_params;
        }

        void updateOptimizationStats(int iterations, double final_nlml, double initial_nlml,
                                     const VectorXd &initial_params, const VectorXd &final_params) {
            last_optimization_stats_ = {iterations, final_nlml, initial_nlml, initial_params, final_params};
            LOG_INFO("Optimization stats updated: iterations={}, initial_nlml={}, final_nlml={}", iterations,
                     initial_nlml, final_nlml);
        }
    };

} // namespace optimization

#endif // HYPERPARAMETER_OPTIMIZER_HPP

/*#ifndef HYPERPARAMETER_OPTIMIZER_HPP
#define HYPERPARAMETER_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/gaussian/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class HyperparameterOptimizer {
    public:
        using VectorXd = Eigen::VectorXd;
        using MatrixXd = Eigen::MatrixXd;

        explicit HyperparameterOptimizer(const int max_iterations = 100, const double convergence_tol = 1e-5,
                                         const int n_restarts = 5, const double param_lower_bound = 1e-6,
                                         const double param_upper_bound = 1e3) :
            max_iterations_(max_iterations), convergence_tol_(convergence_tol), n_restarts_(n_restarts),
            param_lower_bound_(param_lower_bound), param_upper_bound_(param_upper_bound), rng_(std::random_device{}()) {
            max_iterations_ =
                    config::get("optimization.gp.kernel.hyperparameters.optimization.max_iterations", max_iterations_);
            convergence_tol_ = config::get("optimization.gp.kernel.hyperparameters.optimization.convergence_tolerance",
                                           convergence_tol_);
            n_restarts_ = config::get("optimization.gp.kernel.hyperparameters.optimization.restarts", n_restarts_);
            param_lower_bound_ = config::get("optimization.gp.kernel.hyperparameters.optimization.param_lower_bound",
                                             param_lower_bound_);
            param_upper_bound_ = config::get("optimization.gp.kernel.hyperparameters.optimization.param_upper_bound",
                                             param_upper_bound_);
        }

        VectorXd optimize(const MatrixXd &X, const VectorXd &y, Kernel &kernel) {
            if (X.rows() != y.rows() || X.rows() == 0) {
                LOG_ERROR("Invalid input dimensions: X rows: {}, y rows: {}", X.rows(), y.rows());
                return kernel.getParameters();
            }

            VectorXd best_params = kernel.getParameters();
            double best_nlml = std::numeric_limits<double>::infinity();

            for (int restart = 0; restart < n_restarts_; ++restart) {
                VectorXd initial_params = (restart == 0) ? best_params : generateRandomParams();
                VectorXd optimized_params = optimizeFromInitial(X, y, kernel, initial_params);

                const double nlml = computeNLML(X, y, kernel, optimized_params);

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = optimized_params;
                }
            }

            setKernelParameters(kernel, best_params);
            return best_params;
        }

    private:
        int max_iterations_;
        double convergence_tol_;
        int n_restarts_;
        double param_lower_bound_;
        double param_upper_bound_;
        std::mt19937 rng_;

        VectorXd optimizeFromInitial(const MatrixXd &X, const VectorXd &y, Kernel &kernel,
                                     const VectorXd &initial_params) {
            VectorXd params = initial_params;
            VectorXd best_params = params;
            double best_nlml = std::numeric_limits<double>::infinity();

            std::vector<VectorXd> s, y_lbfgs;

            for (int iter = 0; iter < max_iterations_; ++iter) {
                VectorXd grad;
                double nlml = computeNLML(X, y, kernel, params, &grad);

                if (!std::isfinite(nlml)) {
                    LOG_WARN("Non-finite NLML encountered. Using best parameters so far.");
                    return best_params;
                }

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = params;
                }

                if (grad.norm() < convergence_tol_) {
                    LOG_INFO("Hyperparameter optimization converged after {} iterations", iter);
                    break;
                }

                VectorXd direction = computeLBFGSDirection(grad, s, y_lbfgs);
                if (direction.norm() < std::numeric_limits<double>::epsilon()) {
                    LOG_WARN("Zero direction vector. Stopping optimization.");
                    break;
                }

                double step_size = lineSearch(X, y, kernel, params, direction, grad, nlml);
                if (step_size <= std::numeric_limits<double>::epsilon()) {
                    LOG_WARN("Optimization stopped: step size is effectively zero");
                    break;
                }

                VectorXd new_params = constrainParameters(params + step_size * direction);
                VectorXd s_k = new_params - params;
                VectorXd y_k = computeGradient(X, y, kernel, new_params) - grad;

                if (s_k.norm() > std::numeric_limits<double>::epsilon() &&
                    y_k.norm() > std::numeric_limits<double>::epsilon()) {
                    s.push_back(s_k);
                    y_lbfgs.push_back(y_k);

                    if (s.size() > 10) {
                        s.erase(s.begin());
                        y_lbfgs.erase(y_lbfgs.begin());
                    }
                }

                params = new_params;
            }

            return best_params;
        }

        VectorXd generateRandomParams() {
            std::uniform_real_distribution<double> dist(std::log(param_lower_bound_), std::log(param_upper_bound_));
            VectorXd params(3);
            for (int i = 0; i < 3; ++i) {
                params(i) = std::exp(dist(rng_));
            }
            return params;
        }

        VectorXd constrainParameters(const VectorXd &params) const {
            return params.cwiseMax(param_lower_bound_).cwiseMin(param_upper_bound_);
        }

        void setKernelParameters(Kernel &kernel, const VectorXd &params) {
            try {
                kernel.setParameters(params(0), params(1), params(2));
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to set kernel parameters: {}. Using constrained parameters.", e.what());
                VectorXd constrained_params = constrainParameters(params);
                kernel.setParameters(constrained_params(0), constrained_params(1), constrained_params(2));
            }
        }

        // Compute the Negative Log Marginal Likelihood (NLML) and its gradient
        // NLML = 1/2 * y^T * K^(-1) * y + 1/2 * log|K| + n/2 * log(2π)
        double computeNLML(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                           VectorXd *grad = nullptr) {
            setKernelParameters(kernel, params);
            MatrixXd K = kernel.computeGramMatrix(X);
            Eigen::LLT<MatrixXd> llt(K);

            if (llt.info() != Eigen::Success) {
                LOG_WARN("Cholesky decomposition failed. Adding jitter to the diagonal.");
                double jitter = 1e-9;
                while (llt.info() != Eigen::Success && jitter < 1e-3) {
                    K.diagonal().array() += jitter;
                    llt.compute(K);
                    jitter *= 10;
                }
                if (llt.info() != Eigen::Success) {
                    LOG_ERROR("Cholesky decomposition failed even with jitter.");
                    return std::numeric_limits<double>::infinity();
                }
            }

            VectorXd alpha = llt.solve(y);
            double nlml = 0.5 * y.dot(alpha) + 0.5 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum() +
                          0.5 * X.rows() * std::log(2 * M_PI);

            if (grad) {
                MatrixXd K_inv = llt.solve(MatrixXd::Identity(X.rows(), X.rows()));
                MatrixXd alpha_alpha_t = alpha * alpha.transpose();
                MatrixXd factor = alpha_alpha_t - K_inv;

                grad->resize(params.size());
                for (int i = 0; i < params.size(); ++i) {
                    MatrixXd K_grad = kernel.computeGradientMatrix(X, i);
                    (*grad)(i) = 0.5 * (factor.array() * K_grad.array()).sum();
                }
            }

            return nlml;
        }

        VectorXd computeGradient(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params) {
            VectorXd grad;
            computeNLML(X, y, kernel, params, &grad);
            return grad;
        }

        // Compute the L-BFGS direction for optimization
        // This method uses the limited-memory BFGS algorithm to approximate the inverse Hessian
        static VectorXd computeLBFGSDirection(const VectorXd &grad, const std::vector<VectorXd> &s,
                                              const std::vector<VectorXd> &y) {
            if (s.empty() || y.empty() || s.size() != y.size()) {
                return -grad;
            }

            VectorXd q = -grad;
            std::vector<double> alpha(s.size());

            for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
                const double denominator = y[i].dot(s[i]);
                if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
                    continue;
                }
                alpha[i] = s[i].dot(q) / denominator;
                q -= alpha[i] * y[i];
            }

            double scale = 1.0;
            double denominator = y.back().squaredNorm();
            if (denominator > std::numeric_limits<double>::epsilon()) {
                scale = s.back().dot(y.back()) / denominator;
            }
            VectorXd z = q * scale;

            for (size_t i = 0; i < s.size(); ++i) {
                double denominator = y[i].dot(s[i]);
                if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
                    continue;
                }
                double beta = y[i].dot(z) / denominator;
                z += s[i] * (alpha[i] - beta);
            }

            return z;
        }

        // Perform a backtracking line search to find an appropriate step size
        // This method uses the Armijo condition to ensure sufficient decrease in the objective function
        double lineSearch(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                          const VectorXd &direction, const VectorXd &grad, const double current_nlml) {
            double alpha = 1.0;
            const double c = 0.5;
            const double rho = 0.5;

            const double phi_0 = current_nlml;
            const double dphi_0 = grad.dot(direction);

            for (int i = 0; i < 10; ++i) {
                VectorXd new_params = constrainParameters(params + alpha * direction);
                const double phi = computeNLML(X, y, kernel, new_params);

                if (phi <= phi_0 + c * alpha * dphi_0) {
                    return alpha;
                }

                alpha *= rho;
            }

            return 0.0;
        }
    };

} // namespace optimization

#endif // HYPERPARAMETER_OPTIMIZER_HPP*/
