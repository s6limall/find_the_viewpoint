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

#endif // HYPERPARAMETER_OPTIMIZER_HPP