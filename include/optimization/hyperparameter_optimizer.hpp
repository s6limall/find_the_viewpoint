// File: optimization/hyperparameter_optimizer.hpp

#ifndef HYPERPARAMETER_OPTIMIZER_HPP
#define HYPERPARAMETER_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "common/logging/logger.hpp"
#include "optimization/kernel/matern_52.hpp"

namespace optimization {

    template<typename Kernel = kernel::Matern52<>>
    class HyperparameterOptimizer {
    public:
        using VectorXd = Eigen::VectorXd;
        using MatrixXd = Eigen::MatrixXd;

        explicit HyperparameterOptimizer(const int max_iterations = 100, const double convergence_tol = 1e-5) :
            max_iterations_(max_iterations), convergence_tol_(convergence_tol) {}

        VectorXd optimize(const MatrixXd &X, const VectorXd &y, Kernel &kernel) {
            if (X.rows() != y.rows() || X.rows() == 0) {
                LOG_ERROR("Invalid input dimensions: X rows: {}, y rows: {}", X.rows(), y.rows());
                return kernel.getParameters(); // Return current parameters if input is invalid
            }

            VectorXd params = kernel.getParameters();
            VectorXd best_params = params;
            double best_nlml = std::numeric_limits<double>::infinity();

            VectorXd grad(params.size());
            std::vector<VectorXd> s, y_lbfgs;

            for (int iter = 0; iter < max_iterations_; ++iter) {
                double nlml = computeNLML(X, y, kernel, params, grad);

                if (!std::isfinite(nlml)) {
                    LOG_WARN("Non-finite NLML encountered. Using best parameters so far.");
                    break;
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

                if (step_size > std::numeric_limits<double>::epsilon()) {
                    VectorXd new_params = params + step_size * direction;
                    new_params = new_params.cwiseMax(1e-6).cwiseMin(10.0); // Constrain parameters

                    VectorXd s_k = new_params - params;
                    VectorXd y_k = computeGradient(X, y, kernel, new_params) - grad;

                    if (s_k.norm() > std::numeric_limits<double>::epsilon() &&
                        y_k.norm() > std::numeric_limits<double>::epsilon()) {
                        s.push_back(s_k);
                        y_lbfgs.push_back(y_k);

                        if (s.size() > 10) { // Limit memory of L-BFGS
                            s.erase(s.begin());
                            y_lbfgs.erase(y_lbfgs.begin());
                        }
                    }

                    params = new_params;
                } else {
                    LOG_WARN("Optimization stopped: step size is effectively zero");
                    break;
                }
            }

            // Ensure best_params are within valid range
            best_params = best_params.cwiseMax(1e-6).cwiseMin(10.0);

            // Use try-catch to handle potential exceptions from setParameters
            try {
                kernel.setParameters(best_params(0), best_params(1), best_params(2));
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to set kernel parameters: {}", e.what());
                // Revert to original parameters
                kernel.setParameters(params(0), params(1), params(2));
            }

            return best_params;
        }

    private:
        int max_iterations_;
        double convergence_tol_;

        static double computeNLML(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                                  VectorXd &grad) {
            kernel.setParameters(params(0), params(1), params(2));
            MatrixXd K = kernel.computeGramMatrix(X);
            Eigen::LLT<MatrixXd> llt(K);
            VectorXd alpha = llt.solve(y);

            double nlml = 0.5 * y.dot(alpha) + 0.5 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum() +
                          0.5 * X.rows() * std::log(2 * M_PI);

            MatrixXd K_inv = llt.solve(MatrixXd::Identity(X.rows(), X.rows()));
            MatrixXd alpha_alpha_t = alpha * alpha.transpose();
            MatrixXd factor = alpha_alpha_t - K_inv;

            for (int i = 0; i < params.size(); ++i) {
                MatrixXd K_grad = kernel.computeGradientMatrix(X, i);
                grad(i) = 0.5 * (factor.array() * K_grad.array()).sum();
            }

            return nlml;
        }

        VectorXd computeGradient(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params) {
            VectorXd grad(params.size());
            computeNLML(X, y, kernel, params, grad);
            return grad;
        }

        static VectorXd computeLBFGSDirection(const VectorXd &grad, const std::vector<VectorXd> &s,
                                              const std::vector<VectorXd> &y) {
            if (s.empty() || y.empty() || s.size() != y.size()) {
                return -grad;
            }

            VectorXd q = -grad;
            std::vector<double> alpha(s.size());

            for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
                double denominator = y[i].dot(s[i]);
                if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
                    LOG_WARN("Division by zero avoided in L-BFGS computation");
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

        double lineSearch(const MatrixXd &X, const VectorXd &y, Kernel &kernel, const VectorXd &params,
                          const VectorXd &direction, const VectorXd &grad, const double current_nlml) {
            double alpha = 1.0;
            const double c = 0.5;
            const double rho = 0.5;

            const double phi_0 = current_nlml;
            const double dphi_0 = grad.dot(direction);

            VectorXd new_grad(params.size());
            for (int i = 0; i < 10; ++i) { // Max 10 iterations for line search
                VectorXd new_params = params + alpha * direction;
                const double phi = computeNLML(X, y, kernel, new_params, new_grad);

                if (phi <= phi_0 + c * alpha * dphi_0) {
                    return alpha;
                }

                alpha *= rho;
            }

            return alpha;
        }
    };

} // namespace optimization

#endif // HYPERPARAMETER_OPTIMIZER_HPP
