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
            VectorXd params = kernel.getParameters();
            VectorXd best_params = params;
            double best_nlml = std::numeric_limits<double>::infinity();

            VectorXd grad(params.size());
            std::vector<VectorXd> s, y_lbfgs;

            for (int iter = 0; iter < max_iterations_; ++iter) {
                double nlml = computeNLML(X, y, kernel, params, grad);

                if (nlml < best_nlml) {
                    best_nlml = nlml;
                    best_params = params;
                }

                if (grad.norm() < convergence_tol_) {
                    LOG_INFO("Hyperparameter optimization converged after {} iterations", iter);
                    break;
                }

                VectorXd direction = computeLBFGSDirection(grad, s, y_lbfgs);
                double step_size = lineSearch(X, y, kernel, params, direction, grad, nlml);

                VectorXd new_params = params + step_size * direction;
                new_params = new_params.cwiseMax(1e-6).cwiseMin(10.0); // Constrain parameters

                s.emplace_back(new_params - params);
                y_lbfgs.push_back(computeGradient(X, y, kernel, new_params) - grad);

                if (s.size() > 10) { // Limit memory of L-BFGS
                    s.erase(s.begin());
                    y_lbfgs.erase(y_lbfgs.begin());
                }

                params = new_params;
            }

            kernel.setParameters(best_params(0), best_params(1), best_params(2));

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
            VectorXd q = -grad;
            std::vector<double> alpha(s.size());

            for (size_t i = s.size() - 1; i >= 0; --i) {
                alpha[i] = s[i].dot(q) / y[i].dot(s[i]);
                q -= alpha[i] * y[i];
            }

            VectorXd z = q * ((s.back().dot(y.back()) / y.back().squaredNorm()));

            for (size_t i = 0; i < s.size(); ++i) {
                double beta = y[i].dot(z) / y[i].dot(s[i]);
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
