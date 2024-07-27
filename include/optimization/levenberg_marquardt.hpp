#ifndef LEVENBERG_MARQUARDT_HPP
#define LEVENBERG_MARQUARDT_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace optimization {

    template<typename Scalar, int Dim>
    class LevenbergMarquardt {
    public:
        using IsometryType = Eigen::Transform<Scalar, Dim, Eigen::Isometry>;
        using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
        using MatrixType = Eigen::Matrix<Scalar, Dim, Dim>;
        using Vector6Type = Eigen::Matrix<Scalar, 6, 1>;

        struct Options {
            Scalar initial_lambda = static_cast<Scalar>(1e-3);
            Scalar lambda_factor = static_cast<Scalar>(10.0);
            int max_iterations = 100;
            Scalar gradient_threshold = static_cast<Scalar>(1e-8);
            Scalar parameter_threshold = static_cast<Scalar>(1e-8);
            Scalar error_threshold = static_cast<Scalar>(1e-8);
        };

        struct Result {
            IsometryType pose;
            Scalar final_error;
            int iterations;
            bool converged;
        };

        explicit LevenbergMarquardt(const Options &options = Options{}) : options_{options} {}

        template<typename Camera>
        std::optional<Result> optimize(const IsometryType &initial_pose, const std::vector<VectorType> &world_points,
                                       const std::vector<Eigen::Matrix<Scalar, 2, 1>> &image_points,
                                       const Camera &camera) const noexcept {
            if (world_points.size() != image_points.size() || world_points.empty()) {
                return std::nullopt;
            }

            LOG_DEBUG("Optimizing using Levenberg-Marquardt...");

            auto current_pose = initial_pose;
            auto lambda = options_.initial_lambda;
            auto prev_error = std::numeric_limits<Scalar>::max();

            Result result{current_pose, static_cast<Scalar>(0.0), 0, false};

            for (int iter = 0; iter < options_.max_iterations; ++iter) {
                Eigen::Matrix<Scalar, Eigen::Dynamic, 6> jacobian(2 * world_points.size(), 6);
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> error_vector(2 * world_points.size());

                const auto current_error = computeErrorAndJacobian(current_pose, world_points, image_points, camera,
                                                                   jacobian, error_vector);

                if (std::abs(prev_error - current_error) < options_.error_threshold) {
                    result.converged = true;
                    break;
                }

                const Eigen::Matrix<Scalar, 6, 6> JTJ = jacobian.transpose() * jacobian;
                const Eigen::Matrix<Scalar, 6, 1> JTe = jacobian.transpose() * error_vector;

                const Eigen::Matrix<Scalar, 6, 6> augmented_JTJ =
                        JTJ + lambda * Eigen::Matrix<Scalar, 6, 6>::Identity();
                const Eigen::Matrix<Scalar, 6, 1> delta = augmented_JTJ.ldlt().solve(-JTe);

                if (delta.norm() < options_.parameter_threshold || JTe.norm() < options_.gradient_threshold) {
                    result.converged = true;
                    break;
                }

                const auto new_pose = current_pose * expSE3(delta);
                const auto new_error = computeError(new_pose, world_points, image_points, camera);

                if (new_error < current_error) {
                    current_pose = new_pose;
                    lambda /= options_.lambda_factor;
                    prev_error = current_error;
                } else {
                    lambda *= options_.lambda_factor;
                }

                result.iterations = iter + 1;
            }

            result.pose = current_pose;
            result.final_error = computeError(current_pose, world_points, image_points, camera);

            LOG_DEBUG("Optimization completed in {} iterations. Final error: {}", result.iterations,
                      result.final_error);

            return result;
        }

    private:
        Options options_;

        template<typename Camera>
        static Scalar computeError(const IsometryType &pose, const std::vector<VectorType> &world_points,
                                   const std::vector<Eigen::Matrix<Scalar, 2, 1>> &image_points,
                                   const Camera &camera) noexcept {
            Scalar total_error = 0;
            for (size_t i = 0; i < world_points.size(); ++i) {
                total_error += (camera.project(pose * world_points[i]) - image_points[i]).squaredNorm();
            }
            return total_error / world_points.size();
        }


        template<typename Camera>
        static Scalar computeErrorAndJacobian(const IsometryType &pose, const std::vector<VectorType> &world_points,
                                              const std::vector<Eigen::Matrix<Scalar, 2, 1>> &image_points,
                                              const Camera &camera, Eigen::Matrix<Scalar, Eigen::Dynamic, 6> &jacobian,
                                              Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &error_vector) noexcept {
            Scalar total_error = 0.0;
            for (size_t i = 0; i < world_points.size(); ++i) {
                const auto transformed = pose * world_points[i];
                const auto projected = camera.project(transformed);
                const Eigen::Matrix<Scalar, 2, 1> error = projected - image_points[i];

                error_vector.template segment<2>(2 * i) = error;
                total_error += error.squaredNorm();

                const Eigen::Matrix<Scalar, 2, 3> J_proj = camera.projectJacobian(transformed);
                const Eigen::Matrix<Scalar, 3, 6> J_trans = computeTransformJacobian(transformed);
                jacobian.template block<2, 6>(2 * i, 0) = J_proj * J_trans;
            }
            return total_error / world_points.size();
        }

        static Eigen::Matrix<Scalar, 3, 6> computeTransformJacobian(const VectorType &point) noexcept {
            Eigen::Matrix<Scalar, 3, 6> J;
            J.template block<3, 3>(0, 0) = MatrixType::Identity();
            J.template block<3, 3>(0, 3) = -skewSymmetric(point);
            return J;
        }

        static MatrixType skewSymmetric(const VectorType &v) noexcept {
            MatrixType m;
            m << Scalar(0), -v.z(), v.y(), v.z(), Scalar(0), -v.x(), -v.y(), v.x(), Scalar(0);
            return m;
        }

        static IsometryType expSE3(const Vector6Type &xi) noexcept {
            const Eigen::Matrix<Scalar, 3, 1> rho = xi.template head<3>();
            const Eigen::Matrix<Scalar, 3, 1> phi = xi.template tail<3>();
            const Scalar theta = phi.norm();

            MatrixType R;
            VectorType t;

            if (theta < Scalar(1e-8)) {
                R = MatrixType::Identity() + skewSymmetric(phi);
                t = rho;
            } else {
                const MatrixType phi_hat = skewSymmetric(phi);
                R = MatrixType::Identity() + std::sin(theta) / theta * phi_hat +
                    (Scalar(1) - std::cos(theta)) / (theta * theta) * phi_hat * phi_hat;
                const MatrixType V = MatrixType::Identity() +
                                     (Scalar(1) - std::cos(theta)) / (theta * theta) * phi_hat +
                                     (theta - std::sin(theta)) / (theta * theta * theta) * phi_hat * phi_hat;
                t = V * rho;
            }

            IsometryType T = IsometryType::Identity();
            T.linear() = R;
            T.translation() = t;
            return T;
        }
    };

} // namespace optimization

#endif // LEVENBERG_MARQUARDT_HPP
