#ifndef VIEWPOINT_HPP
#define VIEWPOINT_HPP

#include <tuple>
#include <type_traits>
#include <cmath>
#include <opencv2/core.hpp>
#include <fmt/core.h>
#include <vector>

#include "core/view.hpp"
#include "common/logging/logger.hpp"

template<typename T = double>
class ViewPoint {
    static_assert(std::is_arithmetic_v<T>, "ViewPoint template must be numeric");

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructors
    ViewPoint() = delete;

    constexpr ViewPoint(T x, T y, T z, const double score = 0.0) noexcept :
        position_(x, y, z), score_(score), cluster_id_(-1) {
        validatePosition();
        view_ = core::View::fromPosition(position_);
    }

    explicit constexpr ViewPoint(const Eigen::Matrix<T, 3, 1> &position, double score = 0.0) noexcept :
        ViewPoint(position.x(), position.y(), position.z(), score) {
    }

    // Getters
    constexpr const Eigen::Matrix<T, 3, 1> &getPosition() const noexcept { return position_; }
    [[nodiscard]] constexpr int getClusterId() const noexcept { return cluster_id_; }
    [[nodiscard]] constexpr double getScore() const noexcept { return score_; }

    // Setters
    constexpr void setClusterId(const int cluster_id) noexcept { cluster_id_ = cluster_id; }
    constexpr void setScore(const double score) noexcept { score_ = score; }

    // Conversion to Cartesian coordinates
    constexpr std::tuple<T, T, T> toCartesian() const noexcept {
        return std::make_tuple(position_.x(), position_.y(), position_.z());
    }

    // Distance calculation
    template<typename Derived>
    constexpr std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>, T>
    distance(const Eigen::MatrixBase<Derived> &other) const noexcept {
        return (position_ - other).norm();
    }

    constexpr T distance(const cv::Point3d &other) const noexcept {
        Eigen::Matrix<T, 3, 1> other_eigen(other.x, other.y, other.z);
        return (position_ - other_eigen).norm();
    }

    T distance(const std::vector<T> &other) const {
        if (other.size() != 3) {
            throw std::invalid_argument("Vector size must be 3 for distance calculation.");
        }
        Eigen::Matrix<T, 3, 1> other_eigen(other[0], other[1], other[2]);
        return (position_ - other_eigen).norm();
    }

    constexpr T distance(const ViewPoint &other) const noexcept {
        return (position_ - other.position_).norm();
    }

    // Static factory methods
    static constexpr ViewPoint fromCartesian(T x, T y, T z, double score = 0.0) noexcept {
        return ViewPoint(x, y, z, score);
    }

    static constexpr ViewPoint fromSpherical(T radius, T polar_angle, T azimuthal_angle, double score = 0.0) noexcept {
        const T x = radius * std::sin(polar_angle) * std::cos(azimuthal_angle);
        const T y = radius * std::sin(polar_angle) * std::sin(azimuthal_angle);
        const T z = radius * std::cos(polar_angle);
        return ViewPoint(x, y, z, score);
    }

    static ViewPoint fromView(const core::View &view, double score = 0.0) noexcept {
        const auto position = view.getPosition();
        return ViewPoint(position.x(), position.y(), position.z(), score);
    }

    // Conversion from Eigen
    template<typename Derived>
    static constexpr std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>, ViewPoint>
    fromEigen(const Eigen::MatrixBase<Derived> &eigenVector, double score = 0.0) noexcept {
        static_assert(std::is_same_v<typename Derived::Scalar, T>,
                      "Eigen matrix scalar type must match ViewPoint template type");
        return ViewPoint(eigenVector, score);
    }

    // Conversion from OpenCV
    static constexpr ViewPoint fromOpenCV(const cv::Point3f &cvPoint, double score = 0.0) noexcept {
        static_assert(std::is_same_v<cv::Point3f::value_type, T>,
                      "OpenCV Point type must match ViewPoint template type");
        Eigen::Matrix<T, 3, 1> position(cvPoint.x, cvPoint.y, cvPoint.z);
        return ViewPoint(position, score);
    }

    // Conversion to Spherical coordinates
    constexpr std::tuple<T, T, T> toSpherical() const noexcept {
        const T radius = position_.norm();
        const T polar_angle = std::acos(position_.z() / radius);
        const T azimuthal_angle = std::atan2(position_.y(), position_.x());
        return std::make_tuple(radius, polar_angle, azimuthal_angle);
    }

    [[nodiscard]] core::View toView(const Eigen::Vector3d &object_center = Eigen::Vector3d::Zero()) const noexcept {
        return core::View::fromPosition(position_, object_center);
    }

    // Serialization to string
    [[nodiscard]] std::string toString() const {
        return fmt::format("ViewPoint(x: {}, y: {}, z: {}, score: {}, cluster_id: {})",
                           position_.x(), position_.y(), position_.z(), score_, cluster_id_);
    }

private:
    Eigen::Matrix<T, 3, 1> position_;
    core::View view_;
    double score_;
    int cluster_id_; // -1 = unset, -2 = noise, >= 0 = cluster_id

    // Validation
    void validatePosition() const {
        if (position_.hasNaN()) {
            LOG_ERROR("Error initializing ViewPoint: position must not contain NaN values.");
            throw std::invalid_argument("Position must not contain NaN values.");
        }
        if (position_.isZero()) {
            LOG_WARN("ViewPoint's position is the zero vector.");
        }
    }
};

#endif // VIEWPOINT_HPP
