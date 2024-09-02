// File: types/viewpoint.hpp

#ifndef VIEWPOINT_HPP
#define VIEWPOINT_HPP

#include <Eigen/Dense>
#include <cmath>
#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <optional>
#include <tuple>
#include <vector>

#include "common/logging/logger.hpp"
#include "core/view.hpp"
#include "types/concepts.hpp"

template<FloatingPoint T = double>
class ViewPoint {
public:
    // Constructors
    constexpr ViewPoint() noexcept : position_(Eigen::Matrix<T, 3, 1>::Zero()), score_(T(0)), uncertainty_(T(1)) {}

    constexpr ViewPoint(T x, T y, T z, T score = T(0), T uncertainty = T(1)) noexcept :
        position_(x, y, z), score_(score), uncertainty_(uncertainty) {
        validatePosition();
    }

    explicit constexpr ViewPoint(const Eigen::Matrix<T, 3, 1> &position, T score = T(0), T uncertainty = T(1)) noexcept
        : position_(position), score_(score), uncertainty_(uncertainty) {
        validatePosition();
    }

    // Rule of five
    ViewPoint(const ViewPoint &) = default;
    ViewPoint &operator=(const ViewPoint &) = default;
    ViewPoint(ViewPoint &&) noexcept = default;
    ViewPoint &operator=(ViewPoint &&) noexcept = default;
    ~ViewPoint() = default;

    // Getters
    [[nodiscard]] constexpr const Eigen::Matrix<T, 3, 1> &getPosition() const noexcept { return position_; }
    [[nodiscard]] constexpr T getScore() const noexcept { return score_; }
    [[nodiscard]] constexpr T getUncertainty() const noexcept { return uncertainty_; }

    [[nodiscard]] const core::View &getView() const {
        if (!view_.has_value()) {
            view_ = core::View::fromPosition(position_);
        }
        return view_.value();
    }

    // Setters
    constexpr void setScore(T score) noexcept { score_ = score; }
    constexpr void setUncertainty(T uncertainty) noexcept { uncertainty_ = uncertainty; }

    // Check if optional values are set
    [[nodiscard]] constexpr bool hasScore() const noexcept { return score_ != T(0); }
    [[nodiscard]] constexpr bool hasUncertainty() const noexcept { return uncertainty_ != T(1); }

    // Distance calculation
    template<typename Derived>
    [[nodiscard]] constexpr T distance(const Eigen::MatrixBase<Derived> &other) const noexcept
        requires std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>
    {
        return (position_ - other).norm();
    }

    [[nodiscard]] constexpr T distance(const cv::Point3_<T> &other) const noexcept {
        return distance(Eigen::Matrix<T, 3, 1>(other.x, other.y, other.z));
    }

    [[nodiscard]] T distance(const std::vector<T> &other) const {
        if (other.size() != 3) {
            throw std::invalid_argument("Vector size must be 3 for distance calculation.");
        }
        return distance(Eigen::Matrix<T, 3, 1>(other[0], other[1], other[2]));
    }

    [[nodiscard]] constexpr T distance(const ViewPoint &other) const noexcept { return distance(other.position_); }

    // Static factory methods
    [[nodiscard]] static constexpr ViewPoint fromCartesian(T x, T y, T z, T score = T(0),
                                                           T uncertainty = T(1)) noexcept {
        return ViewPoint(x, y, z, score, uncertainty);
    }

    [[nodiscard]] static ViewPoint fromSpherical(T radius, T polar_angle, T azimuthal_angle, T score = T(0),
                                                 T uncertainty = T(1)) noexcept {
        const Eigen::Matrix<T, 3, 1> position = {radius * std::sin(polar_angle) * std::cos(azimuthal_angle),
                                                 radius * std::sin(polar_angle) * std::sin(azimuthal_angle),
                                                 radius * std::cos(polar_angle)};
        return ViewPoint(position, score, uncertainty);
    }

    [[nodiscard]] static ViewPoint fromView(const core::View &view, T score = T(0), T uncertainty = T(1)) noexcept {
        return ViewPoint(view.getPosition(), score, uncertainty);
    }

    [[nodiscard]] static constexpr ViewPoint fromPosition(const Eigen::Vector3<T> &position, T score = T(0),
                                                          T uncertainty = T(1)) noexcept {
        return ViewPoint(position, score, uncertainty);
    }

    // Conversion from Eigen
    template<typename Derived>
    [[nodiscard]] static constexpr ViewPoint fromEigen(const Eigen::MatrixBase<Derived> &eigenVector, T score = T(0),
                                                       T uncertainty = T(1)) noexcept
        requires std::is_same_v<typename Derived::Scalar, T> && std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>
    {
        return ViewPoint(eigenVector, score, uncertainty);
    }

    // Conversion from OpenCV
    [[nodiscard]] static constexpr ViewPoint fromOpenCV(const cv::Point3_<T> &cvPoint, T score = T(0),
                                                        T uncertainty = T(1)) noexcept {
        return ViewPoint(cvPoint.x, cvPoint.y, cvPoint.z, score, uncertainty);
    }

    // Conversion to Cartesian coordinates
    [[nodiscard]] constexpr std::tuple<T, T, T> toCartesian() const noexcept {
        return {position_.x(), position_.y(), position_.z()};
    }

    // Conversion to Spherical coordinates
    [[nodiscard]] std::tuple<T, T, T> toSpherical() const noexcept {
        if (!spherical_coordinates_.has_value()) {
            const T radius = position_.norm();
            const T polar_angle = std::acos(position_.z() / radius);
            const T azimuthal_angle = std::atan2(position_.y(), position_.x());
            spherical_coordinates_ = std::make_tuple(radius, polar_angle, azimuthal_angle);
        }
        return spherical_coordinates_.value();
    }

    [[nodiscard]] core::View toView(const Eigen::Vector3<T> &object_center = Eigen::Vector3<T>::Zero()) const {
        if (!view_.has_value()) {
            view_ = core::View::fromPosition(position_, object_center);
        }
        return view_.value();
    }

    // Serialization to string
    [[nodiscard]] std::string toString() const {
        return fmt::format("ViewPoint(x: {}, y: {}, z: {}, score: {}, uncertainty: {})", position_.x(), position_.y(),
                           position_.z(), score_, uncertainty_);
    }

    // Comparison operators
    constexpr bool operator<(const ViewPoint &other) const noexcept {
        return std::tie(position_.x(), position_.y(), position_.z()) <
               std::tie(other.position_.x(), other.position_.y(), other.position_.z());
    }

    constexpr bool operator>(const ViewPoint &other) const noexcept {
        return std::tie(position_.x(), position_.y(), position_.z()) >
               std::tie(other.position_.x(), other.position_.y(), other.position_.z());
    }

    constexpr bool operator==(const ViewPoint &other) const noexcept { return position_ == other.position_; }

    constexpr bool operator!=(const ViewPoint &other) const noexcept { return !(*this == other); }

private:
    Eigen::Matrix<T, 3, 1> position_;
    mutable std::optional<core::View> view_;
    mutable std::optional<std::tuple<T, T, T>> spherical_coordinates_; // {radius, polar angle, azimuthal angle}
    T score_;
    T uncertainty_;

    // Validation
    constexpr void validatePosition() const {
        if (position_.hasNaN()) {
            LOG_ERROR("Error initializing ViewPoint: position must not contain NaN values. Received: ({}, {}, {}).",
                      position_.x(), position_.y(), position_.z());
            throw std::invalid_argument("Position must not contain NaN values.");
        }
        if (position_.isZero()) {
            LOG_WARN("ViewPoint's position is the zero vector: ({}, {}, {}).", position_.x(), position_.y(),
                     position_.z());
        }
    }
};

#endif // VIEWPOINT_HPP
