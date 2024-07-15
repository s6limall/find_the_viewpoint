#ifndef GAUSSIAN_PROCESS_HPP
#define GAUSSIAN_PROCESS_HPP

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

template<typename T>
class GaussianProcess {
public:
    GaussianProcess(const std::vector<T> &points, const std::vector<double> &values) :
        points_(points), values_(values) {
        computeKernelMatrix();
    }

    const std::vector<T> &getPoints() const { return points_; }
    const std::vector<double> &getValues() const { return values_; }

    double mean(const T &point) const {
        Eigen::VectorXd k = computeKernelVector(point);
        return k.transpose() * alpha_;
    }

    double covariance(const T &point1, const T &point2) const {
        return kernel(point1, point2);
    }

    void addSample(const T &point, double value) {
        points_.push_back(point);
        values_.push_back(value);
        computeKernelMatrix();
    }

private:
    std::vector<T> points_;
    std::vector<double> values_;
    Eigen::MatrixXd K_;
    Eigen::VectorXd alpha_;

    void computeKernelMatrix() {
        size_t n = points_.size();
        K_.resize(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                K_(i, j) = kernel(points_[i], points_[j]);
            }
        }
        Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(values_.data(), values_.size());
        alpha_ = K_.ldlt().solve(y);
    }

    Eigen::VectorXd computeKernelVector(const T &point) const {
        Eigen::VectorXd k(points_.size());
        for (size_t i = 0; i < points_.size(); ++i) {
            k[i] = kernel(point, points_[i]);
        }
        return k;
    }

    double kernel(const T &point1, const T &point2) const {
        // Using RBF kernel
        double l = 1.0; // Length scale
        double sigma_f = 1.0; // Signal variance
        return sigma_f * std::exp(-0.5 * (point1 - point2).squaredNorm() / (l * l));
    }
};

#endif // GAUSSIAN_PROCESS_HPP
