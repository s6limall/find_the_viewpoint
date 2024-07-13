// File: optimization/gaussian_process.hpp

#ifndef GAUSSIAN_PROCESS_HPP
#define GAUSSIAN_PROCESS_HPP

#include <Eigen/Core>
#include <vector>

class GaussianProcess {
public:
    GaussianProcess(double length_scale, double variance, double noise_level);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    Eigen::VectorXd predict(const Eigen::MatrixXd& X_pred) const;

    Eigen::VectorXd sampleNewPoints(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, size_t num_samples) const;

private:
    double length_scale_;
    double variance_;
    double noise_level_;

    Eigen::MatrixXd X_train_;
    Eigen::VectorXd y_train_;

    Eigen::MatrixXd computeKernel(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2) const;
};

#endif //GAUSSIAN_PROCESS_HPP
