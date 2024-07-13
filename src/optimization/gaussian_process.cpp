// File: optimization/gaussian_process.cpp


#include "optimization/gaussian_process.hpp"

GaussianProcess::GaussianProcess(double length_scale, double variance, double noise_level)
    : length_scale_(length_scale), variance_(variance), noise_level_(noise_level) {}

void GaussianProcess::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    X_train_ = X;
    y_train_ = y;
}

Eigen::MatrixXd GaussianProcess::computeKernel(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2) const {
    Eigen::MatrixXd dists = ((X1.rowwise().squaredNorm()).transpose().replicate(X2.rows(), 1) +
                             X2.rowwise().squaredNorm().replicate(1, X1.rows()) -
                             2 * X1 * X2.transpose());

    return variance_ * (-0.5 / (length_scale_ * length_scale_) * dists.array()).exp().matrix();
}

Eigen::VectorXd GaussianProcess::predict(const Eigen::MatrixXd& X_pred) const {
    Eigen::MatrixXd K = computeKernel(X_train_, X_train_);
    K.diagonal().array() += noise_level_;

    Eigen::MatrixXd K_s = computeKernel(X_train_, X_pred);
    Eigen::MatrixXd K_ss = computeKernel(X_pred, X_pred);
    K_ss.diagonal().array() += noise_level_;

    Eigen::MatrixXd K_inv = K.inverse();

    Eigen::VectorXd mu = K_s.transpose() * K_inv * y_train_;
    return mu;
}

Eigen::VectorXd GaussianProcess::sampleNewPoints(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, size_t num_samples) const {
    Eigen::MatrixXd X_pred = Eigen::MatrixXd::Random(num_samples, X.cols());
    Eigen::VectorXd mu = predict(X_pred);

    // Return the points with the highest predicted mean values
    std::vector<int> indices(mu.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + num_samples, indices.end(),
                      [&mu](int a, int b) { return mu(a) > mu(b); });

    Eigen::MatrixXd new_points(num_samples, X.cols());
    for (size_t i = 0; i < num_samples; ++i) {
        new_points.row(i) = X_pred.row(indices[i]);
    }

    return new_points;
}
