#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <functional>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class GaussianProcess {
public:
    GaussianProcess(double lengthScale, double signalVariance, double noiseVariance) :
        lengthScale_(lengthScale),
        signalVariance_(signalVariance),
        noiseVariance_(noiseVariance) {
    }

    void addSample(const VectorXd &x, double y) {
        samples_.push_back(x);
        targets_.push_back(y);
        if (samples_.size() > 1) {
            isTrained_ = false;
        } else {
            initializeKernelMatrix();
        }
    }

    double predict(const VectorXd &x) {
        if (!isTrained_) {
            train();
        }

        VectorXd k = computeKernel(x);
        double mean = k.dot(alpha_);
        return mean;
    }

    double expectedImprovement(const VectorXd &x, double yBest) {
        if (!isTrained_) {
            train();
        }

        VectorXd k = computeKernel(x);
        double mean = k.dot(alpha_);
        double variance = signalVariance_ - k.transpose() * K_inv_ * k;
        double stddev = std::sqrt(variance);

        double improvement = mean - yBest;
        double z = improvement / stddev;
        return improvement * stddev * std::erfc(-z / std::sqrt(2)) + stddev * std::exp(-0.5 * z * z) /
               std::sqrt(2 * M_PI);
    }

private:
    double lengthScale_;
    double signalVariance_;
    double noiseVariance_;
    bool isTrained_ = false;

    std::vector<VectorXd> samples_;
    std::vector<double> targets_;
    VectorXd alpha_;
    MatrixXd K_inv_;

    void train() {
        if (samples_.size() == 1) {
            initializeKernelMatrix();
            return;
        }

        int n = samples_.size();
        MatrixXd K(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                K(i, j) = kernel(samples_[i], samples_[j]);
            }
        }

        K += noiseVariance_ * MatrixXd::Identity(n, n);
        Eigen::LLT<MatrixXd> llt(K);
        alpha_ = llt.solve(VectorXd::Map(targets_.data(), targets_.size()));
        K_inv_ = llt.solve(MatrixXd::Identity(n, n));
        isTrained_ = true;
    }

    void initializeKernelMatrix() {
        int n = samples_.size();
        MatrixXd K(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                K(i, j) = kernel(samples_[i], samples_[j]);
            }
        }
        K += noiseVariance_ * MatrixXd::Identity(n, n);
        K_inv_ = K.inverse();
        alpha_ = K_inv_ * VectorXd::Map(targets_.data(), targets_.size());
        isTrained_ = true;
    }

    double kernel(const VectorXd &x1, const VectorXd &x2) {
        return signalVariance_ * std::exp(-(x1 - x2).squaredNorm() / (2 * lengthScale_ * lengthScale_));
    }

    VectorXd computeKernel(const VectorXd &x) {
        int n = samples_.size();
        VectorXd k(n);
        for (int i = 0; i < n; ++i) {
            k(i) = kernel(samples_[i], x);
        }
        return k;
    }
};
