// File: optimization/pose_optimizer.hpp

#ifndef POSE_OPTIMIZER_HPP
#define POSE_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <vector>
#include "types/viewpoint.hpp"

template<typename T = double>
class PoseOptimizer {
public:
    PoseOptimizer(double initialStepSize, double improvementThreshold, const ViewPoint<T> &initialViewPoint) :
        stepSize(initialStepSize), improvementThreshold(improvementThreshold), bestViewPoint(initialViewPoint),
        bestScore(-std::numeric_limits<double>::infinity()) {}

    void setScoreFunction(const std::function<double(const ViewPoint<T> &)> &func) noexcept { scoreFunction = func; }

    void addEvaluation(const ViewPoint<T> &viewpoint, double score) noexcept {
        evaluations.push_back({viewpoint, score});
        if (score > bestScore) {
            bestScore = score;
            bestViewPoint = viewpoint;
            computeAndStoreGradients(viewpoint, score);
        }
    }

    std::vector<ViewPoint<T>> generateNewCandidates() {
        std::vector<ViewPoint<T>> candidates;
        Eigen::Vector3d bestPosition = bestViewPoint.getPosition();
        for (const auto &grad: gradients) {
            Eigen::Vector3d newPosition = bestPosition + stepSize * grad;
            candidates.emplace_back(newPosition.x(), newPosition.y(), newPosition.z());
        }
        return candidates;
    }

    void adjustStepSize(double score) noexcept {
        double improvement = score - bestScore;
        if (improvement < improvementThreshold) {
            stepSize *= 0.7; // Conservative decrease
        } else {
            stepSize *= 1.1; // Moderate increase
        }
    }

    ViewPoint<T> getBestViewPoint() const noexcept { return bestViewPoint; }

    bool isConverged(double threshold) const noexcept { return bestScore > threshold; }

private:
    void computeAndStoreGradients(const ViewPoint<T> &viewpoint, double score) {
        Eigen::Vector3d newGradients = Eigen::Vector3d::Zero();
        double epsilon = 1e-5;

        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d perturbedPosition = viewpoint.getPosition();
            perturbedPosition[i] += epsilon;

            double perturbedScore =
                    scoreFunction(ViewPoint<T>(perturbedPosition.x(), perturbedPosition.y(), perturbedPosition.z()));
            newGradients[i] = (perturbedScore - score) / epsilon;
        }

        gradients = {newGradients};
    }

    double stepSize;
    double improvementThreshold;
    double bestScore;
    ViewPoint<T> bestViewPoint;
    std::vector<Eigen::Vector3d> gradients;
    struct Evaluation {
        ViewPoint<T> viewpoint;
        double score;
    };
    std::vector<Evaluation> evaluations;
    std::function<double(const ViewPoint<T> &)> scoreFunction;
};

#endif // POSE_OPTIMIZER_HPP
