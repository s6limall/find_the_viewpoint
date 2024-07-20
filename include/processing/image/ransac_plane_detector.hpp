/*
#ifndef RANSAC_PLANE_DETECTOR_HPP
#define RANSAC_PLANE_DETECTOR_HPP

#include <Eigen/Dense>
#include <random>
#include "detector.hpp"

template<typename Point3D, typename Point2D>
class RansacPlaneDetector : public Detector<Point3D, Point2D> {
public:
    RansacPlaneDetector(double distanceThreshold = 1.0, int maxIterations = 1000);

    typename Detector<Point3D, Point2D>::DetectionResult detect(
        const std::vector<Point3D>& objectPoints,
        const std::vector<Point2D>& imagePoints) const noexcept override;

private:
    double distance_threshold;
    int max_iterations;

    typename Detector<Point3D, Point2D>::DetectionResult findPlaneModel(
        const std::vector<Point3D>& objectPoints,
        const std::vector<Point2D>& imagePoints,
        const std::vector<int>& indices) const noexcept;

    std::vector<int> getInliers(const Eigen::VectorXd& coefficients,
                                const std::vector<Point3D>& objectPoints) const noexcept;

    Eigen::VectorXd fitPlane(const std::vector<Point3D>& points) const noexcept;
};

template<typename Point3D, typename Point2D>
RansacPlaneDetector<Point3D, Point2D>::RansacPlaneDetector(double distanceThreshold, int maxIterations)
    : distance_threshold(distanceThreshold), max_iterations(maxIterations) {}

template<typename Point3D, typename Point2D>
typename Detector<Point3D, Point2D>::DetectionResult RansacPlaneDetector<Point3D, Point2D>::detect(
    const std::vector<Point3D>& objectPoints,
    const std::vector<Point2D>& imagePoints) const noexcept {

    int bestInlierCount = 0;
    typename Detector<Point3D, Point2D>::DetectionResult bestModel;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < max_iterations; ++i) {
        std::vector<int> sampleIndices(3);
        std::sample(objectPoints.begin(), objectPoints.end(), sampleIndices.begin(), 3, gen);

        auto model = findPlaneModel(objectPoints, imagePoints, sampleIndices);
        if (model.inliers.size() > bestInlierCount) {
            bestInlierCount = model.inliers.size();
            bestModel = model;
        }
    }

    return bestModel;
}

template<typename Point3D, typename Point2D>
typename Detector<Point3D, Point2D>::DetectionResult RansacPlaneDetector<Point3D, Point2D>::findPlaneModel(
    const std::vector<Point3D>& objectPoints,
    const std::vector<Point2D>& imagePoints,
    const std::vector<int>& indices) const noexcept {

    std::vector<Point3D> samplePoints = { objectPoints[indices[0]], objectPoints[indices[1]], objectPoints[indices[2]]
}; Eigen::VectorXd planeCoefficients = fitPlane(samplePoints);

    std::vector<int> inliers = getInliers(planeCoefficients, objectPoints);

    return { planeCoefficients, inliers };
}

template<typename Point3D, typename Point2D>
std::vector<int> RansacPlaneDetector<Point3D, Point2D>::getInliers(
    const Eigen::VectorXd& coefficients,
    const std::vector<Point3D>& objectPoints) const noexcept {

    const auto normal = coefficients.head<3>();
    const auto d = coefficients[3];
    const auto denominator = normal.norm();

    std::vector<int> inliers;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        const auto& point = objectPoints[i];
        double distance = std::abs(normal.dot(point) + d) / denominator;
        if (distance < distance_threshold) {
            inliers.push_back(static_cast<int>(i));
        }
    }
    return inliers;
}

template<typename Point3D, typename Point2D>
Eigen::VectorXd RansacPlaneDetector<Point3D, Point2D>::fitPlane(const std::vector<Point3D>& points) const noexcept {
    Eigen::MatrixXd A(points.size(), 3);
    Eigen::VectorXd B = Eigen::VectorXd::Ones(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        A.row(i) = points[i].template head<3>();
    }

    Eigen::VectorXd coefficients = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    coefficients.conservativeResize(4);
    coefficients(3) = -1.0;

    return coefficients;
}

#endif // RANSAC_PLANE_DETECTOR_HPP
*/
