/*
#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <Eigen/Core>
#include <vector>

template<typename Point3D, typename Point2D>
class Detector {
public:
    struct DetectionResult {
        Eigen::VectorXd modelCoefficients;
        std::vector<int> inliers;
    };

    virtual ~Detector() = default;
    virtual DetectionResult detect(const std::vector<Point3D>& objectPoints,
                                   const std::vector<Point2D>& imagePoints) const noexcept = 0;
};

#endif // DETECTOR_HPP
*/
