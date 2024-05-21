//
// Created by ayush on 5/21/24.
//

#ifndef VIEW_HPP
#define VIEW_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class View {
public:
    Eigen::Matrix4d pose_6d;

    View();
    ~View();

    void computePoseFromPositionAndObjectCenter(Eigen::Vector3d position, Eigen::Vector3d object_center);
};

#endif // VIEW_HPP
