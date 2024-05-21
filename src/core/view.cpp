//
// Created by ayush on 5/21/24.
//

#include "../../include/core/view.hpp"

View::View() {
    pose_6d = Eigen::Matrix4d::Identity();
}

View::~View() {}

void View::computePoseFromPositionAndObjectCenter(Eigen::Vector3d position, Eigen::Vector3d object_center) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,1>(0,3) = -position;

    Eigen::Vector3d Z = (object_center - position).normalized();
    Eigen::Vector3d X = (-Z).cross(Eigen::Vector3d(0, 1, 0)).normalized();
    Eigen::Vector3d Y = X.cross(-Z).normalized();

    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
    R.block<3,3>(0,0) << X, Y, Z;

    pose_6d = (R.inverse() * T).inverse();
}
