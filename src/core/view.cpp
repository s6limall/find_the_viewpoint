//
// Created by ayush on 5/21/24.
//

#include "../../include/core/view.hpp"


View::View() {
    camera.setPosition(0, 0, 0); // Initialize camera position.
}

View::~View() {
}

void View::computePoseFromPositionAndObjectCenter(Eigen::Vector3f position, Eigen::Vector3f object_center) {
    camera.setPosition(position.x(), position.y(), position.z()); // Set camera position.
    camera.lookAt(object_center.cast<float>()); // Orient camera towards the object center.
}


/*void View::computePoseFromPositionAndObjectCenter(Eigen::Vector3d position, Eigen::Vector3d object_center) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,1>(0,3) = -position;

    Eigen::Vector3d Z = (object_center - position).normalized();
    Eigen::Vector3d X = (-Z).cross(Eigen::Vector3d(0, 1, 0)).normalized();
    Eigen::Vector3d Y = X.cross(-Z).normalized();

    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
    R.block<3,3>(0,0) << X, Y, Z;

    pose_6d = (R.inverse() * T).inverse();
}*/
