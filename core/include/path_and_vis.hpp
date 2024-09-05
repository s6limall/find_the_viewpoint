#ifndef PATH_AND_VIS_HPP
#define PATH_AND_VIS_HPP

#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <json/json.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

namespace path_and_vis {
	typedef enum rs2_distortion
	{
		RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
		RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
		RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
		RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
		RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
		RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
		RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
	} rs2_distortion;
	/** \brief Video stream intrinsics. */
	typedef struct rs2_intrinsics
	{
		int           width;     /**< Width of the image in pixels */
		int           height;    /**< Height of the image in pixels */
		float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
		float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
		float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
		float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
		rs2_distortion model;    /**< Distortion model of the image */
		float         coeffs[5]; /**< Distortion coefficients */
	} rs2_intrinsics;

	std::pair<int, double> get_local_path(Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double r);
	double get_trajectory_xyz(std::vector<Eigen::Vector3d>& points, Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double predicted_size, double distanse_of_pre_move, double camera_to_object_dis); 
	static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth);
	Eigen::Vector3d project_pixel_to_ray_end(int x, int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range);
}
#endif // PATH_AND_VIS_HPP

