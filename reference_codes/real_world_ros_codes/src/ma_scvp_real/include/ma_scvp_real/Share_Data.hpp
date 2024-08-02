#pragma once
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>  
#include <string>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <map>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


using namespace std;

/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
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

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

#define RandomIterative 0
#define RandomOneshot 1
#define EnsembleRGB 2
#define EnsembleRGBDensity 3
#define PVBCoverage 4

class Share_Data
{
public:
	//可变输入参数
	string yaml_file_path;
	string name_of_pcd;
	string orginalviews_path;
	string viewspace_path;
	string instant_ngp_path;
	string pvb_path;

	int num_of_views;					//一次采样视点个数
	rs2_intrinsics color_intrinsics;
	double depth_scale;
	double view_space_radius;

	int num_of_max_iteration;			//最大迭代次数

	Eigen::Matrix4d now_camera_pose_world;		//当前相机位姿

	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径

	int method_of_IG;					//信息增益计算方法

	string pre_path;
	string gt_path;
	string save_path;

	vector<vector<double>> pt_sphere;
	double pt_norm;

	int ray_casting_aabb_scale;
	int n_steps;

	int ensemble_num;
	int evaluate;

	int test_id;

	// 可交换视角: 0<->2, 3<->4 或者 <0,2,3,4> 四元群
	vector<int> init_view_ids_case_v1;
	vector<int> init_view_ids_case_v2;
	vector<int> init_view_ids_case_v3;
	vector<int> init_view_ids_case_v4;
	vector<int> init_view_ids_case_v5;

	vector<int> init_view_ids;

	int mode;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_now;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_scene;
	double height_of_ground;
	double height_to_filter_arm;
	double move_dis_pre_point;
	double min_z_table;
	double up_shift;

	Eigen::Matrix4d camera_depth_to_rgb;

	cv::Mat rgb_now;

	Share_Data(string _config_file_path, string test_name = "", int _num_of_views = -1, int test_method = -1)
	{
		yaml_file_path = _config_file_path;
		//读取yaml文件
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["mode"] >> mode;
		fs["height_of_ground"] >> height_of_ground;
		fs["height_to_filter_arm"] >> height_to_filter_arm;
		fs["move_dis_pre_point"] >> move_dis_pre_point;
		fs["min_z_table"] >> min_z_table;
		fs["up_shift"] >> up_shift;
		fs["test_id"] >> test_id;
		fs["pre_path"] >> pre_path;
		fs["viewspace_path"] >> viewspace_path;
		fs["instant_ngp_path"] >> instant_ngp_path;
		fs["orginalviews_path"] >> orginalviews_path;
		fs["pvb_path"] >> pvb_path;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["method_of_IG"] >> method_of_IG;
		fs["n_steps"] >> n_steps;
		fs["evaluate"] >> evaluate;
		fs["ensemble_num"] >> ensemble_num;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["num_of_views"] >> num_of_views;
		fs["ray_casting_aabb_scale"] >> ray_casting_aabb_scale;
		fs["view_space_radius"] >> view_space_radius;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs["predicted_size"] >> predicted_size;
		fs["object_center_world(0)"] >> object_center_world(0);
		fs["object_center_world(1)"] >> object_center_world(1);
		fs["object_center_world(2)"] >> object_center_world(2);
		fs.release();
		if (test_name != "") name_of_pcd = test_name;
		if (test_method != -1) method_of_IG = test_method;
		if (_num_of_views != -1) num_of_views = _num_of_views;
		
		//初始化相机位姿
		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);

		//path
		gt_path = pre_path + "Coverage_images/";
		save_path = pre_path + "Compare/" ;

		gt_path += name_of_pcd;
		save_path += name_of_pcd;

		if (ensemble_num < 0) {
			if (method_of_IG == EnsembleRGB) {
				ensemble_num = 2;
			}
			else if (method_of_IG == EnsembleRGBDensity) {
				ensemble_num = 5;
			}
		}

		cout << "gt_path is: " << gt_path << endl;

		//read viewspace
		ifstream fin_sphere(orginalviews_path + to_string(num_of_views) + ".txt");
		pt_sphere.resize(num_of_views);
		for (int i = 0; i < num_of_views; i++) {
			pt_sphere[i].resize(3);
			for (int j = 0; j < 3; j++) {
				fin_sphere >> pt_sphere[i][j];
			}
		}
		cout<< "view space size is: " << pt_sphere.size() << endl;
		Eigen::Vector3d pt0(pt_sphere[0][0], pt_sphere[0][1], pt_sphere[0][2]);
		pt_norm = pt0.norm();

		//init view ids
		init_view_ids_case_v1.push_back(1);

		init_view_ids_case_v2.push_back(0);
		init_view_ids_case_v2.push_back(1);

		init_view_ids_case_v3.push_back(0);
		init_view_ids_case_v3.push_back(1);
		init_view_ids_case_v3.push_back(3);

		init_view_ids_case_v4.push_back(0);
		init_view_ids_case_v4.push_back(1);
		init_view_ids_case_v4.push_back(2);
		init_view_ids_case_v4.push_back(3);

		init_view_ids_case_v5.push_back(0);
		init_view_ids_case_v5.push_back(1);
		init_view_ids_case_v5.push_back(2);
		init_view_ids_case_v5.push_back(3);
		init_view_ids_case_v5.push_back(4);

		cloud_now = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_scene = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

		camera_depth_to_rgb(0, 0) = 0; camera_depth_to_rgb(0, 1) = 1; camera_depth_to_rgb(0, 2) = 0; camera_depth_to_rgb(0, 3) = 0;
		camera_depth_to_rgb(1, 0) = 0; camera_depth_to_rgb(1, 1) = 0; camera_depth_to_rgb(1, 2) = 1; camera_depth_to_rgb(1, 3) = 0;
		camera_depth_to_rgb(2, 0) = 1; camera_depth_to_rgb(2, 1) = 0; camera_depth_to_rgb(2, 2) = 0; camera_depth_to_rgb(2, 3) = 0;
		camera_depth_to_rgb(3, 0) = 0;	camera_depth_to_rgb(3, 1) = 0;	camera_depth_to_rgb(3, 2) = 0;	camera_depth_to_rgb(3, 3) = 1;

		init_view_ids = init_view_ids_case_v3;
		

		save_path += "_m" + to_string(method_of_IG);
		
		save_path += "_v" + to_string(init_view_ids.size());
		save_path += "_t" + to_string(test_id);

		cout << "save_path is: " << save_path << endl;
	}

	~Share_Data() {
		//释放内存
		pt_sphere.clear();
		pt_sphere.shrink_to_fit();
		init_view_ids_case_v1.clear();
		init_view_ids_case_v1.shrink_to_fit();
		init_view_ids_case_v2.clear();
		init_view_ids_case_v2.shrink_to_fit();
		init_view_ids_case_v3.clear();
		init_view_ids_case_v3.shrink_to_fit();
		init_view_ids_case_v4.clear();
		init_view_ids_case_v4.shrink_to_fit();
		init_view_ids_case_v5.clear();
		init_view_ids_case_v5.shrink_to_fit();
		cloud_now->points.clear();
		cloud_now->points.shrink_to_fit();
		cloud_now.reset();
		cloud_scene->points.clear();
		cloud_scene->points.shrink_to_fit();
		cloud_scene.reset();
	}

	Eigen::Matrix4d get_toward_pose(int toward_state)
	{
		Eigen::Matrix4d pose(Eigen::Matrix4d::Identity(4, 4));
		switch (toward_state) {
			case 0://z<->z
				return pose;
			case 1://z<->-z
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 0; pose(2, 2) = -1; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 2://z<->x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = 1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 3://z<->-x
				pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
				pose(2, 0) = -1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 4://z<->y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = 1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
			case 5://z<->-y
				pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
				pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
				pose(2, 0) = 0; pose(2, 1) = -1; pose(2, 2) = 0; pose(2, 3) = 0;
				pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
				return pose;
		}
		return pose;
	}

	void access_directory(string cd)
	{   //���༶Ŀ¼���ļ����Ƿ���ڣ������ھʹ���
		cout << cd << endl;
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), F_OK) == -1) mkdir(temp.c_str(), 0777);
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), F_OK) == -1) mkdir(temp.c_str(), 0777);
	}

};

inline double pow2(double x) {
	return x * x;
}

inline octomap::point3d project_pixel_to_ray_end(int x, int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range) {
	float pixel[2] = { float(x),float(y) };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);
	point_world = now_camera_pose_world * point_world;
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}

//转换白背景为透明
void convertToAlpha(cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
	for (int y = 0; y < dst.rows; ++y)
	{
		for (int x = 0; x < dst.cols; ++x)
		{
			cv::Vec4b& pixel = dst.at<cv::Vec4b>(y, x);
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255){
				pixel[3] = 0;
			}
		}
	}
}

//色彩化深度图
void colorize_depth(cv::Mat& src, cv::Mat& dst)
{
	dst=src.clone();
	dst.convertTo(dst, CV_32F);
	double min, max;
	cv::minMaxIdx(dst, &min, &max);
	convertScaleAbs(dst, dst, 255 / max);
	applyColorMap(dst, dst, cv::COLORMAP_JET);
	for (int y = 0; y < dst.rows; ++y)
	{
		for (int x = 0; x < dst.cols; ++x)
		{
			unsigned short depth = src.at<unsigned short>(y, x);
			if (depth == 0) {
				dst.at<cv::Vec3b>(y, x)[0] = 255;
				dst.at<cv::Vec3b>(y, x)[1] = 255;
				dst.at<cv::Vec3b>(y, x)[2] = 255;
			}
		}
	}
}

double ColorfulNess(cv::Mat& frame)
{
	// split image to 3 channels (B,G,R)
	cv::Mat channelsBGR[3];
	cv::split(frame, channelsBGR);

	// rg = R - G
	// yb = 0.5*(R + G) - B
	cv::Mat rg, yb;
	cv::absdiff(channelsBGR[2], channelsBGR[1], rg);
	cv::absdiff(0.5 * (channelsBGR[2] + channelsBGR[1]), channelsBGR[0], yb);

	// calculate the mean and std for rg and yb
	cv::Mat rgMean, rgStd; // 1*1矩阵
	cv::meanStdDev(rg, rgMean, rgStd);
	cv::Mat ybMean, ybStd; // 1*1矩阵
	cv::meanStdDev(yb, ybMean, ybStd);

	// calculate the mean and std for rgyb
	double stdRoot, meanRoot;
	stdRoot = sqrt(pow(rgStd.at<double>(0, 0), 2)
		+ pow(ybStd.at<double>(0, 0), 2));
	meanRoot = sqrt(pow(rgMean.at<double>(0, 0), 2)
		+ pow(ybMean.at<double>(0, 0), 2));

	// return colorfulNess
	return stdRoot + (0.3 * meanRoot);
}
