#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/io.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/conversions.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "../include/config.hpp"
#include "../include/image.hpp"

typedef unsigned long long pop_t;

using namespace std;

namespace task2{
//Robot
class Robot {
	;
};

//View
class View {
public:
	Eigen::Matrix4d pose_6d; // 6D pose of the camera

	// Constructor
	View() {
		pose_6d = Eigen::Matrix4d::Identity();
	}

	// Function to get the camera position from pose_6d
	Eigen::Vector3d getCameraPosition() const {
		Eigen::Vector3d position = pose_6d.block<3, 1>(0, 3);
		return position;
	}

	// Destructor
	~View() {
		;
	}

	// Get the 6D pose of the camera
	void compute_pose_from_positon_and_object_center(Eigen::Vector3d positon, Eigen::Vector3d object_center) {
		Eigen::Matrix4d T(4, 4);
		T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -positon(0);
		T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -positon(1);
		T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -positon(2);
		T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
		Eigen::Vector3d Z;	 Z = object_center - positon;	 Z = Z.normalized();
		Eigen::Vector3d X;	 X = (-Z).cross(Eigen::Vector3d(0, 1, 0));	 X = X.normalized();
		Eigen::Vector3d Y;	 Y = X.cross(-Z);	 Y = Y.normalized();
		Eigen::Matrix4d R(4, 4);
		R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
		R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
		R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
		R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
		pose_6d = (R.inverse() * T).inverse();
	}
};

//Preception
class Perception {
public:
	double width; // Width of the camera
	double height; // Height of the camera
	double fov_x; // Field of view in x direction
	double fov_y; // Field of view in y direction
	Eigen::Matrix3f intrinsics; // Camera intrinsics

	pcl::PolygonMesh::Ptr mesh_ply; // Object mesh

	pcl::visualization::PCLVisualizer::Ptr viewer; // Viewer

	// Constructor
	Perception (string object_path) {
		// Set camera parameters
		width = 640;
		height = 480;
		fov_x = 0.95;
		fov_y = 0.75;
		double fx = width / (2 * tan(fov_x / 2));
		double fy = height / (2 * tan(fov_y / 2));
		intrinsics << fx, 0, width / 2,
					  0, fy, height / 2,
					  0, 0, 1;
		// cout << "camrea width: " << width << endl;
		// cout << "camrea height: " << height << endl;
		// cout << "camrea fov_x: " << fov_x << endl;
		// cout << "camrea fov_y: " << fov_y << endl;
		// cout << "camrea intrinsics: " << intrinsics << endl;

		// Load object mesh
		mesh_ply = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
		pcl::io::loadPolygonFilePLY(object_path, *mesh_ply);
		if (mesh_ply->cloud.data.empty() || mesh_ply->polygons.empty()) {
			cout << "Load object: " << object_path << " failed!" << endl;
			exit(1);
		}
		// cout << "Load object: " << object_path << " successfully!" << endl;
		//normalize the object
		int mesh_data_offset = mesh_ply->cloud.data.size() / mesh_ply->cloud.width / mesh_ply->cloud.height;
		pcl::PointCloud<pcl::PointXYZ>::Ptr vertex;
		vertex = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromPCLPointCloud2(mesh_ply->cloud, *vertex);
		vector<Eigen::Vector3d> points;
		for (auto& ptr : vertex->points) {
			points.push_back(Eigen::Vector3d(ptr.x, ptr.y, ptr.z));
		}
		Eigen::Vector3d object_center = Eigen::Vector3d(0, 0, 0);
		for (auto& ptr : points) {
			object_center(0) += ptr(0);
			object_center(1) += ptr(1);
			object_center(2) += ptr(2);
		}
		object_center(0) /= points.size();
		object_center(1) /= points.size();
		object_center(2) /= points.size();
		double object_size = 0.0;
		for (auto& ptr : points) {
			object_size = max(object_size, (object_center - ptr).norm());
		}
		double scale = 1.0 / object_size;
		for (int i = 0; i < mesh_ply->cloud.data.size(); i += mesh_data_offset) {
			int arrayPosX = i + mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			X = float((X - object_center(0)) * scale);
			Y = float((Y - object_center(1)) * scale);
			Z = float((Z - object_center(2)) * scale);
			memcpy(&mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
			memcpy(&mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
			memcpy(&mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
		}

		// Create viewer
		viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Render Viewer"));
		viewer->setBackgroundColor(255, 255, 255);
		viewer->initCameraParameters();
		viewer->addPolygonMesh(*mesh_ply, "object");
		viewer->setSize(width, height);
		viewer->spinOnce(100);
		cout << "Create viewer successfully!" << endl;
	}

	// Destructor
	~Perception() {
		viewer->removePolygonMesh("mesh_ply");
		viewer->close();
		cout << "Close viewer successfully!" << endl;
		cout << "Perception destruct successfully!" << endl;
	}

	// render RGB image from a viewpoint
	void render(View view, string image_save_path = "../rgb.png") {
		// Set viewer parameters
		Eigen::Matrix4f extrinsics = view.pose_6d.cast<float>();
		viewer->setCameraParameters(intrinsics, extrinsics);
		viewer->spinOnce(100);
		// Save image
		string test_image_save_path = image_save_path.substr(0, image_save_path.size() - 4) + "_test.png";
		viewer->saveScreenshot(test_image_save_path);
		// Note that pcl may scale the window, use opencv to check the image
		cv::Mat img = cv::imread(test_image_save_path);
		if (img.cols != width || img.rows != height) {
			img = img(cv::Rect(img.cols - width, img.rows - height, width, height));
		}
		cv::imwrite(image_save_path, img);
		remove((test_image_save_path).c_str());
	}
};

//View_Planning_Simulator
class View_Planning_Simulator {
public:
	Perception* perception_simulator; // perception simulator
	cv::Mat dst_img; // target image
	vector<View> selected_views; // selected views
	View output_view;
	vector<cv::Mat> rendered_images; // rendered images
	std::unordered_map<std::string, double> ratio_map;

	// Constructor
	View_Planning_Simulator(Perception* _perception_simulator, View _target_view) {
		perception_simulator = _perception_simulator;
		dst_img = render_view_image(_target_view);
	}

	// Destructor
	~View_Planning_Simulator() {
		cout << "View_Planning_Simulator destruct successfully!" << endl;
	}

	// render view image
	cv::Mat render_view_image(View view) {
		string image_save_path = "./tmp/rgb.png";
		perception_simulator->render(view, image_save_path);
		cv::Mat rendered_image = cv::imread(image_save_path);
		remove((image_save_path).c_str());
		return rendered_image;
	}

	// check if the view is target
	bool is_target(View src_view) {
		Eigen::Vector3d dst_view = Eigen::Vector3d(-0.879024, 0.427971, 0.210138).normalized() * 3.0;
		double dis = (dst_view - src_view.pose_6d.block<3, 1>(0, 3)).norm();
		spdlog::info("distance to target {}", dis);
		return 0;
	}


	bool test_view(View src_view, double max_score) {
		Eigen::Vector3d dst_view = Eigen::Vector3d(-0.879024, 0.427971, 0.210138).normalized() * 3.0;
		double dis = (dst_view - src_view.pose_6d.block<3, 1>(0, 3)).norm();
		// spdlog::info("distance to target {}", dis);
		perception_simulator->render(src_view, "./task2/score/s_" + std::to_string(max_score) + "_dst_" + std::to_string(dis) + ".png");

		return 0;
	}

	// calculate centroid between A,B, and C on a sphere
	View calculate_new_center(const View & A, const View & B, const View & C) {
		Eigen::Vector3d a = A.getCameraPosition();
		Eigen::Vector3d b = B.getCameraPosition();
		Eigen::Vector3d c = C.getCameraPosition();

		a.normalize();
		b.normalize();
		c.normalize();

		Eigen::Vector3d centroid = (a + b + c).normalized();

		// std::cout << "(" << centroid(0) << "," << centroid(1) << "," << centroid(2) << ")" << std::endl;

		View new_center;
		new_center.compute_pose_from_positon_and_object_center(centroid * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));

		return new_center;
	}

	// Apply H to candiate_view on the plane that is intersecting the view and has the candiate view as the support vector
	View applyHomographyToView(const View & can_view, Eigen::Matrix3d& H)
	{
		Eigen::Matrix3d R = can_view.pose_6d.block<3, 3>(0, 0);
		Eigen::Vector3d t = can_view.pose_6d.block<3, 1>(0, 3);

		Eigen::Vector3d n = t.normalized();  // normal vector

		// Construct the plane transformation matrix P
		Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
		P.block<3, 3>(0, 0) = R;  // rotation part remains the same
		P.block<3, 1>(0, 3) = t;  // translation remains the same

		// Apply H on the plane that is parallel to the plane defined by camera pose
		Eigen::Matrix2d H_R = H.block<2, 2>(0, 0);
		Eigen::Vector2d H_t = H.block<2, 1>(0, 2);

		// Project translation vector H_t onto the plane defined by camera pose
		Eigen::Vector3d H_t_3d = Eigen::Vector3d::Identity();
		H_t_3d.head(2) = H_t;  // embed 2D translation into 3D

		// Project H_t_3d onto the plane
		H_t_3d -= H_t_3d.dot(n) * n;  // project onto the plane defined by n

		// Construct the 3D transformation matrix H'
		Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
		transform.block<2, 2>(0, 0) = H_R;  // embed 2D rotation into 3D rotation
		transform.block<3, 1>(0, 3) = H_t_3d;  // embed translated vector into 3D translation

		// Apply transform to the camera pose to get the transformed camera pose
		View applied_view;
		applied_view.pose_6d = transform * P;
		return applied_view;
	}

	
	// Function to calculate sum of absolute differences (SAD) between two images in HSV color space
	double calculateLoss(const cv::Mat& src_img, const cv::Mat& dst_img) {
		cv::Mat diff;
		absdiff(src_img, dst_img, diff);
		
		// Split channels and calculate sum of absolute differences
		std::vector<cv::Mat> channels;
		split(diff, channels);
		
		// Compute total loss as the sum of absolute differences in each channel
		double total_loss = 0.0;
		for (const auto& channel : channels) {
			total_loss += sum(channel)[0];
		}
		
		return total_loss;
	}

	// Function to rotate camera pose around origin (0,0,0) 
	View shift_view(View src_view, double theta, double alpha) {
		View new_view;
		Eigen::Matrix3d R_theta;
		Eigen::Matrix3d R_alpha;

		R_theta << cos(theta), -sin(theta), 0,
				sin(theta),  cos(theta), 0,
						0,         0, 1;
		
		R_alpha << cos(alpha), 0, sin(alpha),
							0, 1,         0,
				-sin(alpha), 0, cos(alpha);
		
		new_view.pose_6d.block<3, 3>(0, 0) = R_theta * R_alpha * src_view.pose_6d.block<3, 3>(0, 0);;

		new_view.pose_6d.block<3, 1>(0, 3) = R_theta * R_alpha * src_view.pose_6d.block<3, 1>(0, 3);;

		return new_view;
	}

	// distance
	View fine_registration_naive(const View & src_view, int max_iterations, double convergence_threshold){	
		cv::Mat can_img;
		double can_loss;
		View can_view;
		
		cv::Mat bst_img;
		double bst_loss;
		View bst_view;

		double max_loss;

		double learning_rate;
		

		bst_loss = std::numeric_limits<double>::infinity();
		bst_view = src_view;
		bst_img = render_view_image(bst_view);
		max_loss = calculateLoss(bst_img, dst_img);

		int iterations = 0;
		while (iterations < max_iterations) {
			iterations++;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					learning_rate = std::min(max_loss / 1'000'000'000.0, 1.0);
					can_view = shift_view(bst_view, dy * 0.1, dy * 0.1);
					can_img = render_view_image(can_view);
					double can_loss = calculateLoss(can_img, dst_img);
					spdlog::info("loss : {}, lr : {}", can_loss, learning_rate);
					is_target(can_view);
					
					if (can_loss < bst_loss) {
						bst_loss = can_loss;
						bst_view = can_view;
					}
				}
			}
			
			// Check for convergence
			if (abs(max_loss - bst_loss) < convergence_threshold)
				break;
			max_loss = bst_loss;
		}
		

		spdlog::info("Before Fine Registration");
		is_target(src_view);
		spdlog::info("After Fine Registration");
		is_target(bst_view);

		return bst_view;
	}

	std::string generateKeyFromPose(const Eigen::Matrix4d &pose) {
		std::ostringstream oss;
		oss << pose;
		return oss.str();
	}

	double compute_score(const View &src_view) {
		const Eigen::Matrix4d &pose = src_view.pose_6d;
		std::string key = generateKeyFromPose(pose);

		auto it = ratio_map.find(key);
		if (it != ratio_map.end()) {
			return it->second;
		}

		cv::Mat src_img = render_view_image(src_view);
		double ratio = computeSIFTMatchRatio(src_img, dst_img, 0.8f, true);

		ratio_map[key] = ratio;

		return ratio;
	}



	View dfs_next_view(const View & A, const View & B, const View & C, double & max_score) {
		View D;
		View max_view;
		View bst_view;
		double bst_score = 0;
		double can_score = 0; // candidate score
		

		
		bst_view = D = calculate_new_center(A,B,C);
		View views[] = { A, B, C, D };
    
		for (const View & can_view : views) {
			can_score = compute_score(can_view);
			if(can_score>bst_score)
				bst_score = can_score;
				bst_view = can_view;
		}
		
		// spdlog::info("bst_score {} vs {} max_score",  bst_score, max_score);

		if (max_score >= bst_score)
			return bst_view;
		max_view = bst_view;
		max_score = bst_score;
		// spdlog::info("going deeper");
		//test_view(max_view, max_score);

		//check for final perfect match if done return early
		std::vector<std::pair<View, View>> edges = {
			{A, B},
			{B, C},
			{C, A}
		};
		
		View can_view;
		can_score = max_score;
		bst_score = max_score;

		for (const auto &edge : edges) {
			// Call new function if better add score if score well 
			can_view = dfs_next_view(edge.first, edge.second, D, can_score);
			
			if (bst_score >= can_score)
				continue;
			
			bst_view = can_view;
			bst_score = can_score;
		}
		
		if (max_score >= bst_score)
			return max_view;
		
		max_score = bst_score;
		max_view = bst_view; //fine_registration_naive(bst_view, 10, 1e-4);
		//src_img = render_view_image(bst_view);
		//max_score = computeSIFTMatchRatio(src_img, dst_img, .8f, true);
		return max_view;
	}


	// depth first search
	void dfs() {
		size_t view_num = 4; // Number of triangles to look at (e.g., 4 for a pyramid with a square base)
		std::vector<View> search_views(view_num + 1); // +1 for the top view

		// Create base views dynamically
		for (size_t i = 0; i < view_num; ++i) {
			double angle = (2.0 * M_PI * i) / view_num;
			Eigen::Vector3d position(std::cos(angle), std::sin(angle), 0.0);
			search_views[i].compute_pose_from_positon_and_object_center(position.normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
		}

		// Create top view
		search_views[view_num].compute_pose_from_positon_and_object_center(Eigen::Vector3d(0.0, 0.0, 1.0) * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));

		View max_view;
		double max_score = 0;
		View can_view;
		double can_score = 0;

		for (size_t i = 0; i < view_num; ++i) {
			size_t next_index = (i + 1) % view_num;
			can_score = 0;
			can_view = dfs_next_view(search_views[i], search_views[next_index], search_views[view_num], can_score);

			spdlog::info("Best score for iteration {} is {}", i, can_score);
			if (can_score > max_score){
				max_view = can_view;
				max_score = can_score;
			}
		}
		output_view = fine_registration_naive(max_view, 100, 1e-4);
	}

};

void task2(string object_name, int test_num) {
	spdlog::info("Version {}", 5);

	View test_view;
	
	// Create a perception simulator
	Perception* perception_simulator = new Perception("./3d_models/" + object_name + ".ply");

	// for each test, select a view
	set<int> selected_view_indices;
	for (int test_id = 0; test_id < test_num; test_id++) {
		View target_view;
		target_view.compute_pose_from_positon_and_object_center(Eigen::Vector3d(-0.879024, 0.427971, 0.210138).normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
	
		View_Planning_Simulator view_planning_simulator(perception_simulator, target_view);
		
		/* spdlog::info("Test");
		test_view.compute_pose_from_positon_and_object_center(Eigen::Vector3d(1, 0, 0).normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
		cout << test_view.pose_6d << endl;
		test_view = view_planning_simulator.shift_view(test_view, 1.5708f, 1.5708f);
		cout << test_view.pose_6d << endl; */

		view_planning_simulator.dfs();
		// Save the selected views
		ofstream fout("./task2/selected_views/" + object_name + "/test_" + to_string(test_id) + ".txt");
		fout << view_planning_simulator.output_view.pose_6d << endl;
		cout << view_planning_simulator.output_view.pose_6d << endl;
		fout.close();
		perception_simulator->render(view_planning_simulator.output_view, "./task2/selected_views/" + object_name + "/final.png");
	}
	// Delete the perception simulator
	delete perception_simulator;
}

int run_level_3()
{
	vector<string> objects;
	objects.push_back("obj_000020");
	// Task 1
	for (auto& object : objects) {
		task2(object, 1);
	}

	return 0;
}

}