#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <json/json.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <opencv2/quality/qualitypsnr.hpp>

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
#include "../include/lightglue.hpp"
#include "../include/image.hpp"
#include "../include/path_and_vis.hpp"


typedef unsigned long long pop_t;

using namespace std;

namespace task2{

Eigen::Vector3d object_center_world = Eigen::Vector3d(1e-100, 1e-100, 1e-100);

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
		if ((Z - Eigen::Vector3d(0, 0, -1)).norm() < 1e-6) { Z = Eigen::Vector3d(1e-100, 1e-100, -1); } //to avoid -Z = (0,0,1) so there will be no X
		if ((Z - Eigen::Vector3d(0, 0, 1)).norm() < 1e-6) { Z = Eigen::Vector3d(1e-100, 1e-100, 1); } //to avoid Z = (0,0,1) so there will be no X
		Eigen::Vector3d X;	 X = (-Z).cross(Eigen::Vector3d(0, 0, 1));	 X = X.normalized(); //change is here from (0,1,0) to (0,0,1)
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
	//LightGlueWrapper lightglue;

	Json::Value cfg;

	// Constructor
	View_Planning_Simulator(Perception* _perception_simulator, View _target_view) {
		perception_simulator = _perception_simulator;
		dst_img = render_view_image(_target_view);

		Config config = Config(); 
		cfg = config.get_config();

		string feature_type = cfg["compute_score"]["feature_type"].as<string>();

		if (feature_type == "light_glue") {
			//lightglue.initialize();
		}
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

	// Fine registration
	View fine_registration_naive(const View& src_view) { 
		cv::Mat can_img;
		double can_loss;
		View can_view;

		cv::Mat bst_img;
		double bst_loss;
		View bst_view;

		double max_loss;

		const int max_iterations 		= cfg["fine_registration"]["max_iterations"].as<int>();
		double convergence_threshold 	= cfg["fine_registration"]["convergence_threshold"].as<double>();
		double learning_rate 			= cfg["fine_registration"]["learning_rate"].as<double>();
		const int max_stagnation 		= cfg["fine_registration"]["max_stagnation"].as<int>();
		const double learning_decay 	= cfg["fine_registration"]["learning_decay"].as<double>();


		bst_view = src_view;
		bst_img = render_view_image(bst_view);
		max_loss = calculateLoss(bst_img, dst_img);
		bst_loss = max_loss;
		double prev_loss = max_loss;
		

		int iterations = 0;
		int stagnation_counter = 0;
		while (iterations < max_iterations) {
			iterations++;
			bool improvement_found = false;

			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					if (dy == 0 && dx == 0) continue;
					can_view = shift_view(bst_view, dy * learning_rate, dx * learning_rate);
					can_img = render_view_image(can_view);
					can_loss = calculateLoss(can_img, dst_img);

					if (can_loss < bst_loss) {
						bst_loss = can_loss;
						bst_view = can_view;
						improvement_found = true;
					}
				}
			}

			if (!improvement_found) {
				learning_rate *= learning_decay;
			} else {
				stagnation_counter = 0;
			}

			spdlog::info("loss: {}, lr: {}", bst_loss, learning_rate);
			is_target(bst_view);

			if (learning_rate < convergence_threshold || abs(prev_loss - bst_loss) < convergence_threshold) {
				stagnation_counter++;
				if (stagnation_counter >= max_stagnation) {
					spdlog::info("registration converged");
					break;
				}
			}

			prev_loss = bst_loss;
			max_loss = bst_loss;
		}

		return bst_view;
	}

	View fine_registration_homography(const View& src_view) { 
		cv::Mat can_img;
		double can_loss;
		View can_view;

		cv::Mat bst_img;
		double bst_loss;
		View bst_view;

		double max_loss;

		const int max_iterations 		= cfg["fine_registration"]["max_iterations"].as<int>();
		double convergence_threshold 	= cfg["fine_registration"]["convergence_threshold"].as<double>();
		double learning_rate 			= cfg["fine_registration"]["learning_rate"].as<double>();
		const int max_stagnation 		= cfg["fine_registration"]["max_stagnation"].as<int>();
		const double learning_decay 	= cfg["fine_registration"]["learning_decay"].as<double>();


		bst_view = src_view;
		bst_img = render_view_image(bst_view);
		max_loss = calculateLoss(bst_img, dst_img);
		bst_loss = max_loss;
		double prev_loss = max_loss;
		

		int iterations = 0;
		int stagnation_counter = 0;
		while (iterations < max_iterations) {
			iterations++;
			bool improvement_found = false;


			can_view = shift_view(bst_view, learning_rate, learning_rate);
			can_img = render_view_image(can_view);
			can_loss = calculateLoss(can_img, dst_img);

			if (can_loss < bst_loss) {
				bst_loss = can_loss;
				bst_view = can_view;
				improvement_found = true;
			}
			

			if (!improvement_found) {
				learning_rate *= learning_decay;
			} else {
				stagnation_counter = 0;
			}

			spdlog::info("loss: {}, lr: {}", bst_loss, learning_rate);
			is_target(bst_view);

			if (learning_rate < convergence_threshold || abs(prev_loss - bst_loss) < convergence_threshold) {
				stagnation_counter++;
				if (stagnation_counter >= max_stagnation) {
					spdlog::info("registration converged");
					break;
				}
			}

			prev_loss = bst_loss;
			max_loss = bst_loss;
		}

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

		string feature_type = cfg["compute_score"]["feature_type"].as<string>();
		bool enable_HSV = cfg["compute_score"]["use_HSV"].as<bool>();

		double ratio;

		spdlog::info("Selected {}",  feature_type);

		if (feature_type == "sift") {
			ratio = compute_match_ratio_SIFT(src_img, dst_img, 0.8f, enable_HSV);
		} else if (feature_type == "light_glue") {
			cv::imwrite("./LightGlue/assets/img_01.jpg", src_img);
			cv::imwrite("./LightGlue/assets/img_02.jpg", dst_img);
			//ratio = lightglue.compute_match_ratio_LIGHTGLUE();
		}

		spdlog::info("ratio {}",  ratio);

		ratio_map[key] = ratio;
		selected_views.push_back(src_view);

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
			is_target(can_view);
			if(can_score>bst_score)
				bst_score = can_score;
				bst_view = can_view;
		}
		
		//spdlog::info("bst_score {} vs {} max_score",  bst_score, max_score);

		if (max_score >= bst_score)
			return bst_view;
		max_view = bst_view;
		max_score = bst_score;
		// spdlog::info("going deeper");
		test_view(max_view, max_score);

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
		
		if (max_score >= bst_score){
			is_target(bst_view);
			return max_view;
		}

		max_score = bst_score;
		max_view = bst_view; //fine_registration_naive(bst_view, 10, 1e-4);
		//src_img = render_view_image(bst_view);
		//max_score = computeSIFTMatchRatio(src_img, dst_img, .8f, true);
		return max_view;
	}


	// depth first search
	void dfs() {
		size_t view_num = cfg["dfs"]["num_corners"].as<size_t>(); // Number of triangles to look at (e.g., 4 for a pyramid with a square base)
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

			//spdlog::info("Best score for iteration {} is {}", i, can_score);
			if (can_score > max_score){
				max_view = can_view;
				max_score = can_score;
			}
		}
		output_view = fine_registration_naive(max_view);
	}

	double get_traversed_distance(){
		double total_length = 0.0;
		for (int i = 1; i < selected_views.size(); i++) {
			pair<int, double> local_path = path_and_vis::get_local_path(selected_views[i-1].pose_6d.block<3, 1>(0, 3), selected_views[i].pose_6d.block<3, 1>(0, 3), (Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), 1.0);
			if (local_path.first < 0) { cout << "local path not found." << endl;}
			total_length += local_path.second;
		}
		return total_length;
	}

	double calculate_SSIM(){
		cv::Mat src_img = render_view_image(output_view);

		cv::Mat Map;
		cv::Scalar scalar = cv::quality::QualitySSIM::compute(src_img, dst_img, Map);
		return scalar[0];
	}

	double calculate_PSNR(){
		cv::Mat src_img = render_view_image(output_view);

		cv::Mat Map;
		cv::Scalar scalar = cv::quality::QualityPSNR::compute(src_img, dst_img, Map);
		return scalar[0];
	}

	void show_view_image_path(string object_path, string pose_file_path, string rgb_file_path) {
		bool highlight_initview = true;
		bool is_show_path = true;
		bool is_global_path = true;
		bool is_show_image = true;
		bool is_show_model = true;
		int vis_num = cfg["visualization"]["vis_num"].as<int>();
		/////////////////////////////////////////////////////////////////
		// Important to modify viewspace path here.                    //
		/////////////////////////////////////////////////////////////////
		//Reading View Space
		Eigen::Vector3d object_center_world = Eigen::Vector3d(1e-100, 1e-100, 1e-100);
		double predicted_size = 1.0;
		vector<View> views;
		ifstream fin(pose_file_path);
		if (fin.is_open()) {
			int num;
			fin >> num;
			for (int i = 0; i < num; i++) {
				Eigen::Vector3d positon;
				fin >> positon[0] >> positon[1] >> positon[2];
				View view;
				view.compute_pose_from_positon_and_object_center(positon, object_center_world);
				views.push_back(view);
			}
			cout << "viewspace readed." << endl;
		}
		else {
			cout << "no view space. check!" << endl;
		}
		//Read selected viewpoints with path sequence
		std::vector<int> chosen_views(vis_num);
		
		for (int i = 0; i < vis_num; ++i) {
			chosen_views[i] = i;
		} // just an example

		// viewer
		auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255, 255, 255);
		//viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		pcl::visualization::Camera cam;
		viewer->getCameraParameters(cam);
		cam.window_size[0] = 1920;
		cam.window_size[1] = 1080;
		viewer->setCameraParameters(cam);
		//setup window viewing pose
		viewer->setCameraPosition(0, 10.0 * sin(35.0 / 180.0 * acos(-1.0)), 10.0 * cos(35.0 / 180.0 * acos(-1.0)), object_center_world(0), object_center_world(1), object_center_world(2), 0, 0, 1);

		//setup camera info
		double fov_x = 0.95;
		double fov_y = 0.75;
		path_and_vis::rs2_intrinsics color_intrinsics;
		color_intrinsics.width = 640;
		color_intrinsics.height = 480;
		color_intrinsics.fx = color_intrinsics.width / (2 * tan(fov_x / 2));
		color_intrinsics.fy = color_intrinsics.height / (2 * tan(fov_y / 2));
		color_intrinsics.ppx = color_intrinsics.width / 2;
		color_intrinsics.ppy = color_intrinsics.height / 2;
		//color_intrinsics.model = RS2_DISTORTION_NONE;
		color_intrinsics.coeffs[0] = 0;
		color_intrinsics.coeffs[1] = 0;
		color_intrinsics.coeffs[2] = 0;
		color_intrinsics.coeffs[3] = 0;
		color_intrinsics.coeffs[4] = 0;

		double view_color[3] = { 0, 0, 255 };
		double path_color[3] = { 128, 0, 128 };

		for (int i = 0; i < chosen_views.size(); i++) {
			Eigen::Matrix4d view_pose_world = views[chosen_views[i]].pose_6d.eval();

			double line_length = 0.3;

			Eigen::Vector3d LeftTop = path_and_vis::project_pixel_to_ray_end(0, 0, color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d RightTop = path_and_vis::project_pixel_to_ray_end(0, 720, color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d LeftBottom = path_and_vis::project_pixel_to_ray_end(1280, 0, color_intrinsics, view_pose_world, line_length);
			Eigen::Vector3d RightBottom = path_and_vis::project_pixel_to_ray_end(1280, 720, color_intrinsics, view_pose_world, line_length);

			Eigen::Vector4d LT(LeftTop(0), LeftTop(1), LeftTop(2), 1);
			Eigen::Vector4d RT(RightTop(0), RightTop(1), RightTop(2), 1);
			Eigen::Vector4d LB(LeftBottom(0), LeftBottom(1), LeftBottom(2), 1);
			Eigen::Vector4d RB(RightBottom(0), RightBottom(1), RightBottom(2), 1);

			Eigen::Vector4d O(0, 0, 0, 1);
			O = view_pose_world * O;

			if (highlight_initview) {
				if (i == 0) {
					view_color[0] = 255;
					view_color[1] = 0;
					view_color[2] = 0;
				}
				else {
					view_color[0] = 0;
					view_color[1] = 0;
					view_color[2] = 255;
				}
			}

			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(LT(0), LT(1), LT(2)), view_color[0], view_color[1], view_color[2], "O-LT" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(RT(0), RT(1), RT(2)), view_color[0], view_color[1], view_color[2], "O-RT" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(LB(0), LB(1), LB(2)), view_color[0], view_color[1], view_color[2], "O-LB" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "O-RB" + to_string(i));

			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(LT(0), LT(1), LT(2)), pcl::PointXYZ(RT(0), RT(1), RT(2)), view_color[0], view_color[1], view_color[2], "LT-RT" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(LT(0), LT(1), LT(2)), pcl::PointXYZ(LB(0), LB(1), LB(2)), view_color[0], view_color[1], view_color[2], "LT-LB" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(RT(0), RT(1), RT(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "RT-RB" + to_string(i));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(LB(0), LB(1), LB(2)), pcl::PointXYZ(RB(0), RB(1), RB(2)), view_color[0], view_color[1], view_color[2], "LB-RB" + to_string(i));

			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-LT" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-RT" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-LB" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "O-RB" + to_string(i));

			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LT-RT" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LT-LB" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "RT-RB" + to_string(i));
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "LB-RB" + to_string(i));

			if (is_show_path) {
				if (i != 0) {
					////////////////////////////////////////////Usage of waypoints
					Eigen::Vector3d now_view_xyz = views[chosen_views[i - 1]].pose_6d.block<3, 1>(0, 3);
					Eigen::Vector3d next_view_xyz = views[chosen_views[i]].pose_6d.block<3, 1>(0, 3);
					vector<Eigen::Vector3d> points;
					int num_of_path = path_and_vis::get_trajectory_xyz(points, now_view_xyz, next_view_xyz, object_center_world, predicted_size, 0.2, 1.0);
					if (num_of_path == -1) {
						cout << "no path. throw" << endl;
						continue;
					}
					if (is_global_path) {
						path_color[0] = 128;
						path_color[1] = 0;
						path_color[2] = 128;
					}
					else {
						path_color[0] = 0;
						path_color[1] = 128;
						path_color[2] = 128;
					}

					viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), path_color[0], path_color[1], path_color[2], "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
					viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
					for (int k = 0; k < points.size() - 1; k++) {
						viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), path_color[0], path_color[1], path_color[2], "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
						viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
					}
				}
			}

			if (is_show_image) {
				//show view image
				cout << "processing " << chosen_views[i] << "th view" << endl;
				/////////////////////////////////////////////////////////////////
				// Important to modify your image path of each viewpoint here. //
				/////////////////////////////////////////////////////////////////
				cv::Mat image = cv::imread(rgb_file_path + "/rgb_" + to_string(chosen_views[i]) + ".png");
				cv::flip(image, image, -1);
				double image_line_length = 0.3;
				int interval = 4;
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_image(new pcl::PointCloud<pcl::PointXYZRGB>);
				for (int x = 0; x < image.cols; x += interval) {
					for (int y = 0; y < image.rows; y += interval) {
						Eigen::Vector3d pixel_end = path_and_vis::project_pixel_to_ray_end(x, y, color_intrinsics, view_pose_world, image_line_length);
						pcl::PointXYZRGB point;
						point.x = pixel_end(0);
						point.y = pixel_end(1);
						point.z = pixel_end(2);
						point.r = image.at<cv::Vec3b>(y, x)[2];
						point.g = image.at<cv::Vec3b>(y, x)[1];
						point.b = image.at<cv::Vec3b>(y, x)[0];
						cloud_image->push_back(point);
					}
				}
				viewer->addPointCloud(cloud_image, "cloud_image" + to_string(chosen_views[i]));
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_image" + to_string(chosen_views[i]));
			}
		}
		if (is_show_model) {
			/////////////////////////////////////////////////////////////////
			// Important to modify object path here.                       //
			/////////////////////////////////////////////////////////////////
			// Load object mesh
			pcl::PolygonMesh::Ptr mesh_ply = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
			pcl::io::loadPolygonFilePLY(object_path, *mesh_ply);
			if (mesh_ply->cloud.data.empty() || mesh_ply->polygons.empty()) {
				cout << "Load object: " << object_path << " failed!" << endl;
				exit(1);
			}
			cout << "Load object: " << object_path << " successfully!" << endl;
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
			viewer->addPolygonMesh(*mesh_ply, "object");
			viewer->spinOnce(100);
		}
		viewer->saveScreenshot (rgb_file_path +"/render.png");

		while (!viewer->wasStopped())
		{
		 	viewer->spinOnce(100);
		 	//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		 }
	}
};

double randomDouble(double min, double max) {
    return min + (max - min) * (static_cast<double>(rand()) / RAND_MAX);
}


void main(string object_name) {
	spdlog::info("Version {}", 8);

	View target_view;
	string object_path = "./3d_models/" + object_name + ".ply";
	
	Config config = Config(); 
	Json::Value cfg = config.get_config();

	int test_num =cfg["main"]["test_num"].as<int>();
	bool enable_dfs =cfg["main"]["enable_dfs"].as<bool>();
	bool enable_generate_imgs =cfg["main"]["enable_generate_imgs"].as<bool>();
	bool enable_meta =cfg["main"]["enable_meta"].as<bool>();
	bool enable_visulize =cfg["main"]["enable_vis"].as<bool>();
	

	Perception* perception_simulator = new Perception(object_path);

	for (int test_id = 0; test_id < test_num; test_id++) {
		string pose_file_path = "./task2/selected_views/" + object_name + "/" + to_string(test_id) + "_views.txt";
		string meta_file_path = "./task2/dfs_meta/" + object_name + "/" + to_string(test_id) + "_meta.txt";
		string rgb_file_path = "./task2/selected_views/" + object_name + "/" + to_string(test_id);
		try {
			std::filesystem::create_directories(rgb_file_path);
		} catch (const std::filesystem::filesystem_error& e) {
			std::cout << "Error creating directory: " << e.what() << '\n';
		}
		target_view.compute_pose_from_positon_and_object_center(Eigen::Vector3d(-0.879024, 0.427971, 0.210138).normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
		//target_view.compute_pose_from_positon_and_object_center(Eigen::Vector3d(randomDouble(-1.0, 1.0), randomDouble(-1.0, 1.0), randomDouble(0.0, 1.0)).normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));

		View_Planning_Simulator view_planning_simulator(perception_simulator, target_view);
		if (enable_dfs){
			auto start = std::chrono::high_resolution_clock::now();
			view_planning_simulator.dfs();
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> duration = end - start;

			// Save the selected views
			//Eigen::Vector3d dst_view = Eigen::Vector3d(-0.879024, 0.427971, 0.210138).normalized() * 3.0;
			double dis = (target_view.pose_6d.block<3, 1>(0, 3) - view_planning_simulator.output_view.pose_6d.block<3, 1>(0, 3)).norm();



			if (enable_meta){
				ofstream fout_meta(meta_file_path);
				fout_meta << "distance to target: " << dis << endl;
				fout_meta << "number of views: " << view_planning_simulator.selected_views.size() << endl;
				fout_meta << "traversed distance: " << view_planning_simulator.get_traversed_distance() << endl;
				fout_meta << "compute time:" << duration.count() << endl;
				fout_meta << "structural similarity (SSIM):" << view_planning_simulator.calculate_SSIM() << endl;
				fout_meta << "peak signal to noise ratio (PSNR):" << view_planning_simulator.calculate_PSNR() << endl;
				fout_meta.close();
			}
			
			if (enable_generate_imgs){
				ofstream fout_pose(pose_file_path);
				fout_pose << view_planning_simulator.selected_views.size() << endl;
				Eigen::Vector3d position;
				for (int i = 0; i < view_planning_simulator.selected_views.size(); i++) {
					position = view_planning_simulator.selected_views[i].pose_6d.block<3, 1>(0, 3);
					fout_pose << position(0) << " " << position(1) << " " << position(2) << endl;
					perception_simulator->render(view_planning_simulator.selected_views[i], rgb_file_path + "/rgb_" + to_string(i) + ".png");
				}
				fout_pose.close();
			}
			perception_simulator->render(view_planning_simulator.output_view, "./task2/selected_views/" + object_name + "/" + to_string(test_id) + "_final.png");
			perception_simulator->render(target_view, "./task2/selected_views/" + object_name + "/" + to_string(test_id) + "_target.png");
		}
		if (enable_visulize){
			view_planning_simulator.show_view_image_path(object_path, pose_file_path, rgb_file_path);
		}
	}
	// Delete the perception simulator
	delete perception_simulator;
}





int run_level_3()
{
	std::vector<string> objects;
	objects.push_back("obj_000019");
	objects.push_back("obj_000020");

	// Task 1
	for (auto& object : objects) {
		try {
			std::filesystem::create_directories("./task2/selected_views/" + object);
		} catch (const std::filesystem::filesystem_error& e) {
			std::cout << "Error creating directory: " << e.what() << '\n';
		}
		main(object);
	}

	return 0;
}

}