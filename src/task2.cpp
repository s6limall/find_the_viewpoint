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
		cout << "camrea width: " << width << endl;
		cout << "camrea height: " << height << endl;
		cout << "camrea fov_x: " << fov_x << endl;
		cout << "camrea fov_y: " << fov_y << endl;
		cout << "camrea intrinsics: " << intrinsics << endl;

		// Load object mesh
		mesh_ply = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
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
	cv::Mat target_image; // target image
	vector<View> selected_views; // selected views
	vector<cv::Mat> rendered_images; // rendered images

	// Constructor
	View_Planning_Simulator(Perception* _perception_simulator, View _target_view) {
		perception_simulator = _perception_simulator;
		target_image = render_view_image(_target_view);
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
	bool is_target(View view) {
		cv::Mat rendered_image = render_view_image(view);
		return compareImages(rendered_image, target_image);
	}

	View calculate_new_center(const View & A, const View & B, const View & C) {
		Eigen::Vector3d a = A.getCameraPosition();
		Eigen::Vector3d b = B.getCameraPosition();
		Eigen::Vector3d c = C.getCameraPosition();

		a.normalize();
		b.normalize();
		c.normalize();

		Eigen::Vector3d centroid = (a + b + c).normalized();

		std::cout << "(" << centroid(0) << "," << centroid(1) << "," << centroid(2) << ")" << std::endl;

		View new_center;
		new_center.compute_pose_from_positon_and_object_center(centroid * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));

		return new_center;
	}


	View dfs_next_view(const View & A, const View & B, const View & C, size_t & best_score) {
		View D = calculate_new_center(A,B,C);
		View best_view = D;
		size_t tmp_score = 0;
		size_t max_score = 0;
		cv::Mat candidate_image;

		View views[] = { A, B, C, D };
    
		for (const View & view : views) {
			candidate_image = render_view_image(view);
			tmp_score += computeSIFTMatches(target_image, candidate_image);
			if(tmp_score>max_score)
				max_score=tmp_score;
		}
		
		spdlog::info("max_score {} vs {} best_score",  max_score, best_score);

		if (best_score >= max_score)
			return best_view;
		spdlog::info("going deeper");
		
		best_score = max_score;

		//check for final perfect match if done return early

		std::vector<std::pair<View, View>> edges = {
			{A, B},
			{B, C},
			{C, D}
		};
		
		View candidate_view;
		for (const auto &edge : edges) {
			// Call new function if better add score if score well 
			candidate_view = dfs_next_view(edge.first, edge.second, D, best_score);
			
			if (max_score > best_score)
				continue;
			best_view = candidate_view;
		}

		return best_view;
	}


	// depth first search
	void dfs() {
		// Number of triangles to look at (e.g., 4 for a pyramid with a square base)

		size_t base_view_count = 4; // You can change this to any number
		std::vector<View> search_views(base_view_count + 1); // +1 for the top view

		// Create base views dynamically
		for (size_t i = 0; i < base_view_count; ++i) {
			double angle = (2.0 * M_PI * i) / base_view_count;
			Eigen::Vector3d position(std::cos(angle), std::sin(angle), 0.0);
			search_views[i].compute_pose_from_positon_and_object_center(position.normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
		}

		// Create top view
		search_views[base_view_count].compute_pose_from_positon_and_object_center(Eigen::Vector3d(0.0, 0.0, 1.0) * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));

		size_t best_score = 0;
		size_t tmp_score = 0;
		View best_view;

		for (size_t i = 0; i < base_view_count; ++i) {
			size_t next_index = (i + 1) % base_view_count;
			tmp_score = 0;
			best_view = dfs_next_view(search_views[i], search_views[next_index], search_views[base_view_count], tmp_score);

			spdlog::info("Best score for iteration {} is {}", i, tmp_score);
			if (tmp_score > best_score){
				selected_views.push_back(best_view);
				best_score = tmp_score;
			}
		}
	}

};

void task2(string object_name, int test_num) {
	// Create a perception simulator
	Perception* perception_simulator = new Perception("./3d_models/" + object_name + ".ply");

	// for each test, select a view
	set<int> selected_view_indices;
	for (int test_id = 0; test_id < test_num; test_id++) {
		View target_view;
		target_view.compute_pose_from_positon_and_object_center(Eigen::Vector3d(5.0, 0.0, 5.0).normalized() * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
	
		View_Planning_Simulator view_planning_simulator(perception_simulator, target_view);
		view_planning_simulator.dfs();
		// Save the selected views
		ofstream fout("./task2/selected_views/" + object_name + "/test_" + to_string(test_id) + ".txt");
		fout << view_planning_simulator.selected_views.size() << endl;
		for (int i = 0; i < view_planning_simulator.selected_views.size(); i++) {
			fout << view_planning_simulator.selected_views[i].pose_6d << endl;
			cout << view_planning_simulator.selected_views[i].pose_6d << endl;
		}
		cout << target_view.pose_6d << endl;
		fout.close();
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