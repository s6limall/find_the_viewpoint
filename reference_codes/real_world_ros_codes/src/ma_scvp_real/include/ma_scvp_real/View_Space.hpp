#pragma once
#include <iostream> 
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <time.h>
#include <mutex>
#include <unordered_set>
#include <bitset>

#include <opencv2/opencv.hpp>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gurobi_c++.h>

using namespace std;

inline double get_random_coordinate(double from, double to) {
	//生成比较随机的0-1随机数并映射到区间[from,to]
	double len = to - from;
	long long x = (long long)rand() * ((long long)RAND_MAX + 1) + (long long)rand();
	long long field = (long long)RAND_MAX * (long long)RAND_MAX + 2 * (long long)RAND_MAX;
	return (double)x / (double)field * len + from;
}

class View
{
public:
	Eigen::Vector3d init_pos;	//初始位置
	Eigen::Matrix4d pose;		//view_i到view_i+1旋转矩阵，用于规划路径
	Eigen::Matrix4d pose_fixed;	//固定的旋转矩阵，读取自文件，用于NeRF数据

	View(Eigen::Vector3d _init_pos) {
		init_pos = _init_pos;
		pose = Eigen::Matrix4d::Identity(4, 4);
		pose_fixed = Eigen::Matrix4d::Identity(4, 4);
	}

	View(const View &other) {
		init_pos = other.init_pos;
		pose = other.pose;
		pose_fixed = other.pose_fixed;
	}

	View& operator=(const View& other) {
		init_pos = other.init_pos;
		pose = other.pose;
		pose_fixed = other.pose_fixed;
		return *this;
	}

	~View() {

	}

	//0 simlarest to previous, 1 y-top
	void get_next_camera_pos(Eigen::Matrix4d now_camera_pose_world, Eigen::Vector3d object_center_world, int type_of_pose = 0) {
		switch (type_of_pose) {
			case 0:
			{
				//归一化乘法
				Eigen::Vector4d object_center_now_camera;
				object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
				Eigen::Vector4d view_now_camera;
				view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
				//定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
				Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
				Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
				Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
				//注意左右手系，不要弄反了
				Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
				Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
				Eigen::Matrix4d T(4, 4);
				T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
				T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
				T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
				T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
				Eigen::Matrix4d R(4, 4);
				R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
				R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
				R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
				R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
				//绕Z轴旋转，使得与上一次旋转计算x轴与y轴夹角最小
				Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
				Eigen::Vector4d x(1, 0, 0, 1);
				Eigen::Vector4d y(0, 1, 0, 1);
				Eigen::Vector4d x_ray(1, 0, 0, 1);
				Eigen::Vector4d y_ray(0, 1, 0, 1);
				x_ray = R.inverse() * T * x_ray;
				y_ray = R.inverse() * T * y_ray;
				double min_y = acos(y(1) * y_ray(1));
				double min_x = acos(x(0) * x_ray(0));
				for (double i = 5; i < 360; i += 5) {
					Eigen::Matrix3d rotation;
					rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
						Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
					Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
					Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
					Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
					Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
					Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
					Eigen::Vector4d x_ray(1, 0, 0, 1);
					Eigen::Vector4d y_ray(0, 1, 0, 1);
					x_ray = (R * Rz).inverse() * T * x_ray;
					y_ray = (R * Rz).inverse() * T * y_ray;
					double cos_y = acos(y(1) * y_ray(1));
					double cos_x = acos(x(0) * x_ray(0));
					if (cos_y < min_y) {
						Rz_min = rotation.eval();
						min_y = cos_y;
						min_x = cos_x;
					}
					else if (fabs(cos_y - min_y) < 1e-6 && cos_x < min_x) {
						Rz_min = rotation.eval();
						min_y = cos_y;
						min_x = cos_x;
					}
				}
				Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
				//cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
				Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
				Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
				Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
				Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
				Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
				pose = ((R * Rz).inverse() * T).eval();
				//pose = (R.inverse() * T).eval();
			}
			break;
			case 1:
			{
				//归一化乘法
				Eigen::Vector4d object_center_now_camera;
				object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
				Eigen::Vector4d view_now_camera;
				view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
				//定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
				Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
				Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
				Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
				//注意左右手系，不要弄反了
				Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
				Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
				Eigen::Matrix4d T(4, 4);
				T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
				T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
				T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
				T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
				Eigen::Matrix4d R(4, 4);
				R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
				R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
				R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
				R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
				//绕Z轴旋转，使得y轴最高
				Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
				Eigen::Vector4d y_highest = (now_camera_pose_world * R * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
				for (double i = 5; i < 360; i += 5) {
					Eigen::Matrix3d rotation;
					rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
						Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
						Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
					Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
					Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
					Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
					Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
					Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
					Eigen::Vector4d y_now = (now_camera_pose_world * R * Rz * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
					if (y_now(2) > y_highest(2)) {
						Rz_min = rotation.eval();
						y_highest = y_now;
					}
				}
				Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
				//cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
				Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
				Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
				Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
				Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
				Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
				pose = ((R * Rz).inverse() * T).eval();
				//pose = (R.inverse() * T).eval();
			}
			break;
		}

	}

};

#define ErrorPath -2
#define WrongPath -1
#define LinePath 0
#define CirclePath 1
//return path mode and length from M to N under an circle obstacle with radius r
pair<int, double> get_local_path(Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double r) {
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	//计算直线MN与球O-r的交点PQ
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//如果没有交点或者一个交点，就可以画直线过去
		double d = (N - M).norm();
		//cout << "d: " << d << endl;
		return make_pair(LinePath, d);
	}
	else {
		//如果需要穿过球体，则沿着球表面行动
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//球外两点，直接过去
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			return make_pair(LinePath, d);
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			//起点或终点在障碍物球内
			return make_pair(WrongPath, 1e10);
		}
		if (t3 > t4) {
			double temp = t3;
			t3 = t4;
			t4 = temp;
		}
		x3 = (x2 - x1) * t3 + x1;
		y3 = (y2 - y1) * t3 + y1;
		z3 = (z2 - z1) * t3 + z1;
		Eigen::Vector3d P(x3, y3, z3);
		//cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
		x4 = (x2 - x1) * t4 + x1;
		y4 = (y2 - y1) * t4 + y1;
		z4 = (z2 - z1) * t4 + z1;
		Eigen::Vector3d Q(x4, y4, z4);
		//cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
		//MON平面方程
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//计算参数方程中P,Q的参数值
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta3 = asin(sin_theta3);
		if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
		if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		double x3_theta3, y3_theta3;
		x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		//cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
		if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
			theta3 = acos(-1.0) - theta3;
			if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
			if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		}
		sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta4 = asin(sin_theta4);
		if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
		if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		double x4_theta4, y4_theta4;
		x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		//cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
		if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
			theta4 = acos(-1.0) - theta4;
			if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
			if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		}
		//cout << "theta3: " << theta3 << endl;
		//cout << "theta4: " << theta4 << endl;
		if (theta3 < theta4) flag = 1;
		else flag = -1;
		MP = (M - P).norm();
		QN = (Q - N).norm();
		L = fabs(theta3 - theta4) * r;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "d: " << d << endl;
		return make_pair(CirclePath, d);
	}
	//未定义行为
	return make_pair(ErrorPath, 1e10);
}

double get_trajectory_xyz(vector<Eigen::Vector3d>& points, Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double predicted_size, double distanse_of_pre_move, double camera_to_object_dis) {
	int num_of_path = -1;
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4, r;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	r = predicted_size + camera_to_object_dis;
	//计算直线MN与球O-r的交点PQ
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//如果没有交点或者一个交点，就可以画直线过去
		double d = (N - M).norm();
		//cout << "d: " << d << endl;
		num_of_path = (int)(d / distanse_of_pre_move) + 1;
		//cout << "num_of_path: " << num_of_path << endl;
		double t_pre_point = d / num_of_path;
		for (int i = 1; i <= num_of_path; i++) {
			double di = t_pre_point * i;
			//cout << "di: " << di << endl;
			double xi, yi, zi;
			xi = (x2 - x1) * (di / d) + x1;
			yi = (y2 - y1) * (di / d) + y1;
			zi = (z2 - z1) * (di / d) + z1;
			points.push_back(Eigen::Vector3d(xi, yi, zi));
		}
		return -2;
	}
	else {
		//如果需要穿过球体，则沿着球表面行动
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//球外两点，直接过去
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			num_of_path = (int)(d / distanse_of_pre_move) + 1;
			//cout << "num_of_path: " << num_of_path << endl;
			double t_pre_point = d / num_of_path;
			for (int i = 1; i <= num_of_path; i++) {
				double di = t_pre_point * i;
				//cout << "di: " << di << endl;
				double xi, yi, zi;
				xi = (x2 - x1) * (di / d) + x1;
				yi = (y2 - y1) * (di / d) + y1;
				zi = (z2 - z1) * (di / d) + z1;
				points.push_back(Eigen::Vector3d(xi, yi, zi));
			}
			return num_of_path;
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			cout << "has one viewport in circle. check?" << endl;
			return -1;
		}
		if (t3 > t4) {
			double temp = t3;
			t3 = t4;
			t4 = temp;
		}
		x3 = (x2 - x1) * t3 + x1;
		y3 = (y2 - y1) * t3 + y1;
		z3 = (z2 - z1) * t3 + z1;
		Eigen::Vector3d P(x3, y3, z3);
		//cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
		x4 = (x2 - x1) * t4 + x1;
		y4 = (y2 - y1) * t4 + y1;
		z4 = (z2 - z1) * t4 + z1;
		Eigen::Vector3d Q(x4, y4, z4);
		//cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
		//MON平面方程
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//计算参数方程中P,Q的参数值
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta3 = asin(sin_theta3);
		if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
		if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		double x3_theta3, y3_theta3;
		x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		//cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
		if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
			theta3 = acos(-1.0) - theta3;
			if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
			if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		}
		sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta4 = asin(sin_theta4);
		if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
		if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		double x4_theta4, y4_theta4;
		x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		//cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
		if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
			theta4 = acos(-1.0) - theta4;
			if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
			if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		}
		//cout << "theta3: " << theta3 << endl;
		//cout << "theta4: " << theta4 << endl;
		if (theta3 < theta4) flag = 1;
		else flag = -1;
		MP = (M - P).norm();
		QN = (Q - N).norm();
		L = fabs(theta3 - theta4) * r;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "d: " << d << endl;
		num_of_path = (int)(d / distanse_of_pre_move) + 1;
		//cout << "num_of_path: " << num_of_path << endl;
		double t_pre_point = d / num_of_path;
		bool on_ground = true;
		for (int i = 1; i <= num_of_path; i++) {
			double di = t_pre_point * i;
			//cout << "di: " << di << endl;
			double xi, yi, zi;
			if (di <= MP || di >= MP + L) {
				xi = (x2 - x1) * (di / d) + x1;
				yi = (y2 - y1) * (di / d) + y1;
				zi = (z2 - z1) * (di / d) + z1;
			}
			else {
				double di_theta = di - MP;
				double theta_i = flag * di_theta / r + theta3;
				//cout << "theta_i: " << theta_i << endl;
				xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				if (zi < 0.05) {
					//如果路径点高度为负值，表示求解的路径是下半部分球体，翻转方向
					on_ground = false;
					break;
				}
			}
			points.push_back(Eigen::Vector3d(xi, yi, zi));
		}
		//return d;
		if (!on_ground) {
			cout << "Another way." << endl;
			L = 2.0 * acos(-1.0) * r - fabs(theta3 - theta4) * r;
			//cout << "L: " << L << endl;
			d = MP + L + QN;
			//cout << "d: " << d << endl;
			num_of_path = (int)(d / distanse_of_pre_move) + 1;
			//cout << "num_of_path: " << num_of_path << endl;
			t_pre_point = d / num_of_path;
			flag = -flag;
			points.clear();
			for (int i = 1; i <= num_of_path; i++) {
				double di = t_pre_point * i;
				//cout << "di: " << di << endl;
				double xi, yi, zi;
				if (di <= MP || di >= MP + L) {
					xi = (x2 - x1) * (di / d) + x1;
					yi = (y2 - y1) * (di / d) + y1;
					zi = (z2 - z1) * (di / d) + z1;
				}
				else {
					double di_theta = di - MP;
					double theta_i = flag * di_theta / r + theta3;
					//cout << "theta_i: " << theta_i << endl;
					xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
					yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
					zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
				}
				points.push_back(Eigen::Vector3d(xi, yi, zi));
			}
		}
	}
	return num_of_path;
}

class View_Space
{
public:
	int num_of_views;							//视点个数
	vector<View> views;							//空间的采样视点
	shared_ptr<Share_Data> share_data;			//共享数据

	View_Space(shared_ptr<Share_Data>& _share_data) {
		share_data = _share_data;
		num_of_views = share_data->num_of_views;

		for (int i = 0; i < share_data->pt_sphere.size(); i++) {
			double scale = 1.0 / share_data->pt_norm * share_data->view_space_radius; //predicted_size * sqrt(3) * ?
			View view(Eigen::Vector3d(share_data->pt_sphere[i][0] * scale + share_data->object_center_world(0), share_data->pt_sphere[i][1] * scale + share_data->object_center_world(1), share_data->pt_sphere[i][2] * scale + share_data->object_center_world(2) + share_data->up_shift));
			views.push_back(view);
		}

		cout << "view_space " << views.size() << " getted." << endl;
	}

	~View_Space() {
		//释放内存
		views.clear();
		views.shrink_to_fit();
	}
};


//Global_Path_Planner.hpp

/* Solve a traveling salesman problem on a randomly generated set of
	points using lazy constraints.   The base MIP model only includes
	'degree-2' constraints, requiring each node to have exactly
	two incident edges.  Solutions to this model may contain subtours -
	tours that don't visit every node.  The lazy constraint callback
	adds new constraints to cut them off. */

	// Given an integer-feasible solution 'sol', find the smallest
	// sub-tour.  Result is returned in 'tour', and length is
	// returned in 'tourlenP'.
void findsubtour(int n, double** sol, int* tourlenP, int* tour) {
	{
		bool* seen = new bool[n];
		int bestind, bestlen;
		int i, node, len, start;

		for (i = 0; i < n; i++)
			seen[i] = false;

		start = 0;
		bestlen = n + 1;
		bestind = -1;
		node = 0;
		while (start < n) {
			for (node = 0; node < n; node++)
				if (!seen[node])
					break;
			if (node == n)
				break;
			for (len = 0; len < n; len++) {
				tour[start + len] = node;
				seen[node] = true;
				for (i = 0; i < n; i++) {
					if (sol[node][i] > 0.5 && !seen[i]) {
						node = i;
						break;
					}
				}
				if (i == n) {
					len++;
					if (len < bestlen) {
						bestlen = len;
						bestind = start;
					}
					start += len;
					break;
				}
			}
		}

		for (i = 0; i < bestlen; i++)
			tour[i] = tour[bestind + i];
		*tourlenP = bestlen;

		delete[] seen;
	}
}

// Subtour elimination callback.  Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class subtourelim : public GRBCallback
{
public:
	GRBVar** vars;
	int n;
	subtourelim(GRBVar** xvars, int xn) {
		vars = xvars;
		n = xn;
	}
protected:
	void callback() {
		try {
			if (where == GRB_CB_MIPSOL) {
				// Found an integer feasible solution - does it visit every node?
				double** x = new double* [n];
				int* tour = new int[n];
				int i, j, len;
				for (i = 0; i < n; i++)
					x[i] = getSolution(vars[i], n);

				findsubtour(n, x, &len, tour);

				if (len < n) {
					// Add subtour elimination constraint
					GRBLinExpr expr = 0;
					for (i = 0; i < len; i++)
						for (j = i + 1; j < len; j++)
							expr += vars[tour[i]][tour[j]];
					addLazy(expr <= len - 1);
				}

				for (i = 0; i < n; i++)
					delete[] x[i];
				delete[] x;
				delete[] tour;
			}
		}
		catch (GRBException e) {
			cout << "Error number: " << e.getErrorCode() << endl;
			cout << e.getMessage() << endl;
		}
		catch (...) {
			cout << "Error during callback" << endl;
		}
	}
};

class Global_Path_Planner {
public:
	shared_ptr<Share_Data> share_data;
	int now_view_id;
	int end_view_id;
	bool solved;
	int n;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	double total_shortest;
	vector<int> global_path;
	GRBEnv* env = NULL;
	GRBVar** vars = NULL;
	GRBModel* model = NULL;
	subtourelim* cb = NULL;

	Global_Path_Planner(shared_ptr<Share_Data> _share_data, vector<View>& views, vector<int>& view_set_label, int _now_view_id, int _end_view_id = -1) {
		share_data = _share_data;
		now_view_id = _now_view_id;
		end_view_id = _end_view_id;
		solved = false;
		total_shortest = -1;
		//构造下标映射
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i;
			(*view_id_out)[i] = view_set_label[i];
		}
		(*view_id_in)[views.size()] = view_set_label.size(); //注意复制节点应该是和视点空间个数相关，映射到所需视点个数
		(*view_id_out)[view_set_label.size()] = views.size();
		//节点数为原始个数+起点的复制节点
		n = view_set_label.size() + 1;
		//local path 完全无向图
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				if (i == n - 1 || j == n - 1) {
					//如果是起点的复制节点，距离为0
					graph[i][j] = 0.0;
				}
				else {
					//交换id
					int u = (*view_id_out)[i];
					int v = (*view_id_out)[j];
					//求两点路径
					pair<int, double> local_path = get_local_path(views[u].init_pos.eval(), views[v].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
					if (local_path.first < 0) {
						cout << "local path not found." << endl;
						graph[i][j] = 1e10;
					}
					else graph[i][j] = local_path.second;
				}
				//cout << "graph[" << i << "][" << j << "] = " << graph[i][j] << endl;
			}
		//创建Gurobi的TSP优化器
		vars = new GRBVar * [n];
		for (int i = 0; i < n; i++)
			vars[i] = new GRBVar[n];
		env = new GRBEnv();
		model = new GRBModel(*env);
		//cout << "Gurobi model created." << endl;
		// Must set LazyConstraints parameter when using lazy constraints
		model->set(GRB_IntParam_LazyConstraints, 1);
		//cout << "Gurobi set LazyConstraints." << endl;
		// Create binary decision variables
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				vars[i][j] = model->addVar(0.0, 1.0, graph[i][j], GRB_BINARY, "x_" + to_string(i) + "_" + to_string(j));
				vars[j][i] = vars[i][j];
			}
		}
		//cout << "Gurobi addVar complete." << endl;
		// Degree-2 constraints
		for (int i = 0; i < n; i++) {
			GRBLinExpr expr = 0;
			for (int j = 0; j < n; j++)
				expr += vars[i][j];
			model->addConstr(expr == 2, "deg2_" + to_string(i));
		}
		//cout << "Gurobi add Degree-2 Constr complete." << endl;
		// Forbid edge from node back to itself
		for (int i = 0; i < n; i++)
			vars[i][i].set(GRB_DoubleAttr_UB, 0);
		//cout << "Gurobi set Forbid edge complete." << endl;
		// Make copy node linked to starting node
		vars[n - 1][(*view_id_in)[now_view_id]].set(GRB_DoubleAttr_LB, 1);
		// 默认不设置终点，多解只返回一个
		if (end_view_id != -1) vars[(*view_id_in)[end_view_id]][n - 1].set(GRB_DoubleAttr_LB, 1);
		//cout << "Gurobi set Make copy node complete." << endl;
		// Set callback function
		cb = new subtourelim(vars, n);
		model->setCallback(cb);
		//cout << "Gurobi set callback function complete." << endl;
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		graph.clear();
		graph.shrink_to_fit();
		global_path.clear();
		global_path.shrink_to_fit();
		for (int i = 0; i < n; i++)
			delete[] vars[i];
		delete[] vars;
		delete env;
		delete model;
		delete cb;
	}

	double solve() {
		double now_time = clock();
		// Optimize model
		model->optimize();
		// Extract solution
		if (model->get(GRB_IntAttr_SolCount) > 0) {
			solved = true;
			total_shortest = 0.0;
			double** sol = new double* [n];
			for (int i = 0; i < n; i++)
				sol[i] = model->get(GRB_DoubleAttr_X, vars[i], n);

			int* tour = new int[n];
			int len;

			findsubtour(n, sol, &len, tour);
			assert(len == n);

			//cout << "Tour: ";
			for (int i = 0; i < len; i++) {
				global_path.push_back(tour[i]);
				if (i != len - 1) {
					total_shortest += graph[tour[i]][tour[i + 1]];
				}
				else {
					total_shortest += graph[tour[i]][tour[0]];
				}
				//cout << tour[i] << " ";
			}
			//cout << endl;

			for (int i = 0; i < n; i++)
				delete[] sol[i];
			delete[] sol;
			delete[] tour;
		}
		else {
			cout << "No solution found" << endl;
		}
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		vector<int> ans = global_path;
		//调准顺序把复制的起点置于末尾
		int copy_node_id = -1;
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				copy_node_id = i;
				break;
			}
		}
		if (copy_node_id == -1) {
			cout << "copy_node_id == -1" << endl;
		}
		for (int i = 0; i < copy_node_id; i++) {
			ans.push_back(ans[0]);
			ans.erase(ans.begin());
		}
		//删除复制的起点
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				ans.erase(ans.begin() + i);
				break;
			}
		}
		//如果起点是第一个就不动，是最后一个就反转
		if (ans[0] != (*view_id_in)[now_view_id]) {
			reverse(ans.begin(), ans.end());
		}
		for (int i = 0; i < ans.size(); i++) {
			ans[i] = (*view_id_out)[ans[i]];
			cout << ans[i] << " ";
		}
		cout << endl;
		//删除出发点
		//ans.erase(ans.begin());
		return ans;
	}
};
