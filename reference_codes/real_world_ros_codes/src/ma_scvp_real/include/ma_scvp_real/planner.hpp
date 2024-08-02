#include <iostream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include <jsoncpp/json/json.h>

//NBV_Planner.hpp
class NBV_Planner
{
public:
	//read
	shared_ptr<Share_Data> share_data;
	shared_ptr<View_Space> view_space;
	vector<View> init_views;
	int first_init_view_id;
	vector<int> init_view_ids;
	int first_view_id;
	int test_id;

	//create
	Json::Value root_nbvs;
	Json::Value root_render;
	vector<int> chosen_nbvs;
	set<int> chosen_nbvs_set;
	vector<int> oneshot_views;
	// double now_time;
	double total_movement_cost;
	int iteration;

	~NBV_Planner() {
		share_data.reset();
		view_space.reset();
		init_views.clear();
		init_views.shrink_to_fit();
		chosen_nbvs.clear();
		chosen_nbvs.shrink_to_fit();
		chosen_nbvs_set.clear();
		oneshot_views.clear();
		oneshot_views.shrink_to_fit();
	}

	//first_view_id是测试视点空间中0 0 1的id，init_view_ids是5覆盖情况下的初始视点集合
	NBV_Planner(shared_ptr<Share_Data>& _share_data, vector<View>& _init_views, int _first_init_view_id) {
		cout << "init view planner." << endl;
		srand(clock());
		share_data = _share_data;
		init_views = _init_views;
		first_init_view_id = _first_init_view_id;
		view_space = make_shared<View_Space>(share_data);
		//specify test_id
		test_id = share_data->test_id;
		//specify init view ids
		init_view_ids = share_data->init_view_ids;
		//get top view id
		first_view_id = -1;
		for (int i = 0; i < share_data->pt_sphere.size(); i++) {
			Eigen::Vector3d test_pos = Eigen::Vector3d(share_data->pt_sphere[i][0]/ share_data->pt_norm, share_data->pt_sphere[i][1]/ share_data->pt_norm, share_data->pt_sphere[i][2]/ share_data->pt_norm);
			if (fabs(test_pos(0)) < 1e-6 && fabs(test_pos(1)) < 1e-6 && fabs(test_pos(2) - 1.0) < 1e-6) {
				first_view_id = i;
			}
		}

		//check input status
		if (init_views.size() == 0) {
			cout << "init_views is empty. read init (5) coverage view space first." << endl;
			return;
		}
		if (first_view_id == -1) {
			first_view_id = 0;
			cout << "first_view_id is -1. use 0 as id." << endl;
		}
		if (init_view_ids.size() == 0) {
			init_view_ids.push_back(1);
			cout << "init_view_ids is empty. use 5 coverage view space top id." << endl;
		}
		if (share_data->method_of_IG != PVBCoverage) {
			cout << "method_of_IG is not PVBCoverage. Read view budget." << endl;
			ifstream fin_view_budget(share_data->pre_path + "Compare/" + share_data->name_of_pcd + "_m4_v" + to_string(init_view_ids.size()) + "_t" + to_string(test_id) + "/view_budget.txt");
			if (fin_view_budget.is_open()) {
				int view_budget;
				fin_view_budget >> view_budget;
				fin_view_budget.close();
				share_data->num_of_max_iteration = view_budget - 1;
				cout << "readed view_budget is " << view_budget << endl;
			}
			else {
				cout << "view_budget.txt is not exist. use deaulft as view budget." << endl;
			}
			cout << "num_of_max_iteration is set as " << share_data->num_of_max_iteration << endl;
		}

		//mkdir
		share_data->access_directory(share_data->save_path + "/json");
		share_data->access_directory(share_data->save_path + "/render_json");
		share_data->access_directory(share_data->save_path + "/metrics");
		share_data->access_directory(share_data->save_path + "/render");
		share_data->access_directory(share_data->save_path + "/train_time");
		share_data->access_directory(share_data->save_path + "/infer_time");
		share_data->access_directory(share_data->save_path + "/movement");

		//json root_nbvs
		root_nbvs["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_nbvs["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_nbvs["fl_x"] = share_data->color_intrinsics.fx;
		root_nbvs["fl_y"] = share_data->color_intrinsics.fy;
		root_nbvs["k1"] = share_data->color_intrinsics.coeffs[0];
		root_nbvs["k2"] = share_data->color_intrinsics.coeffs[1];
		root_nbvs["k3"] = share_data->color_intrinsics.coeffs[2];
		root_nbvs["p1"] = share_data->color_intrinsics.coeffs[3];
		root_nbvs["p2"] = share_data->color_intrinsics.coeffs[4];
		root_nbvs["cx"] = share_data->color_intrinsics.ppx;
		root_nbvs["cy"] = share_data->color_intrinsics.ppy;
		root_nbvs["w"] = share_data->color_intrinsics.width;
		root_nbvs["h"] = share_data->color_intrinsics.height;
		root_nbvs["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_nbvs["scale"] = 1;
		root_nbvs["offset"][0] = 0.5 + fabs(share_data->object_center_world(2)) / share_data->view_space_radius;
		root_nbvs["offset"][1] = 0.5;
		root_nbvs["offset"][2] = 0.5;
		root_nbvs["near_distance"] = (share_data->view_space_radius- share_data->predicted_size) / share_data->view_space_radius;
		//json root_render
		root_render["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_render["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_render["fl_x"] = share_data->color_intrinsics.fx / 16.0;
		root_render["fl_y"] = share_data->color_intrinsics.fy / 16.0;
		root_render["k1"] = 0;
		root_render["k2"] = 0;
		root_render["k3"] = 0;
		root_render["p1"] = 0;
		root_render["p2"] = 0;
		root_render["cx"] = share_data->color_intrinsics.ppx / 16.0;
		root_render["cy"] = share_data->color_intrinsics.ppy / 16.0;
		root_render["w"] = share_data->color_intrinsics.width / 16.0;
		root_render["h"] = share_data->color_intrinsics.height / 16.0;
		root_render["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_render["scale"] = 1;
		root_render["offset"][0] = 0.5 + fabs(share_data->object_center_world(2)) / share_data->view_space_radius;
		root_render["offset"][1] = 0.5;
		root_render["offset"][2] = 0.5;
		root_render["near_distance"] = (share_data->view_space_radius- share_data->predicted_size) / share_data->view_space_radius;

		//5覆盖初始化中除了0 0 1视点外的视点
		for (int i = 0; i < init_view_ids.size(); i++) {
			if (first_init_view_id == init_view_ids[i]) {
				continue;
			}
			//get json
			Json::Value view_image;
			view_image["file_path"] = "../../Coverage_images/" + share_data->name_of_pcd + "/" + to_string(init_views.size()) + "/rgbRembg_" + to_string(init_view_ids[i]) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					//init_views[init_view_ids[i]].get_next_camera_pos(share_data->ref_camera_pose_world, share_data->object_center_world);
					//Eigen::Matrix4d view_pose_world = share_data->ref_camera_pose_world * init_views[init_view_ids[i]].pose.inverse();
					Eigen::Matrix4d view_pose_world = init_views[init_view_ids[i]].pose_fixed;
					//把视点空间的坐标系转换到json的坐标系，即移动到中心，然后缩放到1.0
					view_pose_world(0, 3) = view_pose_world(0, 3) - share_data->object_center_world(0);
					view_pose_world(1, 3) = view_pose_world(1, 3) - share_data->object_center_world(1);
					view_pose_world(2, 3) = view_pose_world(2, 3) - share_data->object_center_world(2) - share_data->up_shift;
					view_pose_world(0, 3) = view_pose_world(0, 3) / share_data->view_space_radius;
					view_pose_world(1, 3) = view_pose_world(1, 3) / share_data->view_space_radius;
					view_pose_world(2, 3) = view_pose_world(2, 3) / share_data->view_space_radius;
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root_nbvs["frames"].append(view_image);
		}

		//set up nbvs
		chosen_nbvs.push_back(first_view_id);
		chosen_nbvs_set.insert(first_view_id);

		//set up iter
		// now_time = clock();
		iteration = 0;
		total_movement_cost = 0.0;


	}

	int train_by_instantNGP(string trian_json_file, string test_json_file = "100", bool nbv_test = false, int ensemble_id = -1) {
		double now_time = clock();
		//使用命令行训练
		ofstream fout_py(share_data->instant_ngp_path + "interact/run_with_c++.py");

		fout_py << "import os" << endl;

		string cmd = "python " + share_data->instant_ngp_path + "run.py";
		//cmd += " --gui";
		cmd += " --train";
		cmd += " --n_steps " + to_string(share_data->n_steps);

		if (!nbv_test) {
			cmd += " --scene " + share_data->gt_path + "/" + trian_json_file + ".json ";
			cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
			cmd += " --save_metrics " + share_data->gt_path + "/" + trian_json_file + ".txt ";
		}
		else {
			cmd += " --scene " + share_data->save_path + "/json/" + trian_json_file + ".json ";
			if (ensemble_id == -1) {
				cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
				cmd += " --save_metrics " + share_data->save_path + "/metrics/" + trian_json_file + ".txt ";
			}
			else {
				cmd += " --screenshot_transforms " + share_data->save_path + "/render_json/" + trian_json_file + ".json ";
				cmd += " --screenshot_dir " + share_data->save_path + "/render/" + trian_json_file + "/ensemble_" + to_string(ensemble_id) + "/";
			}
		}

		string python_cmd = "os.system(\'" + cmd + "\')";
		fout_py << python_cmd << endl;
		fout_py.close();

		ofstream fout_py_ready(share_data->instant_ngp_path + "interact/ready_c++.txt");
		fout_py_ready.close();

		ifstream fin_over(share_data->instant_ngp_path + "interact/ready_py.txt");
		while (!fin_over.is_open()) {
			boost::this_thread::sleep(boost::posix_time::seconds(1));
			fin_over.open(share_data->instant_ngp_path + "interact/ready_py.txt");
		}
		fin_over.close();
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		remove((share_data->instant_ngp_path + "interact/ready_py.txt").c_str());

		double cost_time = (clock() - now_time) / CLOCKS_PER_SEC;
		cout << "train and eval with executed time " << cost_time << " s." << endl;

		if (nbv_test) {
			if (ensemble_id == -1) {
				ofstream fout_time(share_data->save_path + "/train_time/" + trian_json_file + ".txt");
				fout_time << cost_time << endl;
				fout_time.close();
			}
		}

		return 0;
	}

	//返回是否需要迭代
	bool loop_once() {

		//生成当前视点集合json
		Json::Value now_nbvs_json(root_nbvs);
		Json::Value now_render_json(root_render);
		for (int i = 0; i < share_data->num_of_views; i++) {
			Json::Value view_image;
			view_image["file_path"] = "../../Coverage_images/" + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "/rgbRembg_" + to_string(i) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					//view_space->views[i].get_next_camera_pos(share_data->ref_camera_pose_world, share_data->object_center_world);
					//Eigen::Matrix4d view_pose_world = share_data->ref_camera_pose_world * view_space->views[i].pose.inverse();
					Eigen::Matrix4d view_pose_world = view_space->views[i].pose_fixed;
					//把视点空间的坐标系转换到json的坐标系，即移动到中心，然后x,y,z->y,z,x
					view_pose_world(0, 3) = view_pose_world(0, 3) - share_data->object_center_world(0);
					view_pose_world(1, 3) = view_pose_world(1, 3) - share_data->object_center_world(1);
					view_pose_world(2, 3) = view_pose_world(2, 3) - share_data->object_center_world(2) - share_data->up_shift;
					view_pose_world(0, 3) = view_pose_world(0, 3) / share_data->view_space_radius;
					view_pose_world(1, 3) = view_pose_world(1, 3) / share_data->view_space_radius;
					view_pose_world(2, 3) = view_pose_world(2, 3) / share_data->view_space_radius;
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			if (chosen_nbvs_set.count(i)) now_nbvs_json["frames"].append(view_image);
			else now_render_json["frames"].append(view_image);
		}
		Json::StyledWriter writer_nbvs_json;
		ofstream fout_nbvs_json(share_data->save_path + "/json/" + to_string(iteration) + ".json");
		fout_nbvs_json << writer_nbvs_json.write(now_nbvs_json);
		fout_nbvs_json.close();
		Json::StyledWriter writer_render_json;
		ofstream fout_render_json(share_data->save_path + "/render_json/" + to_string(iteration) + ".json");
		fout_render_json << writer_render_json.write(now_render_json);
		fout_render_json.close();

		//如果需要测试，则训练当前视点集合
		cout << "iteration " << iteration << endl;
		cout << "chosen_nbvs: ";
		for (int i = 0; i < chosen_nbvs.size(); i++) {
			cout << chosen_nbvs[i] << ' ';
		}
		cout << endl;
		//if (share_data->evaluate) {
		//	cout << "evaluating..." << endl;
		//	train_by_instantNGP(to_string(iteration), "100", true);
		//	ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
		//	string metric_name;
		//	double metric_value;
		//	while (fin_metrics >> metric_name >> metric_value) {
		//		cout << metric_name << ": " << metric_value << endl;
		//	}
		//	fin_metrics.close();
		//}

		//如果达到最大迭代次数，则结束
		if (iteration == share_data->num_of_max_iteration) {
			// //保存运行时间
			// double loops_time = (clock() - now_time) / CLOCKS_PER_SEC;
			// ofstream fout_loops_time(share_data->save_path + "/run_time.txt");
			// fout_loops_time << loops_time << endl;
			// fout_loops_time.close();
			// cout << "run time " << loops_time << " s." << endl;
			//如果不需要逐步测试，则训练最终视点集合
			if (!share_data->evaluate) {
				cout << "final evaluating..." << endl;
				train_by_instantNGP(to_string(iteration), "100", true);
				ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
				string metric_name;
				double metric_value;
				while (fin_metrics >> metric_name >> metric_value) {
					cout << metric_name << ": " << metric_value << endl;
				}
				fin_metrics.close();
			}
			//返回是否需要迭代
			return false;
		}

		//根据不同方法获取NBV
		double infer_time = clock();
		int next_view_id = -1;
		double largest_view_uncertainty = -1e100;
		int best_view_id = -1;
		switch (share_data->method_of_IG) {
		case 0: //RandomIterative
			next_view_id = rand() % share_data->num_of_views;
			while (chosen_nbvs_set.count(next_view_id)) {
				next_view_id = rand() % share_data->num_of_views;
			}
			break;

		case 1: //RandomOneshot
			if (oneshot_views.size() == 0) {
				//随机50次，选出分布最均匀的一个，即相互之间距离之和最大
				int check_num = 50;
				set<int> best_oneshot_views_set;
				double largest_pair_dis = -1e100;
				while (check_num--) {
					set<int> oneshot_views_set;
					oneshot_views_set.insert(first_view_id);
					for (int i = 0; i < share_data->num_of_max_iteration; i++) {
						int random_view_id = rand() % share_data->num_of_views;
						while (oneshot_views_set.count(random_view_id)) {
							random_view_id = rand() % share_data->num_of_views;
						}
						oneshot_views_set.insert(random_view_id);
					}
					double now_dis = 0;
					for (auto it = oneshot_views_set.begin(); it != oneshot_views_set.end(); it++) {
						auto it2 = it;
						it2++;
						while (it2 != oneshot_views_set.end()) {
							now_dis += (view_space->views[*it].init_pos - view_space->views[*it2].init_pos).norm();
							it2++;
						}
					}
					if (now_dis > largest_pair_dis) {
						largest_pair_dis = now_dis;
						best_oneshot_views_set = oneshot_views_set;
						cout << "largest_pair_dis: " << largest_pair_dis << endl;
					}
				}
				set<int> oneshot_views_set = best_oneshot_views_set;

				for (auto it = oneshot_views_set.begin(); it != oneshot_views_set.end(); it++) {
					oneshot_views.push_back(*it);
				}
				cout << "oneshot_views num is: " << oneshot_views.size() << endl;
				Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, first_view_id);
				double total_dis = gloabl_path_planner->solve();
				oneshot_views = gloabl_path_planner->get_path_id_set();
				if (oneshot_views.size() != share_data->num_of_max_iteration + 1) {
					cout << "oneshot_views.size() != share_data->num_of_max_iteration + 1" << endl;
				}
				cout << "total_dis: " << total_dis << endl;
				delete gloabl_path_planner;
				//删除初始视点
				oneshot_views.erase(oneshot_views.begin());
				//更新迭代次数，取出NBV
				share_data->num_of_max_iteration = oneshot_views.size();
				next_view_id = oneshot_views[0];
				oneshot_views.erase(oneshot_views.begin());
			}
			else {
				next_view_id = oneshot_views[0];
				oneshot_views.erase(oneshot_views.begin());
			}
			break;

		case 2: //EnsembleRGB
			//交给instantngp训练
			for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
				train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
			}
			//计算评价指标
			for (int i = 0; i < share_data->num_of_views; i++) {
				if (chosen_nbvs_set.count(i)) continue;
				vector<cv::Mat> rgb_images;
				for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
					cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
					rgb_images.push_back(rgb_image);
				}
				//使用ensemble计算uncertainty
				double view_uncertainty = 0.0;
				for (int j = 0; j < rgb_images[0].rows; j++) {
					for (int k = 0; k < rgb_images[0].cols; k++) {
						double mean_r = 0.0;
						double mean_g = 0.0;
						double mean_b = 0.0;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
							//注意不要归一化，不然会导致取log为负
							mean_r += rgba[0];
							mean_g += rgba[1];
							mean_b += rgba[2];
						}
						//计算方差
						mean_r /= share_data->ensemble_num;
						mean_g /= share_data->ensemble_num;
						mean_b /= share_data->ensemble_num;
						double variance_r = 0.0;
						double variance_g = 0.0;
						double variance_b = 0.0;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
							variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
							variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
							variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
						};
						variance_r /= share_data->ensemble_num;
						variance_g /= share_data->ensemble_num;
						variance_b /= share_data->ensemble_num;
						if (variance_r > 1e-10) view_uncertainty += log(variance_r);
						if (variance_g > 1e-10) view_uncertainty += log(variance_g);
						if (variance_b > 1e-10) view_uncertainty += log(variance_b);
					}
				}
				//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
				if (view_uncertainty > largest_view_uncertainty) {
					largest_view_uncertainty = view_uncertainty;
					best_view_id = i;
				}
				rgb_images.clear();
				rgb_images.shrink_to_fit();
			}
			//选择最好的视点
			next_view_id = best_view_id;
			break;

		case 3: //EnsembleRGBDensity	
			//交给instantngp训练
			for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
				train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
			}
			//计算评价指标
			for (int i = 0; i < share_data->num_of_views; i++) {
				if (chosen_nbvs_set.count(i)) continue;
				vector<cv::Mat> rgb_images;
				for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
					cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
					rgb_images.push_back(rgb_image);
				}
				//使用ensemble计算uncertainty，其中density存于alpha通道
				double view_uncertainty = 0.0;
				for (int j = 0; j < rgb_images[0].rows; j++) {
					for (int k = 0; k < rgb_images[0].cols; k++) {
						double mean_r = 0.0;
						double mean_g = 0.0;
						double mean_b = 0.0;
						double mean_density = 0.0;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
							//注意不要归一化，不然会导致取log为负
							mean_r += rgba[0];
							mean_g += rgba[1];
							mean_b += rgba[2];
							//注意alpha通道存储的是density，要归一化到0-1
							mean_density += rgba[3] / 255.0;
						}
						mean_r /= share_data->ensemble_num;
						mean_g /= share_data->ensemble_num;
						mean_b /= share_data->ensemble_num;
						mean_density /= share_data->ensemble_num;
						//cout << mean_r << " " << mean_g << " " << mean_b << " " << mean_density << endl;
						//计算方差
						double variance_r = 0.0;
						double variance_g = 0.0;
						double variance_b = 0.0;
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
							variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
							variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
							variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
						};
						variance_r /= share_data->ensemble_num;
						variance_g /= share_data->ensemble_num;
						variance_b /= share_data->ensemble_num;
						view_uncertainty += (variance_r + variance_g + variance_b) / 3.0;
						view_uncertainty += (1.0 - mean_density) * (1.0 - mean_density);
					}
				}
				//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
				if (view_uncertainty > largest_view_uncertainty) {
					largest_view_uncertainty = view_uncertainty;
					best_view_id = i;
				}
				rgb_images.clear();
				rgb_images.shrink_to_fit();
			}
			//选择最好的视点
			next_view_id = best_view_id;
			break;

		case 4: //PVBCoverage
			if (oneshot_views.size() == 0) {
				////通过网络获取视点预算
				share_data->access_directory(share_data->pvb_path + "data/images");
				for (int i = 0; i < init_view_ids.size(); i++) {
					ofstream fout_image(share_data->pvb_path + "data/images/" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
					ifstream fin_image(share_data->gt_path + "/" + to_string(init_views.size()) + "/rgbRembg_" + to_string(init_view_ids[i]) + ".png", std::ios::binary);
					fout_image << fin_image.rdbuf();
					fout_image.close();
					fin_image.close();
				}
				ofstream fout_ready(share_data->pvb_path + "data/ready_c++.txt");
				fout_ready.close();
				//等待python程序结束
				ifstream fin_over(share_data->pvb_path + "data/ready_py.txt");
				while (!fin_over.is_open()) {
					boost::this_thread::sleep(boost::posix_time::milliseconds(100));
					fin_over.open(share_data->pvb_path + "data/ready_py.txt");
				}
				fin_over.close();
				boost::this_thread::sleep(boost::posix_time::milliseconds(100));
				remove((share_data->pvb_path + "data/ready_py.txt").c_str());
				////读取view budget, bus25 gt20, airplane0 gt14
				int view_budget = -1;
				ifstream fin_view_budget(share_data->pvb_path + "data/view_budget.txt");
				if (!fin_view_budget.is_open()) {
					cout << "view budget file not found!" << endl;
				}
				fin_view_budget >> view_budget;
				fin_view_budget.close();
				cout << "view budget is: " << view_budget << endl;
				//读取coverage view space
				share_data->num_of_views = view_budget;
				ifstream fin_sphere(share_data->orginalviews_path + to_string(share_data->num_of_views) + ".txt");
				share_data->pt_sphere.clear();
				share_data->pt_sphere.resize(share_data->num_of_views);
				for (int i = 0; i < share_data->num_of_views; i++) {
					share_data->pt_sphere[i].resize(3);
					for (int j = 0; j < 3; j++) {
						fin_sphere >> share_data->pt_sphere[i][j];
					}
				}
				cout << "coverage view space size is: " << share_data->pt_sphere.size() << endl;
				Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
				share_data->pt_norm = pt0.norm();
				view_space.reset();
				view_space = make_shared<View_Space>(share_data);
				int now_first_view_id = -1;
				for (int i = 0; i < share_data->num_of_views; i++) {
					Eigen::Vector3d test_pos = Eigen::Vector3d(share_data->pt_sphere[i][0] / share_data->pt_norm, share_data->pt_sphere[i][1] / share_data->pt_norm, share_data->pt_sphere[i][2] / share_data->pt_norm);
					if (fabs(test_pos(0)) < 1e-6 && fabs(test_pos(1)) < 1e-6 && fabs(test_pos(2) - 1.0) < 1e-6) {
						now_first_view_id = i;
					}
					oneshot_views.push_back(i);
				}
				if (now_first_view_id == -1) {
					cout << "can not find now view id" << endl;
				}
				chosen_nbvs.clear();
				chosen_nbvs.push_back(now_first_view_id);
				chosen_nbvs_set.clear();
				chosen_nbvs_set.insert(now_first_view_id);
				//执行全局路径规划
				Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, now_first_view_id);
				double total_dis = gloabl_path_planner->solve();
				oneshot_views = gloabl_path_planner->get_path_id_set();
				cout << "total_dis: " << total_dis << endl;
				delete gloabl_path_planner;
				//保存所有视点个数
				ofstream fout_iteration(share_data->save_path + "/view_budget.txt");
				fout_iteration << oneshot_views.size() << endl;
				//删除初始视点
				oneshot_views.erase(oneshot_views.begin());
				//更新迭代次数，取出NBV
				share_data->num_of_max_iteration = oneshot_views.size();
				next_view_id = oneshot_views[0];
				oneshot_views.erase(oneshot_views.begin());
			}
			else {
				next_view_id = oneshot_views[0];
				oneshot_views.erase(oneshot_views.begin());
			}
			break;
		}
		chosen_nbvs.push_back(next_view_id);
		chosen_nbvs_set.insert(next_view_id);
		cout << "next_view_id: " << next_view_id << endl;

		infer_time = (clock() - infer_time) / CLOCKS_PER_SEC;
		ofstream fout_infer_time(share_data->save_path + "/infer_time/" + to_string(iteration) + ".txt");
		fout_infer_time << infer_time << endl;
		fout_infer_time.close();

		//运动代价：视点id，当前代价，总体代价
		int now_nbv_id = chosen_nbvs[iteration];
		int next_nbv_id = chosen_nbvs[iteration + 1];
		pair<int, double> local_path = get_local_path(view_space->views[now_nbv_id].init_pos.eval(), view_space->views[next_nbv_id].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
		total_movement_cost += local_path.second;
		cout << "local path: " << local_path.second << " total: " << total_movement_cost << endl;

		ofstream fout_move(share_data->save_path + "/movement/" + to_string(iteration) + ".txt");
		fout_move << next_nbv_id << '\t' << local_path.second << '\t' << total_movement_cost << endl;
		fout_move.close();

		//更新迭代次数
		iteration++;

		//返回是否需要迭代
		return true;
	}

	//返回当前视点id
	int get_nbv_id() {
		return chosen_nbvs[iteration];
	}

};

shared_ptr<Share_Data> share_data;
shared_ptr<NBV_Planner> nbv_planner;

int main_base() {

	//读取初始视点5个
	int num_of_max_initviews = 5;
	vector<View> init_views;
	shared_ptr<Share_Data> share_data_initviews = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", num_of_max_initviews);
	shared_ptr<View_Space> view_space_initviews = make_shared<View_Space>(share_data_initviews);
	init_views = view_space_initviews->views;
	//get top view id
	int first_init_view_id = -1;
	for (int i = 0; i < share_data_initviews->pt_sphere.size(); i++) {
		Eigen::Vector3d test_pos = Eigen::Vector3d(share_data_initviews->pt_sphere[i][0] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][1] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][2] / share_data_initviews->pt_norm);
		if (fabs(test_pos(0)) < 1e-6 && fabs(test_pos(1)) < 1e-6 && fabs(test_pos(2) - 1.0) < 1e-6) {
			first_init_view_id = i;
		}
	}
	if (first_init_view_id == -1) {
		cout << "can not find top view id" << endl;
	}
	vector<int> init_view_ids = share_data_initviews->init_view_ids;
	double init_dis = 0.0;
	if (init_view_ids.size() > 1) {
		Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data_initviews, init_views, init_view_ids, first_init_view_id);
		init_dis = gloabl_path_planner->solve();
		init_view_ids = gloabl_path_planner->get_path_id_set();
		//反转路径
		reverse(init_view_ids.begin(), init_view_ids.end());
		cout << "init_dis: " << init_dis << endl;
		delete gloabl_path_planner;
	}
	//输出起始路径
	ofstream fout_move_first(share_data_initviews->pre_path + "/init_movement_v" + to_string(init_view_ids.size()) + ".txt");
	fout_move_first << "init_dis: " << endl;
	fout_move_first << init_dis  << endl;
	fout_move_first << "path: " << endl;
	for (int i = 0; i < init_view_ids.size(); i++) {
		fout_move_first << init_view_ids[i] << '\t';
	}
	fout_move_first << endl;
	fout_move_first.close();
	//move and take images follow the path in ros
	for (int i = 0; i < init_view_ids.size(); i++) {
		//go to init_view_ids[i] in ros
		//take images in ros
		//...
	}
	//数据区清空
	share_data_initviews.reset();
	view_space_initviews.reset();

	//初始化数据区
	shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "", -1, -1);
	shared_ptr<NBV_Planner> nbv_planner = make_shared<NBV_Planner>(share_data, init_views, first_init_view_id);
	cout << "start view planning." << endl;
	while (nbv_planner->loop_once()){
		int nbv_id = nbv_planner->get_nbv_id();
		cout << "next_view_id: " << nbv_id << endl;
		//go to next_view_id in ros
		//take images in ros
		//...
	}
	//数据区清空
	nbv_planner.reset();
	share_data.reset();

	return 0;
}
