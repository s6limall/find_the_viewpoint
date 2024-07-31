//////////////////////////////////////// Get path length with obstacle
#define ErrorPath -2
#define WrongPath -1
#define LinePath 0
#define CirclePath 1
//return path mode and length from M to N under an sphere obstacle with radius r
pair<int, double> get_local_path(Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double r) {
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	//Calculate the point of intersection PQ of the line MN with the sphere O-r
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//If there are no intersections or one intersection, you can draw a straight line through it
		double d = (N - M).norm();
		//cout << "Delta<= 0, LinePath d: " << d << endl;
		return make_pair(LinePath, d);
	}
	else {
		//If it is necessary to cross the sphere, act along the surface of the sphere
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//Two points outside the sphere. Straight through.
			double d = (N - M).norm();
			//cout << "Two points outside the sphere. LinePath d: " << d << endl;
			return make_pair(LinePath, d);
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			//Starting or finishing point inside the obstacle course ball
			cout << "One point inside the sphere. WrongPath" << endl;
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
		//MON Plane Equation
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//Calculate the parameter values of P,Q in the parametric equation
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		if (sin_theta3 > 1.0) sin_theta3 = 1.0; // solve the problem of precision
		if (sin_theta3 < -1.0) sin_theta3 = -1.0; // solve the problem of precision
		theta3 = asin(sin_theta3);
		//cout << "sin_theta3: " << sin_theta3 << " theta3: " << theta3 << endl;
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
		if (sin_theta4 > 1.0) sin_theta4 = 1.0; // solve the problem of precision
		if (sin_theta4 < -1.0) sin_theta4 = -1.0; // solve the problem of precision
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
		//cout << "M: " << M.transpose() << endl;
		//cout << "N: " << N.transpose() << endl;
		//cout << "MP: " << MP << endl;
		//cout << "QN: " << QN << endl;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "CirclePath. d: " << d << endl;
		return make_pair(CirclePath, d);
	}
	//Undefined behaviour
	cout << "Undefined case. ErrorPath" << endl;
	return make_pair(ErrorPath, 1e10);
}

////////// Usage of path length
// pair<int, double> local_path = get_local_path(views[u].init_pos.eval(), views[v].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
// if (local_path.first < 0) { cout << "local path not found." << endl;}
// cout << local_path.second <<endl;


//////////////////////////////////////// Get waypoints on the path with obstacle
double get_trajectory_xyz(vector<Eigen::Vector3d>& points, Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double predicted_size, double distanse_of_pre_move, double camera_to_object_dis) {
	int num_of_path = -1;
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4, r;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	r = predicted_size + camera_to_object_dis;
	//Calculate the point of intersection PQ of the line MN with the sphere O-r
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//If there are no intersections or one intersection, you can draw a straight line through it
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
		//If it is necessary to cross the sphere, act along the surface of the sphere
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//Two points outside the sphere. Straight through.
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
		//MON Plane Equation
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//Calculate the parameter values of P,Q in the parametric equation
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		if (sin_theta3 > 1.0) sin_theta3 = 1.0; // solve the problem of precision
		if (sin_theta3 < -1.0) sin_theta3 = -1.0; // solve the problem of precision
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
		if (sin_theta4 > 1.0) sin_theta4 = 1.0; // solve the problem of precision
		if (sin_theta4 < -1.0) sin_theta4 = -1.0; // solve the problem of precision
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
					//If the path point height is close to negative, it means that the solved path is the lower half of the sphere, and the flip direction
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

////////// Usage of waypoints (example of visualization)
void show_view_image_path() {
	bool highlight_initview = true;
	bool is_show_path = true;
	bool is_global_path = true;
	bool is_show_image = true;
	bool is_show_model = true;
	/////////////////////////////////////////////////////////////////
	// Important to modify viewspace path here.                    //
	/////////////////////////////////////////////////////////////////
	//Reading View Space
	ifstream fin("./view_space/100.txt");
	if (fin.is_open()) {
		int num = 100;
		object_center_world = Eigen::Vector3d(1e-10, 1e-10, 1e-10);
		now_camera_pose_world = Eigen::Matrix4d::Identity();
		predicted_size = 1.0;
		for (int i = 0; i < num; i++) {
			double init_pos[3];
			fin >> init_pos[0] >> init_pos[1] >> init_pos[2];
			View view(Eigen::Vector3d(init_pos[0] * 3.0, init_pos[1] * 3.0, init_pos[2] * 3.0));
			views.push_back(view);
		}
		cout << "viewspace readed." << endl;
	}
	else {
		cout << "no view space. check!" << endl;
	}
	//Read selected viewpoints with path sequence
	vector<int> chosen_views = { 126, 91, 32, 9, 129, 104, 56, 52, 29, 34, 69, 97, 140, 26 }; // just an example
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
	rs2_intrinsics color_intrinsics;
	color_intrinsics.width = 640;
	color_intrinsics.height = 480;
	color_intrinsics.fx = width / (2 * tan(fov_x / 2));
	color_intrinsics.fy = height / (2 * tan(fov_y / 2));
	color_intrinsics.ppx = width / 2;
	color_intrinsics.ppy = height / 2;
	color_intrinsics.model = RS2_DISTORTION_NONE;
	color_intrinsics.coeffs[0] = 0;
	color_intrinsics.coeffs[1] = 0;
	color_intrinsics.coeffs[2] = 0;
	color_intrinsics.coeffs[3] = 0;
	color_intrinsics.coeffs[4] = 0;

	double view_color[3] = { 0, 0, 255 };
	double path_color[3] = { 128, 0, 128 };

	for (int i = 0; i < chosen_views.size(); i++) {
		views[chosen_views[i]].get_next_camera_pos(now_camera_pose_world, object_center_world);
		Eigen::Matrix4d view_pose_world = (now_camera_pose_world * views[chosen_views[i]].pose.inverse()).eval();

		double line_length = 0.3;

		Eigen::Vector3d LeftTop = project_pixel_to_ray_end(0, 0, color_intrinsics, view_pose_world, line_length);
		Eigen::Vector3d RightTop = project_pixel_to_ray_end(0, 720, color_intrinsics, view_pose_world, line_length);
		Eigen::Vector3d LeftBottom = project_pixel_to_ray_end(1280, 0, color_intrinsics, view_pose_world, line_length);
		Eigen::Vector3d RightBottom = project_pixel_to_ray_end(1280, 720, color_intrinsics, view_pose_world, line_length);

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
				Eigen::Vector3d now_view_xyz = views[chosen_views[i - 1]].init_pos;
				Eigen::Vector3d next_view_xyz = views[chosen_views[i]].init_pos;
				vector<Eigen::Vector3d> points;
				int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, object_center_world, predicted_size, 0.2, 0.0);
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
			cv::Mat image = cv::imread("./obj_000020/rgb_" + to_string(chosen_views[i]) + ".png");
			cv::flip(image, image, -1);
			double image_line_length = 0.3;
			int interval = 4;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_image(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (int x = 0; x < image.cols; x += interval) {
				for (int y = 0; y < image.rows; y += interval) {
					Eigen::Vector3d pixel_end = project_pixel_to_ray_end(x, y, color_intrinsics, view_pose_world, image_line_length);
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
		object_path = "./3d_models/obj_000020.ply";
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
		viewer->addPolygonMesh(*mesh_ply, "object");
		viewer->spinOnce(100);
	}
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


