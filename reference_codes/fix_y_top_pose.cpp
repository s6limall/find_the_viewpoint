/////// fixed y_top pose
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