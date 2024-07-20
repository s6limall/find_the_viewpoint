// File: main.cpp

#include <cmath>
#include <iostream>
#include <type_traits>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "common/io/image.hpp"
#include "executor.hpp"
#include "filtering/image/bilateral_filter.hpp"
#include "filtering/image/rpca.hpp"
#include "optimization/cmaes.hpp"
#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "processing/image/preprocessor.hpp"
#include "processing/vision/estimation/pose_estimator.hpp"
#include "types/viewpoint.hpp"
#include "viewpoint/evaluator.hpp"


/*int main() {
    // Load target image and create evaluator
    const auto image = common::io::image::readImage("../../task1/target_images/obj_000020/img.png");

    // Set up camera intrinsics
    core::Camera::Intrinsics intrinsics;
    intrinsics.setIntrinsics(640, 480, 0.95, 0.75);

    // Create an instance of the PoseEstimator
    const processing::vision::PoseEstimator poseEstimator(intrinsics);
    try {
        const Eigen::Matrix4d pose = poseEstimator.estimate(image);
        std::cout << "Estimated Pose: \n" << pose << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }


    return 0;
    EXIT_SUCCESS;
}*/

int main() {

    try {
        Executor::execute();
    } catch (const std::exception &e) {
        LOG_ERROR("An error occurred during execution.", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
