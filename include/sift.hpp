#ifndef SIFT_HPP
#define SIFT_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class SIFT {
public:
    SIFT();
    double compute(const cv::Mat &src_img, const cv::Mat &dst_img, float rt);
    std::pair<double, double> homography(const cv::Mat& img1, const cv::Mat& img2);
};
#endif // SIFT_HPP