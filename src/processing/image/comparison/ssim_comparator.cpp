// File: processing/image/comparison/ssim_comparator.cpp

#include "processing/image/comparison/ssim_comparator.hpp"
#include "config/configuration.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace processing::image {

    double SSIMComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        if (image1.empty() || image2.empty()) {
            throw std::invalid_argument("One or both images are empty");
        }
        if (image1.size() != image2.size() || image1.type() != image2.type()) {
            throw std::invalid_argument("Images must have the same size and type");
        }

        const auto& config = config::Configuration::getInstance();
        int gaussian_window_size = config.get<int>("image_comparator.ssim.gaussian_window_size", 11);
        double gaussian_sigma = config.get<double>("image_comparator.ssim.gaussian_sigma", 1.5);
        double k1 = config.get<double>("image_comparator.ssim.k1", 0.01);
        double k2 = config.get<double>("image_comparator.ssim.k2", 0.03);
        double l = config.get<double>("image_comparator.ssim.l", 255);

        cv::Mat img1, img2;
        image1.convertTo(img1, CV_32F);
        image2.convertTo(img2, CV_32F);

        cv::Mat img1_sq = img1.mul(img1);
        cv::Mat img2_sq = img2.mul(img2);
        cv::Mat img1_img2 = img1.mul(img2);

        cv::Mat mu1, mu2;
        cv::GaussianBlur(img1, mu1, cv::Size(gaussian_window_size, gaussian_window_size), gaussian_sigma);
        cv::GaussianBlur(img2, mu2, cv::Size(gaussian_window_size, gaussian_window_size), gaussian_sigma);

        cv::Mat mu1_sq = mu1.mul(mu1);
        cv::Mat mu2_sq = mu2.mul(mu2);
        cv::Mat mu1_mu2 = mu1.mul(mu2);

        cv::Mat sigma1_sq, sigma2_sq, sigma12;
        cv::GaussianBlur(img1_sq, sigma1_sq, cv::Size(gaussian_window_size, gaussian_window_size), gaussian_sigma);
        sigma1_sq -= mu1_sq;
        cv::GaussianBlur(img2_sq, sigma2_sq, cv::Size(gaussian_window_size, gaussian_window_size), gaussian_sigma);
        sigma2_sq -= mu2_sq;
        cv::GaussianBlur(img1_img2, sigma12, cv::Size(gaussian_window_size, gaussian_window_size), gaussian_sigma);
        sigma12 -= mu1_mu2;

        double c1 = (k1 * l) * (k1 * l);
        double c2 = (k2 * l) * (k2 * l);

        cv::Mat ssim_map;
        cv::Mat t1 = 2 * mu1_mu2 + c1;
        cv::Mat t2 = 2 * sigma12 + c2;
        cv::Mat t3 = t1.mul(t2);

        t1 = mu1_sq + mu2_sq + c1;
        t2 = sigma1_sq + sigma2_sq + c2;
        t1 = t1.mul(t2);

        cv::divide(t3, t1, ssim_map);
        cv::Scalar mssim = cv::mean(ssim_map);

        return mssim[0];
    }
}
