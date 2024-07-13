// // File: evaluation/ssim.cpp
//
// #include "evaluation/ssim.hpp"
// #include <opencv2/opencv.hpp>
//
// namespace evaluation {
//
//     SSIM::SSIM(double c1, double c2) :
//         c1(c1), c2(c2) {
//     }
//
//     double SSIM::compute(const cv::Mat &img1, const cv::Mat &img2) {
//         LOG_INFO("Starting SSIM computation");
//
//         if (img1.empty() || img2.empty()) {
//             LOG_ERROR("One or both images are empty");
//             return -1.0;
//         }
//         if (img1.size() != img2.size() || img1.type() != img2.type()) {
//             LOG_ERROR("Images must be of same size and type");
//             return -1.0;
//         }
//
//         cv::Mat img1Gray, img2Gray;
//         if (img1.channels() == 3) {
//             cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
//             cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);
//         } else {
//             img1Gray = img1;
//             img2Gray = img2;
//         }
//
//         double mean1 = calculateMean(img1Gray);
//         double mean2 = calculateMean(img2Gray);
//
//         double variance1 = calculateVariance(img1Gray, mean1);
//         double variance2 = calculateVariance(img2Gray, mean2);
//
//         double covariance = calculateCovariance(img1Gray, mean1, img2Gray, mean2);
//
//         double ssim = ((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) /
//                       ((mean1 * mean1 + mean2 * mean2 + c1) * (variance1 + variance2 + c2));
//
//         LOG_INFO("SSIM computation completed");
//
//         return ssim;
//     }
//
//     double SSIM::calculateMean(const cv::Mat &img) const {
//         cv::Scalar meanScalar = cv::mean(img);
//         return meanScalar[0];
//     }
//
//     double SSIM::calculateVariance(const cv::Mat &img, double mean) const {
//         cv::Mat temp;
//         cv::subtract(img, mean, temp);
//         cv::multiply(temp, temp, temp);
//         cv::Scalar varianceScalar = cv::mean(temp);
//         return varianceScalar[0];
//     }
//
//     double SSIM::calculateCovariance(const cv::Mat &img1, double mean1, const cv::Mat &img2, double mean2) const {
//         cv::Mat temp1, temp2, temp;
//         cv::subtract(img1, mean1, temp1);
//         cv::subtract(img2, mean2, temp2);
//         cv::multiply(temp1, temp2, temp);
//         cv::Scalar covarianceScalar = cv::mean(temp);
//         return covarianceScalar[0];
//     }
//
// }
