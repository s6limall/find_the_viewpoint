//
// Created by ayush on 5/21/24.
//

#include "../include/image.hpp"
#include <spdlog/spdlog.h>
#include <random>

// Convert image to grayscale
cv::Mat convertToGrayscale(const cv::Mat &img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert BGR image to grayscale
    return gray;
}

cv::Mat convertToHSV(const cv::Mat &img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

// Detect SIFT keypoints and compute descriptors
void detectAndComputeSIFT(const cv::Mat &channel, std::vector<cv::KeyPoint> &kp, cv::Mat &des) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // Create SIFT detector
    sift->detectAndCompute(channel, cv::noArray(), kp, des); // Detect keypoints and compute descriptors
}

// Match descriptors using FLANN-based matcher
std::vector<std::vector<cv::DMatch>> matchSIFTDescriptors(const cv::Mat &src_des, const cv::Mat &dst_des) {
    if (src_des.empty()) {
        spdlog::error("First descriptor matrix is empty.");
        return {};
    }
    if (dst_des.empty()) {
        spdlog::error("Second descriptor matrix is empty.");
        return {};
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(src_des, dst_des, knnMatches, 2); // Perform KNN matching with k=2
    return knnMatches;
}

// Apply the ratio test to filter good matches
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>> &knnMatches, float rt) {
    std::vector<cv::DMatch> goodMatches;
    for (const auto &knnMatch : knnMatches) {
        if (knnMatch[0].distance < rt * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]); // Keep only the good matches
        }
    }
    return goodMatches;
}

// Compute the number of good SIFT matches between two images
size_t computeSIFTMatches(const cv::Mat &src_img, const cv::Mat &dst_img, float rt) {
    if (src_img.empty()) { spdlog::error("First input image is empty."); return 0; }
    if (dst_img.empty()) { spdlog::error("Second input image is empty."); return 0; }

    std::vector<cv::KeyPoint> src_kp, dst_kp;
    cv::Mat src_des, dst_des;

    cv::Mat src_gray = convertToGrayscale(src_img);
    cv::Mat dst_gray = convertToGrayscale(dst_img);
    
    detectAndComputeSIFT(src_gray, src_kp, src_des);
    detectAndComputeSIFT(dst_gray, dst_kp, dst_des);

    auto knnMatches = matchSIFTDescriptors(src_des, dst_des);
    auto goodMatches = applyRatioTest(knnMatches, rt);
    return goodMatches.size();
}

double computeSIFTMatchRatio(const cv::Mat &src_img, const cv::Mat &dst_img, float rt, bool use_HSV) {
    if (src_img.empty()) {
        spdlog::error("Candiate image is empty.");
        return 0;
    }
    if (dst_img.empty()) {
        spdlog::error("Target image is empty.");
        return 0;
    }
        if (src_img.empty()) { spdlog::error("First input image is empty."); return 0; }
    if (dst_img.empty()) { spdlog::error("Second input image is empty."); return 0; }

    std::vector<cv::KeyPoint> src_kp, dst_kp;
    std::vector<cv::Mat> src_des(3), dst_des(3);

    size_t src_des_total = 0;
    if (use_HSV) {
        cv::Mat src_hsv = convertToHSV(src_img);
        cv::Mat dst_hsv = convertToHSV(dst_img);

        // Split HSV channels
        std::vector<cv::Mat> src_channels, dst_channels;
        cv::split(src_hsv, src_channels);
        cv::split(dst_hsv, dst_channels);

        // Compute SIFT descriptors for each channel
        for (int i = 0; i < 3; ++i) {
            detectAndComputeSIFT(src_channels[i], src_kp, src_des[i]);
            detectAndComputeSIFT(dst_channels[i], dst_kp, dst_des[i]);
            src_des_total += src_des[i].total();
        }
    } else {
        cv::Mat src_gray = convertToGrayscale(src_img);
        cv::Mat dst_gray = convertToGrayscale(dst_img);

        detectAndComputeSIFT(src_gray, src_kp, src_des[0]); // Use only the first element for grayscale
        detectAndComputeSIFT(dst_gray, dst_kp, dst_des[0]);
        src_des_total += src_des[0].total();
    }

    size_t matched_des_total = 0; // total matches in all channels
    std::vector<std::vector<cv::DMatch>> knn_Matches; // matches
    std::vector<cv::DMatch> good_matchs; // matches that fullfill rt
    double mr; // ratio of matched descriptors vs all found in src
    
    for (int i = 0; i < 3; ++i) {
        knn_Matches = matchSIFTDescriptors(src_des[i], dst_des[i]);
        good_matchs = applyRatioTest(knn_Matches, rt);
        matched_des_total += good_matchs.size();
    }
    mr = src_des_total == 0 ? 0.0 : static_cast<double>(matched_des_total) / src_des_total;
    
    spdlog::info("Ratio: {}, {}/{} with {}", mr, matched_des_total ,src_des_total, rt);
    return mr;
}

// Compare two images for similarity using SIFT feature matching
bool compareImages(const cv::Mat &src_img, const cv::Mat &dst_img) {
    if (src_img.empty()) {
        spdlog::error("First input image is empty.");
        return false;
    }
    if (dst_img.empty()) {
        spdlog::error("Second input image is empty.");
        return false;
    }
    if (src_img.size() != dst_img.size()) {
        spdlog::error("Input images do not match in size.");
        return false;
    }
    if (src_img.type() != dst_img.type()) {
        spdlog::error("Input images do not match in type.");
        return false;
    }

    size_t goodMatches = computeSIFTMatches(src_img, dst_img);
    constexpr size_t minGoodMatches = 10;

    bool result = goodMatches >= minGoodMatches;
    spdlog::info("Comparison result: {} ({} good matches, minimum required: {}).", result, goodMatches, minGoodMatches);
    return result;
}

std::string generateRandomID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    return std::to_string(dis(gen));
}

cv::Mat calculateTransformation(const cv::Mat &src_img, const cv::Mat &dst_img, float rt, bool use_HSV) {
    if (src_img.empty()) { spdlog::error("Source image is empty."); return cv::Mat::eye(3, 3, CV_64F); }
    if (dst_img.empty()) { spdlog::error("Destination image is empty."); return cv::Mat::eye(3, 3, CV_64F); }

    std::vector<cv::KeyPoint> src_kp, dst_kp;
    std::vector<cv::Mat> src_des(3), dst_des(3);

    size_t src_des_total = 0;
    if (use_HSV) {
        cv::Mat src_hsv = convertToHSV(src_img);
        cv::Mat dst_hsv = convertToHSV(dst_img);

        // Split HSV channels
        std::vector<cv::Mat> src_channels, dst_channels;
        cv::split(src_hsv, src_channels);
        cv::split(dst_hsv, dst_channels);

        // Compute SIFT descriptors for each channel
        for (int i = 0; i < 3; ++i) {
            detectAndComputeSIFT(src_channels[i], src_kp, src_des[i]);
            detectAndComputeSIFT(dst_channels[i], dst_kp, dst_des[i]);
            src_des_total += src_des[i].total();
        }
    } else {
        cv::Mat src_gray = convertToGrayscale(src_img);
        cv::Mat dst_gray = convertToGrayscale(dst_img);

        detectAndComputeSIFT(src_gray, src_kp, src_des[0]); // Use only the first element for grayscale
        detectAndComputeSIFT(dst_gray, dst_kp, dst_des[0]);
        src_des_total += src_des[0].total();
    }

    std::vector<std::vector<cv::DMatch>> knn_Matches; // matches
    std::vector<cv::DMatch> good_matchs; // matches that fullfill rt
    std::vector<cv::DMatch> good_matchs_tmp; // matches that fullfill rt
    double mr; // ratio of matched descriptors vs all found in src
    
    for (int i = 0; i < 3; ++i) {
        knn_Matches = matchSIFTDescriptors(src_des[i], dst_des[i]);
        good_matchs_tmp = applyRatioTest(knn_Matches, rt);
        good_matchs.insert(good_matchs.end(), good_matchs_tmp.begin(), good_matchs_tmp.end());
    }

    if (good_matchs.size() < 10) {
        spdlog::error("Not enough good matches found. Returning identity matrix.");
        return cv::Mat::eye(3, 3, CV_64F);
    }

    /* // Extract the matched keypoints
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto &match : good_matchs) {
        src_pts.push_back(src_kp[match.queryIdx].pt);
        dst_pts.push_back(dst_kp[match.trainIdx].pt);
    }

    // Calculate the homography matrix using RANSAC
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC);

    if (H.empty()) {
        spdlog::error("Homography calculation failed. Returning identity matrix.");
        return cv::Mat::eye(3, 3, CV_64F);
    } */

    // Extract the matched keypoints
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto &match : good_matchs) {
        src_pts.push_back(src_kp[match.queryIdx].pt);
        dst_pts.push_back(dst_kp[match.trainIdx].pt);
    }

    // Draw keypoints on the images
    cv::Mat src_img_drawn, dst_img_drawn;
    cv::drawKeypoints(src_img, src_kp, src_img_drawn, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::drawKeypoints(dst_img, dst_kp, dst_img_drawn, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Save images with keypoints drawn
    std::string rand_id = generateRandomID();
    std::string src_id = "./task2/selected_views/src_" + rand_id + "_with_kp.png";
    std::string dst_id = "./task2/selected_views/dst_" + rand_id + "_with_kp.png";

    cv::imwrite(src_id, src_img_drawn);
    cv::imwrite(dst_id, dst_img_drawn);


    cv::Point2f translation(0, 0);
    for (const auto &match : good_matchs) {
        translation.x += (dst_kp[match.trainIdx].pt.x - src_kp[match.queryIdx].pt.x);
        translation.y += (dst_kp[match.trainIdx].pt.y - src_kp[match.queryIdx].pt.y);
    }
    translation.x /= good_matchs.size();
    translation.y /= good_matchs.size();

    // Construct translation matrix
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    H.at<double>(0, 2) = translation.x;
    H.at<double>(1, 2) = translation.y;

    cv::Mat new_img;
    cv::warpPerspective(src_img, new_img, H, src_img.size());
    std::string new_id = "./task2/selected_views/new_" + rand_id + ".png";

    // Save images with random IDs
    cv::imwrite(src_id, src_img);
    cv::imwrite(dst_id, dst_img);
    cv::imwrite(new_id, new_img);


    return H;
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

// Function to shift image by dy pixels vertically and dx pixels horizontally
cv::Mat shiftImage(const cv::Mat& src_img, int dy, int dx) {
    cv::Mat new_img = cv::Mat::zeros(src_img.size(), src_img.type());

    cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, dx, 0, 1, dy);
    warpAffine(src_img, new_img, M, new_img.size());
    
    return new_img;
}

// Function to align images iteratively in HSV color space
cv::Mat alignImages(const cv::Mat& src_img, cv::Mat dst_img, int max_iterations, double convergence_threshold) {
    double bst_loss = std::numeric_limits<double>::infinity();
    cv::Point bst_shift(0, 0);
    double max_loss = calculateLoss(src_img, dst_img);
    
    int iterations = 0;
    while (iterations < max_iterations) {
        iterations++;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                cv::Mat shifted_dst_img = shiftImage(dst_img, dy, dx);
                double can_loss = calculateLoss(src_img, shifted_dst_img);
                
                if (can_loss < bst_loss) {
                    bst_loss = can_loss;
                    bst_shift = cv::Point(dx, dy);
                    dst_img = shifted_dst_img.clone(); // Update dst_img to the shifted version
                }
            }
        }
        
        // Check for convergence
        if (abs(max_loss - bst_loss) < convergence_threshold)
            break;
        
        max_loss = bst_loss;
    }
    
    cv::Mat transformation_matrix = cv::Mat::eye(3, 3, CV_64F);
    transformation_matrix.at<double>(0, 2) = bst_shift.x;
    transformation_matrix.at<double>(1, 2) = bst_shift.y;
    
    return transformation_matrix;
}
