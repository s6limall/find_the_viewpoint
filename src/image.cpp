//
// Created by ayush on 5/21/24.
//

#include "../include/image.hpp"
#include <spdlog/spdlog.h>
#include <random>
#include <Python.h>
#include <iostream>

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

double compute_match_ratio_SIFT(const cv::Mat &src_img, const cv::Mat &dst_img, float rt, bool use_HSV) {
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

double compute_match_ratio_LIGHTGLUE(){

    Py_Initialize();
    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_DecodeFSDefault("./LightGlue/"));

    // Load the script file (script.py)
    PyObject* pName = PyUnicode_DecodeFSDefault("apply_lightglue");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Load the function from the script
        PyObject* pFunc = PyObject_GetAttrString(pModule, "apply_lightglue");

        // Check if the function is callable
        if (PyCallable_Check(pFunc)) {
            // Call the function with no arguments
            PyObject* pValue = PyObject_CallObject(pFunc, nullptr);

            if (pValue != nullptr) {
                // Convert the result to a C++ float
                float result = static_cast<float>(PyFloat_AsDouble(pValue));
                Py_DECREF(pValue);
                return result;
            } else {
                PyErr_Print();
                std::cerr << "Failed to call the function." << std::endl;
            }
        } else {
            PyErr_Print();
            std::cerr << "Function not found or not callable." << std::endl;
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load the module." << std::endl;
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return 0;
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

double calculateTransformation(const cv::Mat &src_img, const cv::Mat &dst_img, float rt, bool use_HSV) {
    if (src_img.empty()) { spdlog::error("Source image is empty."); return std::numeric_limits<double>::infinity(); }
    if (dst_img.empty()) { spdlog::error("Destination image is empty."); return std::numeric_limits<double>::infinity(); }

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

    if (src_des_total == 0) { spdlog::error("No descriptors found in source image."); return std::numeric_limits<double>::infinity(); }

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < 3; ++i) {
        std::vector<std::vector<cv::DMatch>> knn_matches = matchSIFTDescriptors(src_des[i], dst_des[i]);
        std::vector<cv::DMatch> good_matches_tmp = applyRatioTest(knn_matches, rt);
        good_matches.insert(good_matches.end(), good_matches_tmp.begin(), good_matches_tmp.end());
    }

    if (good_matches.size() < 10) { spdlog::error("Not enough good matches found. Returning infinity."); return std::numeric_limits<double>::infinity(); }

    // Extract the matched keypoints
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto& match : good_matches) {
        src_pts.push_back(src_kp[match.queryIdx].pt);
        dst_pts.push_back(dst_kp[match.trainIdx].pt);
    }

    cv::Point2f translation(0, 0);
    for (const auto& match : good_matches) {
        translation.x += (dst_kp[match.trainIdx].pt.x - src_kp[match.queryIdx].pt.x);
        translation.y += (dst_kp[match.trainIdx].pt.y - src_kp[match.queryIdx].pt.y);
    }
    translation.x /= good_matches.size();
    translation.y /= good_matches.size();

    // Construct translation matrix
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    H.at<double>(0, 2) = translation.x;
    H.at<double>(1, 2) = translation.y;

    double magnitude = std::sqrt(translation.x * translation.x + translation.y * translation.y);
    return magnitude;
}