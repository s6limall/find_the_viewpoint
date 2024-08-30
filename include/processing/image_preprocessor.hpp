// File: processing/image_preprocessor.hpp

#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdexcept>

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(const std::string &model_path = "./u2net.onnx") {
        // Load the pre-trained ONNX model
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            throw std::runtime_error("Failed to load ONNX model from: " + model_path);
        }
    }

     cv::Mat process(const cv::Mat& input_image) {
        cv::Mat processed_image = input_image.clone();

        // Check if background removal is necessary
        if (hasSignificantBackground(input_image)) {
            processed_image = removeBackground(input_image);
        } else {
            processed_image = input_image.clone();
        }
        cv::Mat enhanced_image = enhanceFeatures(processed_image);

        return normalizeImage(enhanced_image);
    }

private:
    cv::dnn::Net net;

    static bool hasSignificantBackground(const cv::Mat& image) {
        // Convert image to grayscale
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // Calculate the standard deviation of pixel values
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray_image, mean, stddev);

        // If the standard deviation is low, it's likely a uniform background
        constexpr double background_threshold = 20.0; // You can adjust this threshold
        return stddev[0] > background_threshold;
    }

     cv::Mat removeBackground(const cv::Mat &image) {
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        cv::Mat output = net.forward();

        // Convert the output to a probability map
        cv::Mat prob_map(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        // Resize the probability map to match the original image size
        cv::resize(prob_map, prob_map, image.size());

        // Threshold the probability map to create a binary mask
        cv::Mat mask;
        cv::threshold(prob_map, mask, 0.5, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);

        // Use the mask to remove the background
        cv::Mat result;
        image.copyTo(result, mask);

        // Optionally, fill the background with white
        cv::Mat white_bg(image.size(), image.type(), cv::Scalar(255, 255, 255));
        cv::bitwise_not(mask, mask);
        white_bg.copyTo(result, mask);

        return result;
    }

    static cv::Mat enhanceFeatures(const cv::Mat &image) {
        cv::Mat lab_image, enhanced;

        // Convert to LAB color space for better contrast manipulation
        cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

        // Split the LAB image into L, A, B channels
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);

        // Apply CLAHE to the L-channel for contrast enhancement
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(lab_planes[0], lab_planes[0]);

        // Merge the LAB channels back
        cv::merge(lab_planes, lab_image);

        // Convert back to BGR color space
        cv::cvtColor(lab_image, enhanced, cv::COLOR_Lab2BGR);

        return enhanced;
    }

    static cv::Mat normalizeImage(const cv::Mat& image) {
        cv::Mat normalized;
        cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        return normalized;
    }

};

#endif // IMAGE_PROCESSOR_HPP
