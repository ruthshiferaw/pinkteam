#include "enhancement_helpers.hpp"

void applyLUT(cv::Mat& frame, const std::string& lutPath) {
    // Simple placeholder for LUT loading; .CUBE file parsing must be implemented.
    // You can replace this with actual 3D LUT parsing code.
    std::cerr << "LUT application needs a proper .CUBE parsing implementation." << std::endl;
}

void applyCLAHE(cv::Mat& frame) {
    cv::Mat labFrame;
    cv::cvtColor(frame, labFrame, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labPlanes(3);
    cv::split(labFrame, labPlanes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(labPlanes[0], labPlanes[0]);
    cv::merge(labPlanes, labFrame);
    cv::cvtColor(labFrame, frame, cv::COLOR_Lab2BGR);
}

void applyFastFilters(cv::Mat& frame) {
    cv::bilateralFilter(frame, frame, 9, 75, 75);
}
