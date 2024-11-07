#ifndef ENHANCEMENT_HELPERS_HPP
#define ENHANCEMENT_HELPERS_HPP

#include <opencv2/opencv.hpp>
#include <string>

void applyLUT(cv::Mat& frame, const std::string& lutPath);
void applyCLAHE(cv::Mat& frame);
void applyFastFilters(cv::Mat& frame);

#endif // ENHANCEMENT_HELPERS_HPP
