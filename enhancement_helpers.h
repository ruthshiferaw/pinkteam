#ifndef ENHANCEMENT_HELPERS_H
#define ENHANCEMENT_HELPERS_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include <string>

cv::Mat downscaleImage(const cv::Mat& img, double scaleFactor = 0.5);
cv::Mat upscaleImage(const cv::Mat& img, const cv::Size& targetSize);
cv::Mat applyCLAHE(const cv::Mat& frame);
cv::Mat applyWhiteBalance(const cv::Mat& img);
cv::Mat applyFastFilters(const cv::Mat& frame);
cv::Mat darkChannelPrior(const cv::Mat& img, int patchSize = 15);
cv::Vec3f atmosphericLight(const cv::Mat& img, const cv::Mat& darkChannel);
cv::Mat estimateTransmission(const cv::Mat& img, const cv::Vec3f& A, double omega = 0.95);
cv::Mat guidedFilter(const cv::Mat& I, const cv::Mat& p, int radius = 60, double epsilon = 1e-3);
cv::Mat recoverScene(const cv::Mat& img, const cv::Vec3f& A, const cv::Mat& t, double t0 = 0.1);
cv::Mat dehazeImage(const cv::Mat& img, double scaleFactor = 0.5, int patchSize = 15);
std::pair<cv::Mat, std::map<std::string, double>> enhanceImage(const cv::Mat& img, bool whiteBalance = true, bool applyDehazing = true, bool applyCLAHE = true, bool applyFastFiltersFlag = true);

#endif // ENHANCEMENT_HELPERS_H
