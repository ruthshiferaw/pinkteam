#ifndef ENHANCEMENT_HELPERS_H
#define ENHANCEMENT_HELPERS_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include <string>

// cv::Mat downscaleImage(const cv::Mat &img, double scaleFactor = 0.5);
// cv::Mat upscaleImage(const cv::Mat &img, const cv::Size &targetSize);
cv::Mat darkChannelPrior(const cv::Mat &img, int patchSize = 15);
cv::Vec3b atmosphericLight(const cv::Mat &img, const cv::Mat &darkChannel);
cv::Mat estimateTransmission(const cv::Mat &src, const cv::Vec3b &A, int size = 15);
cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int radius = 60, double epsilon = 1e-3);
cv::Mat recoverScene(const cv::Mat &src, const cv::Vec3b &A, const cv::Mat &transmission, float t0 = 0.1);
cv::Mat applyWhiteBalance(const cv::Mat &img);
cv::Mat dehazeImage(const cv::Mat &img, int patchSize, int size, int radius, double epsilon, float t0);

#endif // ENHANCEMENT_HELPERS_H
