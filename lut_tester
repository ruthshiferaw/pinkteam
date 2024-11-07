#include <opencv2/opencv.hpp>
#include <iostream>

void testLUTApplication(const std::string& imagePath, const std::string& lutPath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return;
    }

    cv::Mat lut = cv::imread(lutPath, cv::IMREAD_UNCHANGED);
    if (lut.empty()) {
        std::cerr << "Error: Could not load LUT from " << lutPath << std::endl;
        return;
    }

    cv::Mat result;
    cv::LUT(image, lut, result);

    cv::imshow("Original Image", image);
    cv::imshow("LUT Enhanced Image", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./lut_tester <image_path> <lut_path>" << std::endl;
        return -1;
    }
    testLUTApplication(argv[1], argv[2]);
    return 0;
}
