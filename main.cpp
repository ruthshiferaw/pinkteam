#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "enhancement_helpers.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./video_enhancer <video_path>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("enhanced_output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps / 3, cv::Size(frameWidth, frameHeight));

    bool useLUT = true;
    bool useCLAHE = true;
    bool useTemporalSmoothing = true;
    bool useFastFilters = true;

    cv::Mat prevFrame, currentFrame, nextFrame, smoothedFrame;
    int frameCount = 0;
    auto startOverall = std::chrono::high_resolution_clock::now();

    while (cap.isOpened()) {
        cap >> currentFrame;
        if (currentFrame.empty()) break;
        frameCount++;

        // Frame skipping logic: process every 3rd frame
        if (frameCount % 3 != 0) continue;

        auto start = std::chrono::high_resolution_clock::now();
        if (useLUT) {
            applyLUT(currentFrame, "lut_file_path.cube");
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "LUT Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        if (useCLAHE) {
            applyCLAHE(currentFrame);
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "CLAHE Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        if (useFastFilters) {
            applyFastFilters(currentFrame);
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Fast Filters Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        if (useTemporalSmoothing && !prevFrame.empty()) {
            smoothedFrame = 0.5 * prevFrame + 0.5 * currentFrame;
        } else {
            smoothedFrame = currentFrame;
        }

        writer.write(smoothedFrame);
        prevFrame = currentFrame.clone();
    }

    cap.release();
    writer.release();
    auto endOverall = std::chrono::high_resolution_clock::now();
    std::cout << "Total Processing Time: " << std::chrono::duration_cast<std::chrono::seconds>(endOverall - startOverall).count() << " s" << std::endl;
    return 0;
}
