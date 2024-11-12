#include "enhancement_helpers.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file " << videoPath << std::endl;
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Define the codec and create VideoWriter object to save output
    cv::VideoWriter writer("enhanced_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open the output video for write." << std::endl;
        return -1;
    }

    cv::Mat frame, enhancedFrame;
    int frameCount = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Measure the time taken to enhance the frame
        auto start = std::chrono::high_resolution_clock::now();
        
        // Apply enhancement (e.g., CLAHE or other enhancement functions)
        enhancedFrame = applyCLAHE(frame);  // Example: using CLAHE, modify as needed
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> processingTime = end - start;
        
        // Print processing time for the frame
        std::cout << "Frame " << frameCount++ << " enhancement time: " << processingTime.count() << " ms" << std::endl;

        // Write the enhanced frame to the output video
        writer.write(enhancedFrame);

        // Optional: display the enhanced frame
        cv::imshow("Enhanced Video", enhancedFrame);
        if (cv::waitKey(1) >= 0) break;  // Press any key to exit early
    }

    // Release resources
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Enhanced video saved as enhanced_video.mp4" << std::endl;
    return 0;
}
