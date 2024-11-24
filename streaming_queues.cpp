<<<<<<< Updated upstream
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iostream>
#include <utility>
=======
// #include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <thread>
// #include <queue>
// #include <mutex>
// #include <atomic>
// #include <chrono>
// #include <iostream>
// #include <utility>
// #include "enhancement_helpers.h"

// void captureCamera(int cameraIndex, std::queue<std::pair<cv::Mat, double>> &frameQueue, std::mutex &queueMutex, std::atomic<bool> &stopFlag);
// void processFrames(std::queue<std::pair<cv::Mat, double>> &frameQueue1, std::queue<std::pair<cv::Mat, double>> &frameQueue2,
//                    std::queue<cv::Mat> &processedQueue, std::mutex &queueMutex1, std::mutex &queueMutex2,
//                    std::mutex &processedQueueMutex, std::atomic<bool> &stopFlag);
// void displayFrames(std::queue<cv::Mat> &processedQueue, std::mutex &processedQueueMutex, std::atomic<bool> &stopFlag);

// Main function for two cameras
//  int main()
//  {
//      std::queue<std::pair<cv::Mat, double>> frameQueue1, frameQueue2;
//      std::queue<cv::Mat> processedQueue;
//      std::mutex queueMutex1, queueMutex2, processedQueueMutex;
//      std::atomic<bool> stopFlag(false);

//     // Threads for capturing, processing, and displaying frames
//     std::thread captureThread1(captureCamera, 1, std::ref(frameQueue1), std::ref(queueMutex1), std::ref(stopFlag));
//     std::thread captureThread2(captureCamera, 2, std::ref(frameQueue2), std::ref(queueMutex2), std::ref(stopFlag));
//     std::thread processingThread(processFrames, std::ref(frameQueue1), std::ref(frameQueue2), std::ref(processedQueue),
//                                  std::ref(queueMutex1), std::ref(queueMutex2), std::ref(processedQueueMutex), std::ref(stopFlag));
//     std::thread displayThread(displayFrames, std::ref(processedQueue), std::ref(processedQueueMutex), std::ref(stopFlag));

//     // Wait for user to stop the program
//     std::cin.get();
//     stopFlag = true;

//     // Join threads
//     captureThread1.join();
//     captureThread2.join();
//     processingThread.join();
//     displayThread.join();

//     return 0;
// }
>>>>>>> Stashed changes

// // Capture frames and put them in the queue
// std::atomic<double> globalStartTime(-1.0); // Shared across threads
// void captureCamera(int cameraIndex, std::queue<std::pair<cv::Mat, double>> &frameQueue, std::mutex &queueMutex, std::atomic<bool> &stopFlag)
// {
//     cv::VideoCapture cap(cameraIndex, cv::CAP_DSHOW); // Use DirectShow on Windows
//     cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
//     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

//     if (!cap.isOpened())
//     {
//         std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
//         return;
//     }

//     // discard the first i frames
//     for (int i = 0; i < 30; i++) // Adjust number of frames as needed
//     {
//         cv::Mat temp;
//         cap.read(temp); // Discard the frame
//     }

//     int x = 420, y = 0, roiWidth = 1080, roiHeight = 1080;

//     while (!stopFlag)
//     {
//         cv::Mat frame;
//         if (cap.read(frame))
//         {
//             double timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

<<<<<<< Updated upstream
            if (std::abs(frameData1.second - frameData2.second) <= 0.03) {
                cv::Mat sq1 = frameData1.first(cv::Rect(420, 0, 1080, 1080));
                cv::Mat sq2 = frameData2.first(cv::Rect(420, 0, 1080, 1080));

                cv::resize(sq1, sq1, cv::Size(1216, 1216));
                cv::resize(sq2, sq2, cv::Size(1216, 1216));
=======
//             if (globalStartTime < 0.0)
//             {
//                 globalStartTime = timestamp;
//             }
>>>>>>> Stashed changes

//             // Normalize timestamp relative to global start time
//             timestamp -= globalStartTime;

//             // Validate and adjust ROI
//             if (frame.cols < x + roiWidth || frame.rows < y + roiHeight)
//             {
//                 roiWidth = std::min(roiWidth, frame.cols - x);
//                 roiHeight = std::min(roiHeight, frame.rows - y);

<<<<<<< Updated upstream
                std::lock_guard<std::mutex> lockProcessed(processedQueueMutex);
                if (processedQueue.size() >= 10) {
                    processedQueue.pop();
                }
                processedQueue.push(concatenated);
                frameQueue1.pop();
                frameQueue2.pop();
            } else {
                if (frameData1.second < frameData2.second) {
                    frameQueue1.pop();
                } else {
                    frameQueue2.pop();
                }
            }
        }
    }
}

// Display frames from the processed queue and adjust image sizes dynamically
void displayFrames(std::queue<cv::Mat>& processedQueue, std::mutex& processedQueueMutex, std::atomic<bool>& stopFlag) {
    bool isFirstFrame = true;
    cv::Size windowSize(1280, 720); // Default window size
=======
//                 if (roiWidth <= 0 || roiHeight <= 0)
//                 {
//                     std::cerr << "Error: Frame dimensions too small for a valid ROI. Skipping frame." << std::endl;
//                     continue; // Skip the frame
//                 }
//             }

//             // Crop the frame using the valid ROI
//             cv::Mat croppedFrame = frame(cv::Rect(x, y, roiWidth, roiHeight));

//             std::lock_guard<std::mutex> lock(queueMutex);
//             frameQueue.push(std::make_pair(croppedFrame, timestamp));
>>>>>>> Stashed changes

//             std::this_thread::sleep_for(std::chrono::milliseconds(33)); // Simulate 30 FPS
//         }
//         else
//         {
//             std::cerr << "Error: Failed to capture frame from camera " << cameraIndex << std::endl;
//             break;
//         }
//     }
//     cap.release();
// }

<<<<<<< Updated upstream
        if (!concatenated.empty()) {
            if (isFirstFrame) {
                // Create the window only once
                cv::namedWindow("Two Cameras Side by Side", cv::WINDOW_NORMAL);
                cv::resizeWindow("Two Cameras Side by Side", windowSize.width, windowSize.height);
                isFirstFrame = false;
            } else {
                // Dynamically adjust the size based on the existing window size
                cv::Rect windowRect = cv::getWindowImageRect("Two Cameras Side by Side");
                windowSize.width = windowRect.width;
                windowSize.height = windowRect.height;
            }

            // Resize the frame to fit the window while maintaining aspect ratio
            int frameWidth = concatenated.cols;
            int frameHeight = concatenated.rows;

            double scalingFactor = std::min(windowSize.width / (double)frameWidth, windowSize.height / (double)frameHeight);
            cv::Size newSize(static_cast<int>(frameWidth * scalingFactor), static_cast<int>(frameHeight * scalingFactor));
            cv::Mat resizedFrame;
            cv::resize(concatenated, resizedFrame, newSize);

            cv::imshow("Two Cameras Side by Side", resizedFrame);
            if (cv::waitKey(1) == 'q') {
                stopFlag = true;
                break;
            }
        }
    }
    cv::destroyAllWindows();
}

int main() {
    try {
        std::queue<std::pair<cv::Mat, double>> frameQueue1, frameQueue2;
        std::queue<cv::Mat> processedQueue;
        std::mutex queueMutex1, queueMutex2, processedQueueMutex;
        std::atomic<bool> stopFlag(false);

        // Threads for capturing, processing, and displaying frames
        std::thread captureThread1(captureCamera, 0, std::ref(frameQueue1), std::ref(queueMutex1), std::ref(stopFlag));
        std::thread captureThread2(captureCamera, 1, std::ref(frameQueue2), std::ref(queueMutex2), std::ref(stopFlag));
        std::thread processingThread(processFrames, std::ref(frameQueue1), std::ref(frameQueue2), std::ref(processedQueue), 
                                     std::ref(queueMutex1), std::ref(queueMutex2), std::ref(processedQueueMutex), std::ref(stopFlag));
        std::thread displayThread(displayFrames, std::ref(processedQueue), std::ref(processedQueueMutex), std::ref(stopFlag));

        // Inform the user to press Enter to stop the program
        std::cout << "Press Enter to stop the program..." << std::endl;
        std::cin.get(); // Wait for the user to press Enter
        stopFlag = true;

        // Join threads
        captureThread1.join();
        captureThread2.join();
        processingThread.join();
        displayThread.join();
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }

    return 0;
}
=======
// void processFrames(std::queue<std::pair<cv::Mat, double>> &frameQueue1, std::queue<std::pair<cv::Mat, double>> &frameQueue2,
//                    std::queue<cv::Mat> &processedQueue, std::mutex &queueMutex1, std::mutex &queueMutex2,
//                    std::mutex &processedQueueMutex, std::atomic<bool> &stopFlag)
// {
//     int frameCount = 0;                                      // Counter to track frames
//     std::deque<std::pair<cv::Mat, double>> buffer1, buffer2; // Buffers for both cameras

//     while (!stopFlag)
//     {
//         // Check if frames are available in the queues
//         {
//             std::lock_guard<std::mutex> lock1(queueMutex1);
//             std::lock_guard<std::mutex> lock2(queueMutex2);

//             if (!frameQueue1.empty())
//             {
//                 buffer1.push_back(frameQueue1.front());
//                 frameQueue1.pop();
//             }
//             if (!frameQueue2.empty())
//             {
//                 buffer2.push_back(frameQueue2.front());
//                 frameQueue2.pop();
//             }
//         }

//         // Process buffers to find synchronized frames
//         while (!buffer1.empty() && !buffer2.empty())
//         {
//             auto frameData1 = buffer1.front();
//             auto frameData2 = buffer2.front();

//             // Synchronize frames based on timestamp
//             if (std::abs(frameData1.second - frameData2.second) <= 0.1) // Synchronization tolerance
//             {
//                 // Frames are synchronized
//                 buffer1.pop_front();
//                 buffer2.pop_front();

//                 // Define ROI parameters
//                 int x = 420, y = 0, roiWidth = 1080, roiHeight = 1080;

//                 // Validate and adjust ROI for Frame 1
//                 int adjustedWidth1 = std::min(roiWidth, frameData1.first.cols - x);
//                 int adjustedHeight1 = std::min(roiHeight, frameData1.first.rows - y);

//                 if (adjustedWidth1 <= 0 || adjustedHeight1 <= 0)
//                 {
//                     std::cerr << "Error: Invalid ROI for Frame 1. Skipping." << std::endl;
//                     continue;
//                 }

//                 // Extract ROI for Frame 1
//                 cv::Mat sq1 = frameData1.first(cv::Rect(x, y, adjustedWidth1, adjustedHeight1));

//                 // Validate and adjust ROI for Frame 2
//                 int adjustedWidth2 = std::min(roiWidth, frameData2.first.cols - x);
//                 int adjustedHeight2 = std::min(roiHeight, frameData2.first.rows - y);

//                 if (adjustedWidth2 <= 0 || adjustedHeight2 <= 0)
//                 {
//                     std::cerr << "Error: Invalid ROI for Frame 2. Skipping." << std::endl;
//                     continue;
//                 }

//                 // Extract ROI for Frame 2
//                 cv::Mat sq2 = frameData2.first(cv::Rect(x, y, adjustedWidth2, adjustedHeight2));

//                 // Increment frame counter
//                 frameCount++;

//                 // Resize frames every 5 frames
//                 // if (frameCount % 5 == 0)

//                 // Add borders to both frames
//                 cv::Mat leftSide, rightSide;
//                 cv::resize(sq1, sq1, cv::Size(1216, 1216));
//                 cv::copyMakeBorder(sq1, leftSide, 112, 112, 0, 64, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

//                 cv::resize(sq2, sq2, cv::Size(1216, 1216));
//                 cv::copyMakeBorder(sq2, rightSide, 112, 112, 64, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

//                 // Ensure matching types by converting grayscale to color
//                 std::cout << "LeftSide type = " << leftSide.type() << std::endl;
//                 std::cout << "RightSide type = " << rightSide.type() << std::endl;

//                 if (leftSide.type() != rightSide.type())
//                 {
//                     if (leftSide.type() == CV_8UC1)
//                         cv::cvtColor(leftSide, leftSide, cv::COLOR_GRAY2BGR);
//                     if (rightSide.type() == CV_8UC1)
//                         cv::cvtColor(rightSide, rightSide, cv::COLOR_GRAY2BGR);
//                 }

//                 std::cout << "Before dehaze: LeftSide type = " << leftSide.type() << std::endl;

//                 // Convert to CV_8UC3 if necessary
//                 if (leftSide.type() != CV_8UC3)
//                 {
//                     leftSide.convertTo(leftSide, CV_8UC3);
//                     std::cerr << "Error: Input image type is not CV_8UC3. Converting..." << std::endl;
//                 }
//                 if (rightSide.type() != CV_8UC3)
//                     rightSide.convertTo(rightSide, CV_8UC3);

//                 // Apply enhancements from enhancement_helpers.h
//                 leftSide = dehazeImage(leftSide);   // Example: Replace with your actual enhancement function
//                 rightSide = dehazeImage(rightSide); // Process the second camera frame
//                 std::cout << "After dehaze: LeftSide type = " << leftSide.type() << std::endl;

//                 // // Convert to CV_8UC3 if necessary
//                 // if (leftSide.type() != CV_8UC3)
//                 //     leftSide.convertTo(leftSide, CV_8UC3);

//                 // if (rightSide.type() != CV_8UC3)
//                 //     rightSide.convertTo(rightSide, CV_8UC3);

//                 // Concatenate frames horizontally
//                 cv::Mat concatenated;
//                 cv::hconcat(leftSide, rightSide, concatenated);

//                 // Push the processed frame into the processedQueue
//                 {
//                     std::lock_guard<std::mutex> lockProcessed(processedQueueMutex);
//                     if (processedQueue.size() >= 10)
//                     {
//                         processedQueue.pop();
//                     }
//                     processedQueue.push(concatenated);
//                 }
//             }
//             else
//             {
//                 // Remove the older frame from the buffer
//                 if (frameData1.second < frameData2.second)
//                 {
//                     buffer1.pop_front();
//                 }
//                 else
//                 {
//                     buffer2.pop_front();
//                 }
//             }
//         }
//     }
// }

// void displayFrames(std::queue<cv::Mat> &processedQueue, std::mutex &processedQueueMutex, std::atomic<bool> &stopFlag)
// {
//     int displayWidth = 1280, displayHeight = 720;

//     while (!stopFlag)
//     {
//         cv::Mat concatenated;
//         {
//             std::lock_guard<std::mutex> lock(processedQueueMutex);
//             if (!processedQueue.empty())
//             {
//                 concatenated = processedQueue.front();
//                 processedQueue.pop();
//             }
//         }

//         if (!concatenated.empty())
//         {
//             int frameWidth = concatenated.cols;
//             int frameHeight = concatenated.rows;

//             if (frameWidth == 0 || frameHeight == 0)
//             {
//                 std::cerr << "Error: Invalid concatenated frame dimensions." << std::endl;
//                 continue;
//             }

//             double scalingFactor = std::min(displayWidth / (double)frameWidth, displayHeight / (double)frameHeight);
//             std::cout << "Scaling factor: " << scalingFactor << std::endl;

//             cv::Size newSize(static_cast<int>(frameWidth * scalingFactor), static_cast<int>(frameHeight * scalingFactor));
//             cv::Mat resizedFrame;
//             cv::resize(concatenated, resizedFrame, newSize);

//             std::cout << "Resized frame dimensions: cols=" << resizedFrame.cols
//                       << ", rows=" << resizedFrame.rows << std::endl;

//             cv::imshow("Two Cameras Side by Side", resizedFrame);
//             if (cv::waitKey(1) == 'q')
//             {
//                 stopFlag = true;
//                 break;
//             }
//         }
//         else
//         {
//             // std::cerr << "Error: Concatenated frame is empty." << std::endl;
//         }
//     }
//     cv::destroyAllWindows();
// }
>>>>>>> Stashed changes
