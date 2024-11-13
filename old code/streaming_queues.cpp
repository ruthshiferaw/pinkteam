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

void captureCamera(int cameraIndex, std::queue<std::pair<cv::Mat, double>>& frameQueue, std::mutex& queueMutex, std::atomic<bool>& stopFlag);
void processFrames(std::queue<std::pair<cv::Mat, double>>& frameQueue1, std::queue<std::pair<cv::Mat, double>>& frameQueue2,
                   std::queue<cv::Mat>& processedQueue, std::mutex& queueMutex1, std::mutex& queueMutex2, 
                   std::mutex& processedQueueMutex, std::atomic<bool>& stopFlag);
void displayFrames(std::queue<cv::Mat>& processedQueue, std::mutex& processedQueueMutex, std::atomic<bool>& stopFlag);

int main() {
    std::queue<std::pair<cv::Mat, double>> frameQueue1, frameQueue2;
    std::queue<cv::Mat> processedQueue;
    std::mutex queueMutex1, queueMutex2, processedQueueMutex;
    std::atomic<bool> stopFlag(false);

    // Threads for capturing, processing, and displaying frames
    std::thread captureThread1(captureCamera, 1, std::ref(frameQueue1), std::ref(queueMutex1), std::ref(stopFlag));
    std::thread captureThread2(captureCamera, 2, std::ref(frameQueue2), std::ref(queueMutex2), std::ref(stopFlag));
    std::thread processingThread(processFrames, std::ref(frameQueue1), std::ref(frameQueue2), std::ref(processedQueue), 
                                 std::ref(queueMutex1), std::ref(queueMutex2), std::ref(processedQueueMutex), std::ref(stopFlag));
    std::thread displayThread(displayFrames, std::ref(processedQueue), std::ref(processedQueueMutex), std::ref(stopFlag));

    // Wait for user to stop the program
    std::cin.get();
    stopFlag = true;

    // Join threads
    captureThread1.join();
    captureThread2.join();
    processingThread.join();
    displayThread.join();

    return 0;
}

// Capture frames and put them in the queue
void captureCamera(int cameraIndex, std::queue<std::pair<cv::Mat, double>>& frameQueue, std::mutex& queueMutex, std::atomic<bool>& stopFlag) {
    cv::VideoCapture cap(cameraIndex, cv::CAP_DSHOW); // Use DirectShow on Windows
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
        return;
    }

    while (!stopFlag) {
        cv::Mat frame;
        if (cap.read(frame)) {
            double timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(std::make_pair(frame, timestamp));
            std::this_thread::sleep_for(std::chrono::milliseconds(33));  // Simulate 30 FPS
        } else {
            std::cerr << "Error: Failed to capture frame from camera " << cameraIndex << std::endl;
            break;
        }
    }
    cap.release();
}

// Process frames from both queues and add the processed frame to the processed queue
void processFrames(std::queue<std::pair<cv::Mat, double>>& frameQueue1, std::queue<std::pair<cv::Mat, double>>& frameQueue2,
                   std::queue<cv::Mat>& processedQueue, std::mutex& queueMutex1, std::mutex& queueMutex2, 
                   std::mutex& processedQueueMutex, std::atomic<bool>& stopFlag) {
    while (!stopFlag) {
        std::lock_guard<std::mutex> lock1(queueMutex1);
        std::lock_guard<std::mutex> lock2(queueMutex2);

        if (!frameQueue1.empty() && !frameQueue2.empty()) {
            auto frameData1 = frameQueue1.front();
            auto frameData2 = frameQueue2.front();

            if (std::abs(frameData1.second - frameData2.second) <= 0.03) {
                cv::Mat sq1 = frameData1.first(cv::Rect(420, 0, 1080, 1080));
                cv::Mat sq2 = frameData2.first(cv::Rect(420, 0, 1080, 1080));
                cv::resize(sq1, sq1, cv::Size(1216, 1216));
                cv::resize(sq2, sq2, cv::Size(1216, 1216));

                cv::Mat leftSide, rightSide;
                cv::copyMakeBorder(sq1, leftSide, 112, 112, 0, 64, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                cv::copyMakeBorder(sq2, rightSide, 112, 112, 64, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

                cv::Mat concatenated;
                cv::hconcat(leftSide, rightSide, concatenated);

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

// Display frames from the processed queue
void displayFrames(std::queue<cv::Mat>& processedQueue, std::mutex& processedQueueMutex, std::atomic<bool>& stopFlag) {
    int displayWidth = 1280, displayHeight = 720;

    while (!stopFlag) {
        cv::Mat concatenated;
        {
            std::lock_guard<std::mutex> lock(processedQueueMutex);
            if (!processedQueue.empty()) {
                concatenated = processedQueue.front();
                processedQueue.pop();
            }
        }

        if (!concatenated.empty()) {
            int frameWidth = concatenated.cols;
            int frameHeight = concatenated.rows;
            double scalingFactor = std::min(displayWidth / (double)frameWidth, displayHeight / (double)frameHeight);
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
