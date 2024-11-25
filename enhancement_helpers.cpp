#include "enhancement_helpers.h"
#include <opencv2/opencv.hpp>
// #include <opencv2/ximgproc/edge_filter.hpp> // For guidedFilter
#include <iostream>
#include <chrono>
#include <ctime> // For generating timestamp

// // Downscale Image
// cv::Mat downscaleImage(const cv::Mat &img, double scaleFactor)
// {
//     cv::Mat resizedImg;
//     cv::resize(img, resizedImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
//     return resizedImg;
// }

// // Upscale Image
// cv::Mat upscaleImage(const cv::Mat &img, const cv::Size &targetSize)
// {
//     cv::Mat resizedImg;
//     cv::resize(img, resizedImg, targetSize, 0, 0, cv::INTER_LINEAR);
//     return resizedImg;
// }

// Dark Channel Prior
cv::Mat darkChannelPrior(const cv::Mat &src, int patchSize)
{
    cv::Mat minChannels;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // Apply minimum filter on each channel
    cv::min(channels[0], channels[1], minChannels);
    cv::min(minChannels, channels[2], minChannels);

    // Apply a minimum filter with a kernel of the given size
    cv::Mat darkChannelImage;
    cv::erode(minChannels, darkChannelImage, cv::Mat::ones(patchSize, patchSize, CV_8UC1));

    return darkChannelImage;
}

// Atmospheric Light
cv::Vec3b atmosphericLight(const cv::Mat &img, const cv::Mat &darkChannel)
{
    // Find the brightest pixels in the dark channel
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(darkChannel, &minVal, &maxVal, &minLoc, &maxLoc);

    // Atmospheric light is the value of the brightest pixel in the input image
    return img.at<cv::Vec3b>(maxLoc);
}

// Estimate Transmission
cv::Mat estimateTransmission(const cv::Mat &src, const cv::Vec3b &A, int size)
{
    cv::Mat transmission(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            // Estimate transmission using the dark channel value
            float darkChannelVal = std::min({pixel[0] / float(A[0]),
                                             pixel[1] / float(A[1]),
                                             pixel[2] / float(A[2])});

            transmission.at<uchar>(i, j) = static_cast<uchar>((1 - 0.95f * darkChannelVal) * 255);
        }
    }
    return transmission;
}

// Recover Scene
cv::Mat recoverScene(const cv::Mat &src, const cv::Vec3b &A, const cv::Mat &transmission, float t0)
{
    cv::Mat J = src.clone();

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            float t = std::max(transmission.at<uchar>(i, j) / 255.0f, t0);

            cv::Vec3b recoveredPixel;
            for (int c = 0; c < 3; c++)
            {
                recoveredPixel[c] = static_cast<uchar>(std::min(255.0f, ((pixel[c] - A[c]) / t) + A[c]));
            }
            J.at<cv::Vec3b>(i, j) = recoveredPixel;
        }
    }

    return J;
}

// Dehaze Image
cv::Mat dehazeImage(const cv::Mat &img, double scaleFactor, int patchSize)
{
    std::cout << "Dehazing started." << std::endl;

    // cv::Mat smallImg = downscaleImage(img, scaleFactor);

    cv::Mat darkChannel = darkChannelPrior(img, patchSize); // smallImg

    cv::Vec3b A = atmosphericLight(img, darkChannel); // smallImg

    cv::Mat transmission = estimateTransmission(img, A, 0.95); // smallImg

    // cv::Mat upscaledTransmission = upscaleImage(transmission, img.size());

    cv::Mat result = recoverScene(img, A, transmission, 0.1); // upscaledTransmission

    std::cout << "Dehazing completed." << std::endl;
    return result;
}

// Main for two camera processing, correct resizing & borders
int main()
{
    // Open external cameras (camera 1 and camera 2)
    cv::VideoCapture cap1(0); // External camera 1
    cv::VideoCapture cap2(2); // External camera 2

    if (!cap1.isOpened() || !cap2.isOpened())
    {
        std::cerr << "Error: Could not open both external cameras." << std::endl;
        if (!cap1.isOpened())
            std::cerr << "Camera 1 failed to open." << std::endl;
        if (!cap2.isOpened())
            std::cerr << "Camera 2 failed to open." << std::endl;
        return -1;
    }

    // Get properties from both cameras
    int frameWidth1 = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight1 = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps1 = cap1.get(cv::CAP_PROP_FPS);

    int frameWidth2 = static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight2 = static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps2 = cap2.get(cv::CAP_PROP_FPS);

    if (fps1 == 0 || fps2 == 0)
    {
        std::cerr << "Warning: Unable to fetch FPS from one or both cameras, setting default value of 30." << std::endl;
        fps1 = fps2 = 30; // Default FPS
    }

    if (frameWidth1 == 0 || frameHeight1 == 0 || frameWidth2 == 0 || frameHeight2 == 0)
    {
        std::cerr << "Error: Invalid properties for one or both cameras." << std::endl;
        return -1;
    }

    std::cout << "Camera 1 - Width: " << frameWidth1 << ", Height: " << frameHeight1 << ", FPS: " << fps1 << std::endl;
    std::cout << "Camera 2 - Width: " << frameWidth2 << ", Height: " << frameHeight2 << ", FPS: " << fps2 << std::endl;

    // Create unique filenames with timestamps
    std::time_t now = std::time(nullptr);
    std::string timestamp = std::to_string(now);

    std::string folderPath = "Enhanced Videos/";
    std::string rawLeftPath = folderPath + "raw_left_" + timestamp + ".avi";
    std::string rawRightPath = folderPath + "raw_right_" + timestamp + ".avi";

    // Define VideoWriters for raw footage
    cv::VideoWriter rawLeftWriter(rawLeftPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps1, cv::Size(frameWidth1, frameHeight1));
    cv::VideoWriter rawRightWriter(rawRightPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps2, cv::Size(frameWidth2, frameHeight2));

    if (!rawLeftWriter.isOpened() || !rawRightWriter.isOpened())
    {
        std::cerr << "Error: Could not open one or both raw footage writers." << std::endl;
        return -1;
    }

    // Define the codec and create VideoWriter object for combined output
    std::string outputPath = folderPath + "enhanced_split_screen_output" + timestamp + ".avi"; // Save as .avi for compatibility
    int combinedWidth = frameWidth1 + frameWidth2;                                             // Combined width for side-by-side display
    int combinedHeight = std::max(frameHeight1, frameHeight2);
    double fps = std::min(fps1, fps2); // Use the lower FPS to ensure synchronization

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(combinedWidth, combinedHeight));

    if (!writer.isOpened())
    {
        std::cerr << "Error: Could not open the output video for writing: " << outputPath << std::endl;
        return -1;
    }

    cv::Mat frame1, frame2, enhancedFrame1, enhancedFrame2, combinedFrame;
    int frameCount = 0;

    while (true)
    {
        cap1 >> frame1; // Capture frame from camera 1
        cap2 >> frame2; // Capture frame from camera 2

        if (frame1.empty() || frame2.empty())
        {
            std::cerr << "Warning: Empty frame encountered at frame " << frameCount << std::endl;
            break;
        }

        // Save raw footage
        rawLeftWriter.write(frame1);
        rawRightWriter.write(frame2);

        try
        {
            // Measure the time taken to enhance both frames
            auto start = std::chrono::high_resolution_clock::now();

            // Apply enhancement to both frames
            enhancedFrame1 = dehazeImage(frame1, 0.5, 15); // Update with appropriate parameters
            enhancedFrame2 = dehazeImage(frame2, 0.5, 15);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> processingTime = end - start;

            // Print processing time for the frames
            std::cout << "Frame " << frameCount++ << " enhancement time: " << processingTime.count() << " ms" << std::endl;

            // Resize both enhanced frames to match dimensions
            cv::resize(enhancedFrame1, enhancedFrame1, cv::Size(1216, 1216));
            cv::resize(enhancedFrame2, enhancedFrame2, cv::Size(1216, 1216));

            // Add borders to both frames
            cv::Mat leftSide, rightSide;
            cv::copyMakeBorder(enhancedFrame1, leftSide, 112, 112, 0, 64, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            cv::copyMakeBorder(enhancedFrame2, rightSide, 112, 112, 64, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            // Concatenate frames side by side
            cv::hconcat(leftSide, rightSide, combinedFrame);

            // Dynamically resize and display the combined frame
            int displayWidth = 1280, displayHeight = 720;
            double scalingFactor = std::min(displayWidth / (double)combinedFrame.cols, displayHeight / (double)combinedFrame.rows);
            cv::Size newSize(static_cast<int>(combinedFrame.cols * scalingFactor), static_cast<int>(combinedFrame.rows * scalingFactor));

            cv::Mat resizedFrame;
            cv::resize(combinedFrame, resizedFrame, newSize);

            // Dynamically resize and display the combined frame
            if (combinedFrame.size() != cv::Size(combinedWidth, combinedHeight))
            {
                cv::resize(combinedFrame, combinedFrame, cv::Size(combinedWidth, combinedHeight));
            }
            writer.write(combinedFrame);

            // Display the resized frame
            cv::imshow("Enhanced Split-Screen Output", resizedFrame);
            if (cv::waitKey(1) >= 0) // Press any key to exit
                break;
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "OpenCV error at frame " << frameCount << ": " << e.what() << std::endl;
            break;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Standard exception at frame " << frameCount << ": " << e.what() << std::endl;
            break;
        }
        catch (...)
        {
            std::cerr << "Unknown error occurred at frame " << frameCount << std::endl;
            break;
        }
    }

    // Release resources
    cap1.release();
    cap2.release();
    rawLeftWriter.release();
    rawRightWriter.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Raw footage saved as " << rawLeftPath << " and " << rawRightPath << std::endl;
    std::cout << "Enhanced split-screen video saved as " << outputPath << std::endl;
    return 0;
}

// // Main Function to process image "1.png"
// int main()
// {
//     // Hard-coded input and output video paths
//     std::string inputImagePath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Sample Images/1.png";
//     std::string outputImagePath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Enhanced Videos/enhanced_img1.png";

//     // Load the input image
//     cv::Mat inputImage = cv::imread(inputImagePath);

//     if (inputImage.empty())
//     {
//         std::cerr << "Error: Unable to load input image." << std::endl;
//         return -1;
//     }

//     // Set parameters
//     int patchSize = 15;

//     // Start timer
//     auto startTime = std::chrono::high_resolution_clock::now();

//     // Dehaze the image
//     cv::Mat outputImage = dehazeImage(inputImage, 0.5, patchSize);

//     // End timer
//     auto endTime = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsedTime = endTime - startTime;
//     std::cout << "Elapsed Time: " << elapsedTime.count() << " seconds." << std::endl;

//     // Save the output image
//     if (!cv::imwrite(outputImagePath, outputImage))
//     {
//         std::cerr << "Error: Unable to save output image." << std::endl;
//         return -1;
//     }

//     std::cout << "Dehazed image saved at: " << outputImagePath << std::endl;

//     return 0;
// }

// // Main Function to process video "Sample Videos\Vision_Test_33s.mp4", works!
// int main()
// {
//     // Input and output video paths
//     std::string inputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Sample Videos/Vision_Test_33s.mp4";
//     std::string outputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Enhanced Videos/Vision_Test_33s_dehazed.mp4";

//     // Open the input video
//     cv::VideoCapture video(inputVideoPath);
//     if (!video.isOpened())
//     {
//         std::cerr << "Error: Unable to open input video." << std::endl;
//         return -1;
//     }

//     // Get video properties
//     int fps = video.get(cv::CAP_PROP_FPS);
//     cv::Size frameSize(
//         static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
//         static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));

//     // Define codec and create VideoWriter
//     int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MJPEG codec
//     cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);

//     if (!videoWriter.isOpened())
//     {
//         std::cerr << "Error: Unable to open output video for writing." << std::endl;
//         return -1;
//     }

//     // Process each frame
//     cv::Mat frame, dehazedFrame;
//     while (video.read(frame))
//     {
//         if (frame.empty())
//             break;

//         // Apply dehazing
//         dehazedFrame = dehazeImage(frame, 0.5, 15);

//         // Write dehazed frame to output video
//         videoWriter.write(dehazedFrame);
//     }

//     std::cout << "Dehazed video saved at: " << outputVideoPath << std::endl;
//     return 0;
// }

// // Main Function to process video "Sample Videos\Vision_Test_33s.mp4"
// int main()
// {
//     // Input and output video paths
//     std::string inputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Sample Videos/Vision_Test_33s.mp4";
//     std::string outputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Enhanced Videos/Vision_Test_33s_dehazed.avi";

//     // Open the input video
//     cv::VideoCapture video(inputVideoPath);
//     if (!video.isOpened())
//     {
//         std::cerr << "Error: Unable to open input video." << std::endl;
//         return -1;
//     }

//     // Get video properties
//     int fps = video.get(cv::CAP_PROP_FPS);
//     cv::Size frameSize(
//         static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
//         static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));

//     // Define codec and create VideoWriter
//     int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MJPEG codec
//     cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);

//     if (!videoWriter.isOpened())
//     {
//         std::cerr << "Error: Unable to open output video for writing." << std::endl;
//         return -1;
//     }

//     // Process each frame
//     cv::Mat frame, dehazedFrame;
//     while (video.read(frame))
//     {
//         if (frame.empty())
//             break;

//         // Apply dehazing
//         dehazedFrame = dehazeImage(frame, 0.5, 15);

//         // Write dehazed frame to output video
//         videoWriter.write(dehazedFrame);
//     }

//     std::cout << "Dehazed video saved at: " << outputVideoPath << std::endl;
//     return 0;
// }

// Main Function to process image while live-streaming from webcam
// int main()
// {
//     // Open the default camera (camera 0)
//     cv::VideoCapture cap(0); // 0 corresponds to the default webcam
//     if (!cap.isOpened())
//     {
//         std::cerr << "Error: Could not open the webcam." << std::endl;
//         return -1;
//     }

//     // Get webcam properties
//     int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//     int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
//     double fps = cap.get(cv::CAP_PROP_FPS);

//     if (fps == 0)
//     {
//         std::cerr << "Warning: Unable to fetch FPS from webcam, setting to default value of 30." << std::endl;
//         fps = 30; // Default FPS
//     }

//     if (frameWidth == 0 || frameHeight == 0)
//     {
//         std::cerr << "Error: Invalid webcam properties (width: " << frameWidth
//                   << ", height: " << frameHeight << ")" << std::endl;
//         return -1;
//     }

//     std::cout << "Webcam properties - Width: " << frameWidth
//               << ", Height: " << frameHeight
//               << ", FPS: " << fps << std::endl;

//     // Define the codec and create VideoWriter object to save the output
//     std::string outputPath = "enhanced_webcam_output.avi"; // Save as .avi for compatibility
//     cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frameWidth, frameHeight));

//     if (!writer.isOpened())
//     {
//         std::cerr << "Error: Could not open the output video for writing: " << outputPath << std::endl;
//         return -1;
//     }

//     cv::Mat frame, enhancedFrame;
//     int frameCount = 0;

//     while (true)
//     {
//         cap >> frame; // Capture frame from webcam
//         if (frame.empty())
//         {
//             std::cerr << "Warning: Empty frame encountered at frame " << frameCount << std::endl;
//             break;
//         }

//         try
//         {
//             // Measure the time taken to enhance the frame
//             auto start = std::chrono::high_resolution_clock::now();

//             // Apply enhancement (e.g., dehaze or any other image processing)
//             enhancedFrame = dehazeImage(frame, 0.5, 15);

//             auto end = std::chrono::high_resolution_clock::now();
//             std::chrono::duration<double, std::milli> processingTime = end - start;

//             // Print processing time for the frame
//             std::cout << "Frame " << frameCount++ << " enhancement time: " << processingTime.count() << " ms" << std::endl;

//             // Write the enhanced frame to the output video
//             writer.write(enhancedFrame);

//             // Optional: display the enhanced frame
//             cv::imshow("Enhanced Webcam Output", enhancedFrame);
//             if (cv::waitKey(1) >= 0) // Press any key to exit
//                 break;
//         }
//         catch (const cv::Exception &e)
//         {
//             std::cerr << "OpenCV error at frame " << frameCount << ": " << e.what() << std::endl;
//             break;
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Standard exception at frame " << frameCount << ": " << e.what() << std::endl;
//             break;
//         }
//         catch (...)
//         {
//             std::cerr << "Unknown error occurred at frame " << frameCount << std::endl;
//             break;
//         }
//     }

//     // Release resources
//     cap.release();
//     writer.release();
//     cv::destroyAllWindows();

//     std::cout << "Enhanced webcam video saved as " << outputPath << std::endl;
//     return 0;
// }