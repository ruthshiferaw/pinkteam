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
    // cv::Vec3b brightPixel = img.at<cv::Vec3b>(maxLoc);
    // cv::Vec3f A = cv::Vec3f(brightPixel) / 255.0f; // Normalize to [0,1]
    // return A;
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

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int radius, double epsilon)
{
    cv::Mat mean_I, mean_p, mean_Ip, mean_II, var_I, cov_Ip;

    // Ensure input images are CV_32F
    cv::Mat I_float, p_float;
    I.convertTo(I_float, CV_32F, 1.0 / 255.0);
    p.convertTo(p_float, CV_32F, 1.0 / 255.0);

    // Compute the means
    cv::boxFilter(I_float, mean_I, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(p_float, mean_p, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(I_float.mul(p_float), mean_Ip, CV_32F, cv::Size(radius, radius));

    // Compute variances and covariances
    cv::boxFilter(I_float.mul(I_float), mean_II, CV_32F, cv::Size(radius, radius));
    var_I = mean_II - mean_I.mul(mean_I);  // Variance of I
    cov_Ip = mean_Ip - mean_I.mul(mean_p); // Covariance of I and p

    // Compute coefficients a and b
    cv::Mat a = cov_Ip / (var_I + epsilon);
    cv::Mat b = mean_p - a.mul(mean_I);

    // Compute the means of a and b
    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(b, mean_b, CV_32F, cv::Size(radius, radius));

    // Compute the output q
    cv::Mat q = mean_a.mul(I_float) + mean_b;

    // Apply Gaussian blur for additional smoothing
    cv::GaussianBlur(q, q, cv::Size(5, 5), 0);
    // cv::blur(q, q, cv::Size(5, 5));
    // cv::Mat ref_q;
    // cv::bilateralFilter(q, ref_q, 9, 75, 75);

    // Scale back to 8-bit if needed
    q.convertTo(q, CV_8UC1, 255.0);

    return q;
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
                // clamping is necessary to remove bright artifacts from edges
                recoveredPixel[c] = static_cast<uchar>(std::clamp(((pixel[c] - A[c]) / t) + A[c], 0.0f, 255.0f)); // t*0.8f
            }
            J.at<cv::Vec3b>(i, j) = recoveredPixel;
        }
    }

    return J;
}

cv::Mat applyWhiteBalance(const cv::Mat &img)
{
    // Ensure the input image is of the correct type
    CV_Assert(img.type() == CV_8UC3);

    // Split the image into its B, G, and R channels
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // Compute the mean intensity for each channel
    double meanB = cv::mean(channels[0])[0];
    double meanG = cv::mean(channels[1])[0];
    double meanR = cv::mean(channels[2])[0];

    // Compute the overall mean intensity
    double meanGray = (meanB + meanG + meanR) / 3.0;

    // Scale each channel to normalize its intensity relative to the overall mean
    channels[0] = channels[0] * (meanGray / meanB);
    channels[1] = channels[1] * (meanGray / meanG);
    channels[2] = channels[2] * (meanGray / meanR);

    // Merge the channels back into a single image
    cv::Mat balancedImg;
    cv::merge(channels, balancedImg);

    // Clip the pixel values to [0, 255] to avoid overflow
    cv::threshold(balancedImg, balancedImg, 255, 255, cv::THRESH_TRUNC);

    return balancedImg;
}

// Dehaze Image
cv::Mat dehazeImage(const cv::Mat &img, int patchSize, int size, int radius, double epsilon, float t0)
{
    // std::cout << "Dehazing started." << std::endl;
    // cv::Mat smallImg = downscaleImage(img, scaleFactor);
    cv::Mat darkChannel = darkChannelPrior(img, patchSize); // smallImg
    cv::Vec3b A = atmosphericLight(img, darkChannel);       // smallImg
    // std::cout << "Atmospheric Light (A): [B=" << (int)A[0]
    //           << ", G=" << (int)A[1]
    //           << ", R=" << (int)A[2] << "]" << std::endl;
    cv::Mat transmission = estimateTransmission(img, A); // smallImg, size = 0.95?
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    cv::Mat refined_transmission = guidedFilter(grayImg, transmission, radius, epsilon);
    // cv::Mat upscaledTransmission = upscaleImage(transmission, img.size());
    cv::Mat result = recoverScene(img, A, refined_transmission, t0);
    result = applyWhiteBalance(result);
    // std::cout << "Dehazing completed." << std::endl;
    return result;
}

//// Main Function to process video "Sample Videos\Vision_Test_33s.mp4", works!
//int main()
//{
//    // Input and output video paths
//    std::string inputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Sample Videos/Vision_Test_33s.mp4";
//    std::string outputVideoPath = "C:/Users/Ruth/Documents/GitHub/pinkteam/Enhanced Videos/Vision_Test_33s_dehazed.mp4";
//
//
//    // Open the input video
//    cv::VideoCapture video(inputVideoPath);
//    if (!video.isOpened())
//    {
//        std::cerr << "Error: Unable to open input video." << std::endl;
//        return -1;
//    }
//
//    // Get video properties
//    int fps = video.get(cv::CAP_PROP_FPS);
//    cv::Size frameSize(
//        static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
//        static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));
//
//    // Define codec and create VideoWriter
//    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // ('m', 'p', '4', 'v') = .mp4 codec, ('M', 'J', 'P', 'G') = .avi codec
//    cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);
//
//    if (!videoWriter.isOpened())
//    {
//        std::cerr << "Error: Unable to open output video for writing." << std::endl;
//        return -1;
//    }
//
//    // Process each frame
//    cv::Mat frame, dehazedFrame;
//    int frameCount = 0;
//    while (video.read(frame))
//    {
//        if (frame.empty())
//            break;
//
//        // Measure the time taken to enhance frames
//        auto start = std::chrono::high_resolution_clock::now();
//
//        // Apply dehazing
//        dehazedFrame = dehazeImage(frame, 15, 15, 120, 1e-4, 0.1);
//
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double, std::milli> processingTime = end - start;
//        // Print processing time for the frames
//        std::cout << "Frame " << frameCount++ << " enhancement time: " << processingTime.count() << " ms" << std::endl;
//
//        // Write dehazed frame to output video
//        videoWriter.write(dehazedFrame);
//    }
//
//    std::cout << "Dehazed video saved at: " << outputVideoPath << std::endl;
//    return 0;
//}

 // Main for two camera processing, correct resizing & borders
 int main()
 {
     // Open external cameras (camera 1 and camera 2)
     cv::VideoCapture cap1(0); // External camera 1
     cv::VideoCapture cap2(2); // External camera 2

     bool filteringOn = false;

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

             if (filteringOn)
             {
                 // Apply enhancement to both frames
                 enhancedFrame1 = dehazeImage(frame1, 15, 15, 120, 1e-4, 0.1); // Update with appropriate parameters
                 enhancedFrame2 = dehazeImage(frame2, 15, 15, 120, 1e-4, 0.1);
             }
             else {
                 enhancedFrame1 = frame1;
                 enhancedFrame2 = frame2;
             }
             

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
             if (cv::waitKey(1) == 'f') // Press q to exit
                 filteringOn = true;
             if (cv::waitKey(1) == 'u') // Press q to exit
                 filteringOn = false;
             if (cv::waitKey(1) == 'q') // Press q to exit
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
//     cv::Mat outputImage = dehazeImage(inputImage, 15, 15, 120, 1e-4, 0.1);

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
//             enhancedFrame = dehazeImage(frame, 15, 15, 120, 1e-4, 0.1);

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