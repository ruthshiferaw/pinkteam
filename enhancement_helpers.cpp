#include "enhancement_helpers.h"

// Downscale Image
cv::Mat downscaleImage(const cv::Mat& img, double scaleFactor) {
    cv::Mat resizedImg;
    img.convertTo(resizedImg, CV_8UC3); // Ensure input is 8-bit color
    cv::resize(resizedImg, resizedImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
    return resizedImg;
}

// Upscale Image
cv::Mat upscaleImage(const cv::Mat& img, const cv::Size& targetSize) {
    cv::Mat resizedImg;
    img.convertTo(resizedImg, CV_8UC3); // Ensure input is 8-bit color
    cv::resize(resizedImg, resizedImg, targetSize, 0, 0, cv::INTER_LINEAR);
    return resizedImg;
}

// Apply CLAHE
cv::Mat applyCLAHE(const cv::Mat& frame) {
    cv::Mat labFrame, result;
    frame.convertTo(labFrame, CV_8UC3); // Ensure 8-bit color format
    cv::cvtColor(labFrame, labFrame, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels(3);
    cv::split(labFrame, labChannels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(16, 16));
    clahe->apply(labChannels[0], labChannels[0]);
    cv::merge(labChannels, labFrame);
    cv::cvtColor(labFrame, result, cv::COLOR_Lab2BGR);
    return result;
}

// Apply White Balance
cv::Mat applyWhiteBalance(const cv::Mat& img) {
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F); // Convert to 32-bit float
    cv::Scalar avg = cv::mean(imgFloat);

    std::vector<cv::Mat> channels(3);
    cv::split(imgFloat, channels);

    channels[0] = channels[0] * (avg[2] / avg[0]);
    channels[1] = channels[1] * (avg[2] / avg[1]);

    cv::Mat balancedImg;
    cv::merge(channels, balancedImg);
    balancedImg.convertTo(balancedImg, CV_8UC3); // Convert back to 8-bit
    return balancedImg;
}

// Apply Fast Filters
cv::Mat applyFastFilters(const cv::Mat& frame) {
    cv::Mat result;
    frame.convertTo(result, CV_8UC3); // Ensure 8-bit color
    cv::bilateralFilter(result, result, 9, 75, 75);
    return result;
}

// Dark Channel Prior
cv::Mat darkChannelPrior(const cv::Mat& img, int patchSize) {
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    // Ensure all channels are in CV_32F
    for (auto& channel : channels) {
        channel.convertTo(channel, CV_32F);
    }

    cv::Mat minImg;
    cv::min(channels[0], channels[1], minImg);
    cv::Mat darkChannel;
    cv::min(minImg, channels[2], darkChannel);

    // Apply erosion
    cv::erode(darkChannel, darkChannel, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patchSize, patchSize)));

    return darkChannel;
}


// Atmospheric Light
cv::Vec3f atmosphericLight(const cv::Mat& img, const cv::Mat& darkChannel) {
    int nPixels = static_cast<int>(0.001 * darkChannel.total());
    cv::Mat flatImg = img.reshape(1, img.total());
    cv::Mat flatDarkChannel = darkChannel.reshape(1, darkChannel.total());
    cv::Mat indices;
    cv::sortIdx(flatDarkChannel, indices, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

    cv::Vec3f A(0, 0, 0);
    for (int i = 0; i < nPixels; i++) {
        int idx = indices.at<int>(i);
        A += flatImg.at<cv::Vec3f>(idx);
    }
    A *= (1.0 / nPixels);
    return A;
}

// Estimate Transmission
cv::Mat estimateTransmission(const cv::Mat& img, const cv::Vec3f& A, double omega) {
    cv::Mat normImg;

    // Normalize image by atmospheric light
    img.convertTo(normImg, CV_32F);
    normImg /= cv::Scalar(A[0], A[1], A[2]);

    // Compute dark channel
    cv::Mat darkChannel = darkChannelPrior(normImg, 15);

    // Estimate transmission
    cv::Mat transmission = 1.0 - omega * darkChannel;

    return transmission;
}


// Guided Filter
cv::Mat guidedFilter(const cv::Mat& I, const cv::Mat& p, int radius, double epsilon) {
    cv::Mat meanI, meanP, meanIp, meanII, varI, a, b;

    // Convert to double precision for calculations
    I.convertTo(meanI, CV_64F);
    p.convertTo(meanP, CV_64F);

    cv::boxFilter(meanI, meanI, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(meanP, meanP, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(meanI.mul(meanP), meanIp, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(meanI.mul(meanI), meanII, CV_64F, cv::Size(radius, radius));

    varI = meanII - meanI.mul(meanI);
    a = (meanIp - meanI.mul(meanP)) / (varI + epsilon);
    b = meanP - a.mul(meanI);

    cv::boxFilter(a, a, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(b, b, CV_64F, cv::Size(radius, radius));

    // Compute filtered result
    return a.mul(I) + b;
}


// Recover Scene
cv::Mat recoverScene(const cv::Mat& img, const cv::Vec3f& A, const cv::Mat& t, double t0) {
    cv::Mat result, transmission32F;

    // Convert inputs to CV_32F for arithmetic operations
    img.convertTo(result, CV_32F);
    t.convertTo(transmission32F, CV_32F);

    // Ensure transmission is above t0
    cv::Mat transmission = cv::max(transmission32F, t0);

    // Perform recovery
    result = (result - cv::Scalar(A[0], A[1], A[2])) / transmission + cv::Scalar(A[0], A[1], A[2]);

    // Convert back to 8-bit for display
    result.convertTo(result, CV_8UC3, 255.0);

    return result;
}


// Dehaze Image
cv::Mat dehazeImage(const cv::Mat& img, double scaleFactor, int patchSize) {
    cv::Mat smallImg, darkChannel, transmissionRefined, upscaledTransmission, result;

    // Convert input to CV_32F for consistency
    img.convertTo(smallImg, CV_32F, 1.0 / 65535.0); // Normalize 16-bit image to 0-1 range if needed

    // Downscale the image
    smallImg = downscaleImage(smallImg, scaleFactor);

    // Compute dark channel
    darkChannel = darkChannelPrior(smallImg, patchSize);

    // Estimate atmospheric light
    cv::Vec3f A = atmosphericLight(smallImg, darkChannel);

    // Estimate transmission
    cv::Mat transmission = estimateTransmission(smallImg, A, 0.95);

    // Guided filter to refine transmission
    cv::Mat graySmall;
    cv::cvtColor(smallImg, graySmall, cv::COLOR_BGR2GRAY);
    graySmall.convertTo(graySmall, CV_32F, 1.0 / 255.0); // Ensure grayscale is in 0-1 range
    transmissionRefined = guidedFilter(graySmall, transmission, 15, 0.001);

    // Upscale the transmission map
    upscaledTransmission = upscaleImage(transmissionRefined, img.size());

    // Recover scene
    result = recoverScene(img, A, upscaledTransmission, 0.1);

    return result;
}


// Enhance Image (wrapper)
std::pair<cv::Mat, std::map<std::string, double>> enhanceImage(
    const cv::Mat& img, bool whiteBalance, bool applyDehazing, bool useCLAHE, bool applyFastFiltersFlag) {
    std::map<std::string, double> timings;
    cv::Mat result = img.clone();
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Initial image type: " << img.type() << ", size: " << img.size() << std::endl;

    if (whiteBalance) {
        std::cout << "Before white balance: type = " << result.type() << ", size = " << result.size() << std::endl;
        result = applyWhiteBalance(result);
        std::cout << "After white balance: type = " << result.type() << ", size = " << result.size() << std::endl;
        timings["white_balance"] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }

    if (applyDehazing) {
        start = std::chrono::high_resolution_clock::now();
        std::cout << "Before dehazing: type = " << result.type() << ", size = " << result.size() << std::endl;
        result = dehazeImage(result, 0.5, 15); // Example scaleFactor and patchSize
        std::cout << "After dehazing: type = " << result.type() << ", size = " << result.size() << std::endl;
        timings["dehazing"] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }

    if (useCLAHE) {
        start = std::chrono::high_resolution_clock::now();
        std::cout << "Before CLAHE: type = " << result.type() << ", size = " << result.size() << std::endl;
        result = applyCLAHE(result);
        std::cout << "After CLAHE: type = " << result.type() << ", size = " << result.size() << std::endl;
        timings["clahe"] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }

    if (applyFastFiltersFlag) {
        start = std::chrono::high_resolution_clock::now();
        std::cout << "Before fast filters: type = " << result.type() << ", size = " << result.size() << std::endl;
        result = applyFastFilters(result);
        std::cout << "After fast filters: type = " << result.type() << ", size = " << result.size() << std::endl;
        timings["fast_filters"] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }

    return {result, timings};
}
