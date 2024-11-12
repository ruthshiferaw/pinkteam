#include "enhancement_helpers.h"

// Downscale Image
cv::Mat downscaleImage(const cv::Mat &img, double scaleFactor)
{
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
    return resizedImg;
}

// Upscale Image
cv::Mat upscaleImage(const cv::Mat &img, const cv::Size &targetSize)
{
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, targetSize, 0, 0, cv::INTER_LINEAR);
    return resizedImg;
}

// Apply CLAHE
cv::Mat applyCLAHE(const cv::Mat &frame)
{
    cv::Mat labFrame, result;
    cv::cvtColor(frame, labFrame, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels(3);
    cv::split(labFrame, labChannels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(16, 16));
    clahe->apply(labChannels[0], labChannels[0]);
    cv::merge(labChannels, labFrame);
    cv::cvtColor(labFrame, result, cv::COLOR_Lab2BGR);
    return result;
}

// Apply White Balance
cv::Mat applyWhiteBalance(const cv::Mat &img)
{
    cv::Scalar avg = cv::mean(img);
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    channels[0] = channels[0] * (avg[2] / avg[0]);
    channels[1] = channels[1] * (avg[2] / avg[1]);
    cv::Mat balancedImg;
    cv::merge(channels, balancedImg);
    return balancedImg;
}

// Apply Fast Filters
cv::Mat applyFastFilters(const cv::Mat &frame)
{
    cv::Mat result;
    cv::bilateralFilter(frame, result, 9, 75, 75);
    return result;
}

// Dark Channel Prior
cv::Mat darkChannelPrior(const cv::Mat &img, int patchSize)
{
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    cv::Mat minImg;
    cv::min(channels[0], channels[1], minImg);
    cv::Mat darkChannel;
    cv::min(minImg, channels[2], darkChannel);
    cv::erode(darkChannel, darkChannel, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patchSize, patchSize)));
    return darkChannel;
}

// // Atmospheric Light
// cv::Vec3f atmosphericLight(const cv::Mat &img, const cv::Mat &darkChannel)
// {
//     int nPixels = static_cast<int>(0.001 * darkChannel.total());
//     cv::Mat flatImg = img.reshape(1, img.total());
//     cv::Mat flatDarkChannel = darkChannel.reshape(1, darkChannel.total());
//     cv::Mat indices;
//     cv::sortIdx(flatDarkChannel, indices, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

//     cv::Vec3f A(0, 0, 0);
//     for (int i = 0; i < nPixels; i++)
//     {
//         int idx = indices.at<int>(i);
//         A += flatImg.at<cv::Vec3f>(idx);
//     }
//     A *= (1.0 / nPixels);
//     return A;
// }

// Atmospheric Light with std::nth_element
cv::Vec3f atmosphericLight(const cv::Mat &img, const cv::Mat &darkChannel, double sampleFraction = 0.001)
{
    // Flatten the image and dark channel
    cv::Mat flatImg = img.reshape(1, img.total());                         // Flattened image as rows of Vec3b
    cv::Mat flatDarkChannel = darkChannel.reshape(1, darkChannel.total()); // Flattened dark channel

    // Determine the number of brightest pixels to sample
    int numPixels = std::max(1, static_cast<int>(sampleFraction * flatDarkChannel.total()));

    // Convert the flattened dark channel to a vector for std::nth_element
    std::vector<std::pair<float, int>> darkChannelWithIndices;
    darkChannelWithIndices.reserve(flatDarkChannel.total());

    // Fill the vector with pixel values and their indices
    for (int i = 0; i < flatDarkChannel.total(); ++i)
    {
        darkChannelWithIndices.emplace_back(flatDarkChannel.at<float>(i), i);
    }

    // Use nth_element to find the top `numPixels` brightest pixels
    std::nth_element(
        darkChannelWithIndices.begin(),
        darkChannelWithIndices.begin() + numPixels,
        darkChannelWithIndices.end(),
        [](const std::pair<float, int> &a, const std::pair<float, int> &b)
        {
            return a.first > b.first; // Sort in descending order based on dark channel intensity
        });

    // Compute the mean atmospheric light using the top `numPixels` brightest pixels
    cv::Vec3f A(0, 0, 0);
    for (int i = 0; i < numPixels; ++i)
    {
        int idx = darkChannelWithIndices[i].second; // Get the index of the i-th brightest pixel
        A += flatImg.at<cv::Vec3f>(idx);
    }
    A *= (1.0 / numPixels); // Average the sum to get the atmospheric light

    return A;
}

// Estimate Transmission
cv::Mat estimateTransmission(const cv::Mat &img, const cv::Vec3f &A, double omega)
{
    cv::Mat normImg;
    img.convertTo(normImg, CV_32F);
    cv::Mat scaledImg;
    normImg /= cv::Scalar(A[0], A[1], A[2]);
    cv::Mat darkChannel = darkChannelPrior(normImg, 15); // Adjust patchSize as needed
    cv::Mat transmission = 1 - omega * darkChannel;
    return transmission;
}

// Guided Filter
cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int radius, double epsilon)
{
    cv::Mat meanI, meanP, meanIp, meanII, varI, a, b;
    cv::boxFilter(I, meanI, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(p, meanP, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(p), meanIp, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(I), meanII, CV_64F, cv::Size(radius, radius));
    varI = meanII - meanI.mul(meanI);
    a = (meanIp - meanI.mul(meanP)) / (varI + epsilon);
    b = meanP - a.mul(meanI);
    cv::boxFilter(a, a, CV_64F, cv::Size(radius, radius));
    cv::boxFilter(b, b, CV_64F, cv::Size(radius, radius));
    return a.mul(I) + b;
}

// Recover Scene
cv::Mat recoverScene(const cv::Mat &img, const cv::Vec3f &A, const cv::Mat &t, double t0)
{
    cv::Mat result;
    img.convertTo(result, CV_32F);
    cv::Mat transmission = cv::max(t, t0);
    result = (result - cv::Scalar(A[0], A[1], A[2])) / transmission + cv::Scalar(A[0], A[1], A[2]);
    result.convertTo(result, CV_8U); // Convert back to 8-bit
    return result;
}

// Dehaze Image
cv::Mat dehazeImage(const cv::Mat &img, double scaleFactor, int patchSize)
{
    cv::Mat smallImg = downscaleImage(img, scaleFactor);
    cv::Mat darkChannel = darkChannelPrior(smallImg, patchSize);
    cv::Vec3f A = atmosphericLight(smallImg, darkChannel);
    cv::Mat transmission = estimateTransmission(smallImg, A, 0.95);
    cv::Mat graySmall;
    cv::cvtColor(smallImg, graySmall, cv::COLOR_BGR2GRAY);
    graySmall.convertTo(graySmall, CV_32F, 1.0 / 255.0);
    cv::Mat transmissionRefined = guidedFilter(graySmall, transmission, 15, 0.001);
    cv::Mat upscaledTransmission = upscaleImage(transmissionRefined, img.size());
    return recoverScene(img, A, upscaledTransmission, 0.1);
}

// Enhance Image (wrapper)
std::pair<cv::Mat, std::map<std::string, double>> enhanceImage(
    const cv::Mat &img, bool whiteBalance = false, bool applyDehazing = true,
    bool useCLAHE = false, bool applyFastFiltersFlag = false);
{

    std::map<std::string, double> timings;
    cv::Mat result = img.clone();
    auto start = std::chrono::high_resolution_clock::now();

    if (whiteBalance)
    {
        result = applyWhiteBalance(result);
        timings["white_balance"] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    }
    if (applyDehazing)
    {
        start = std::chrono::high_resolution_clock::now();
        result = dehazeImage(result);
        timings["dehazing"] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    }
    if (useCLAHE)
    {
        start = std::chrono::high_resolution_clock::now();
        result = applyCLAHE(result);
        timings["clahe"] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    }
    if (applyFastFiltersFlag)
    {
        start = std::chrono::high_resolution_clock::now();
        result = applyFastFilters(result);
        timings["fast_filters"] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    }
    return {result, timings};
}
