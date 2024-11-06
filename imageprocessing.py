# # Ruth, uncomment these next two lines:
# import sys
# sys.path.append(r"c:\users\ruth\appdata\local\packages\pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0\localcache\local-packages\python312\site-packages")  # Replace with your actual path

import cv2
import numpy as np

def enhance_image(image_path):
    # Step 1: Read and Preprocess the Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    # Step 2: White Balance Adjustment
    def white_balance(img):
        # Compute the average color of each channel
        avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))
        # Scale each channel
        img[:, :, 0] = np.clip(img[:, :, 0] * (avg_r / avg_b), 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * (avg_r / avg_g), 0, 255)
        return img
    img = white_balance(img)
    # Step 3: Dark Channel Prior for Dehazing
    def dark_channel_prior(img, patch_size=15):
        # Min value in RGB channels per patch to get the dark channel
        dark_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        return cv2.erode(dark_channel, kernel)
    dark_channel = dark_channel_prior(img)
    # Estimate atmospheric light
    def atmospheric_light(img, dark_channel):
        flat_img = img.reshape(-1, 3)
        flat_dark = dark_channel.ravel()
        # Choose the top 0.1% brightest pixels in the dark channel
        num_pixels = int(0.001 * len(flat_dark))
        indices = np.argsort(flat_dark)[-num_pixels:]
        A = np.mean(flat_img[indices], axis=0)
        return A
    A = atmospheric_light(img, dark_channel)
    # Transmission estimation
    def estimate_transmission(img, A, omega=0.95):
        norm_img = img / A  # Normalize image with atmospheric light
        dark_channel = dark_channel_prior(norm_img)
        transmission = 1 - omega * dark_channel
        return transmission
    transmission = estimate_transmission(img, A)
    # Soft matting (Guided Filtering)
    def guided_filter(I, p, radius=60, epsilon=1e-3):
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        q = mean_a * I + mean_b
        return q
    # Refine transmission map using guided filter
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    transmission_refined = guided_filter(gray_img, transmission)
    # Recover the scene radiance
    def recover_scene(img, A, t, t0=0.1):
        t = np.maximum(t, t0)  # Avoid division by zero
        J = (img - A) / t[:, :, None] + A
        return np.clip(J, 0, 255).astype(np.uint8)
    recovered_img = recover_scene(img, A, transmission_refined)
    # Step 4: Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab_img = cv2.cvtColor(recovered_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab_img = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    return enhanced_img

# Usage
# # for Ruth:
# image_path = r"C:\Users\Ruth\Downloads\turbid.png"  # Replace with your image path

# for Taylor:
image_path = r"C:\Users\tgfox\OneDrive\Documents\Turbid_Water\Turbid1.png"

enhanced_img = enhance_image(image_path)
# Display the original and enhanced images
cv2.imshow("Enhanced Image", enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()