import cv2
import numpy as np

def enhance_image(img, apply_white_balance=True, apply_dehazing=True, apply_clahe=True):
    # Step 1: Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if apply_white_balance:
        # White Balance Adjustment
        def white_balance(img):
            avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))
            img[:, :, 0] = np.clip(img[:, :, 0] * (avg_r / avg_b), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * (avg_r / avg_g), 0, 255)
            return img
        
        img = white_balance(img)

    if apply_dehazing:
        # Dark Channel Prior for Dehazing
        def dark_channel_prior(img, patch_size=15):
            dark_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
            return cv2.erode(dark_channel, kernel)
        
        dark_channel = dark_channel_prior(img)

        # Estimate Atmospheric Light
        def atmospheric_light(img, dark_channel):
            flat_img = img.reshape(-1, 3)
            flat_dark = dark_channel.ravel()
            num_pixels = int(0.001 * len(flat_dark))
            indices = np.argsort(flat_dark)[-num_pixels:]
            A = np.mean(flat_img[indices], axis=0)
            return A
        
        A = atmospheric_light(img, dark_channel)

        # Transmission Estimation
        def estimate_transmission(img, A, omega=0.95):
            norm_img = img / A
            dark_channel = dark_channel_prior(norm_img)
            transmission = 1 - omega * dark_channel
            return transmission
        
        transmission = estimate_transmission(img, A)

        # Guided Filter for Transmission Refinement
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
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        transmission_refined = guided_filter(gray_img, transmission)

        # Recover Scene Radiance
        def recover_scene(img, A, t, t0=0.1):
            t = np.maximum(t, t0)
            J = (img - A) / t[:, :, None] + A
            return np.clip(J, 0, 255).astype(np.uint8)
        
        img = recover_scene(img, A, transmission_refined)

    if apply_clahe:
        # CLAHE for Contrast Enhancement
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_img = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    return img
