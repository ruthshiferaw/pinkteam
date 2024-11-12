import cv2
import numpy as np
import time

def downscale_image(img, scale_factor=0.5):
    return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

def upscale_image(img, target_shape):
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def apply_CLAHE(frame):
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l = clahe.apply(l)
    lab_frame = cv2.merge((l, a, b))
    return cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)

def apply_white_balance(img):
    # Compute mean intensity for each channel
    avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))

    # Normalize channels by scaling with mean values and converting to uint8
    img = img.astype(np.float32)
    img[:, :, 0] *= (avg_r / avg_b)
    img[:, :, 1] *= (avg_r / avg_g)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # # Scale each channel by ratios, using vectorized operations
    # scale = np.array([avg_r / avg_b, avg_r / avg_g, 1.0])  # CHANGED
    # img = (img * scale).clip(0, 255).astype(np.uint8)  # CHANGED
    return img

def apply_fast_filters(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

def dark_channel_prior(img, patch_size=15):
    dark_channel = np.min(img, axis=2) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel

def atmospheric_light(img, dark_channel): #sample_fraction=0.1
    flat_img = img.reshape(-1, 3)
    flat_dark = dark_channel.ravel()

    num_pixels = int(0.001 * len(flat_dark))
    indices = np.argsort(flat_dark)[-num_pixels:]

    # # Sample only a fraction of the brightest pixels
    # num_pixels = max(1, int(sample_fraction * len(flat_dark)))
    # indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

    A = np.mean(flat_img[indices], axis=0)
    return A

def estimate_transmission(img, A, omega=0.95):
    dark_channel = dark_channel_prior(img/A)
    transmission = 1 - omega * dark_channel
    return transmission

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

def recover_scene(img, A, t, t0=0.1):
    t = np.maximum(t, t0)
    img = (img - A) / t[:, :, None] + A
    return np.clip(img, 0, 255).astype(np.uint8)

# Main function that applies dehazing with downscaling
def dehaze_image(img, scale_factor=0.5, patch_size=15):
    small_img = downscale_image(img, scale_factor=scale_factor)  # DOWNscaled image
    dark_channel = dark_channel_prior(small_img, patch_size=patch_size)
    A = atmospheric_light(small_img, dark_channel)
    transmission = estimate_transmission(small_img, A)
    gray_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY) / 255.0
    transmission_refined = guided_filter(gray_img, transmission)
    transmission_refined = upscale_image(transmission_refined, img.shape)
    recovered_img = recover_scene(img, A, transmission_refined)
    return recovered_img

def enhance_image(img, white_balance=True, apply_dehazing=True, apply_clahe=True, apply_fast_filters_flag=True):
    timings = {}  # Dictionary to store timing for each function
    # Ensure the image is in uint8 RGB format at the beginning
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if white_balance:
        start_time = time.time()
        img = apply_white_balance(img)
        timings['white_balance'] = round((time.time() - start_time) * 1000, 2)  # in ms

    if apply_dehazing:
        start_time = time.time()
        img = dehaze_image(img)
        timings['dehazing'] = round((time.time() - start_time) * 1000, 2)

    if apply_clahe:
        start_time = time.time()
        img = apply_CLAHE(img)
        timings['clahe'] = round((time.time() - start_time) * 1000, 2)

    if apply_fast_filters_flag:
        start_time = time.time()
        img = apply_fast_filters(img)
        timings['fast_filters'] = round((time.time() - start_time) * 1000, 2)

    return img, timings
