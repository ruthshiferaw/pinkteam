import cv2
import numpy as np
import time

def apply_LUT(frame, lut_path):
    lut, lut_size = load_cube_lut(lut_path)
    if lut_size == 0:
        print(f"Error: Failed to load LUT from {lut_path}")
        return frame

    normalized_frame = frame.astype(np.float32) / 255.0
    output_frame = np.zeros_like(normalized_frame)

    for i in range(normalized_frame.shape[0]):
        for j in range(normalized_frame.shape[1]):
            r, g, b = normalized_frame[i, j]
            r_idx = int(r * (lut_size - 1))
            g_idx = int(g * (lut_size - 1))
            b_idx = int(b * (lut_size - 1))
            output_frame[i, j] = lut[r_idx, g_idx, b_idx]

    output_frame = np.clip(output_frame * 255, 0, 255).astype(np.uint8)
    return output_frame

def load_cube_lut(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lut_size = 0
    lut_data = []

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if 'LUT_3D_SIZE' in line:
            lut_size = int(line.split()[-1])
            lut_data = []
            continue
        if 'DOMAIN_MIN' in line or 'DOMAIN_MAX' in line:
            continue

        if lut_size > 0 and len(line.split()) == 3:
            r, g, b = map(float, line.split())
            lut_data.append([r, g, b])

    lut_data = np.array(lut_data, dtype=np.float32)
    if lut_data.shape[0] != lut_size ** 3:
        raise ValueError("LUT data size mismatch. Check the .CUBE file format.")

    lut = lut_data.reshape((lut_size, lut_size, lut_size, 3))
    return lut, lut_size

def apply_CLAHE(frame):
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab_frame = cv2.merge((l, a, b))
    return cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)

def apply_white_balance(img):
    avg_b, avg_g, avg_r = np.mean(img, axis=(0, 1))
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_r / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_r / avg_g), 0, 255)
    return img

def apply_fast_filters(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

def enhance_image(img, white_balance=True, apply_dehazing=True, apply_clahe=True, apply_fast_filters_flag=True, lut_path=None):
    timings = {}  # Dictionary to store timing for each function
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if white_balance:
        start_time = time.time()
        img = apply_white_balance(img)
        timings['white_balance'] = round((time.time() - start_time) * 1000, 2)  # in ms

    if apply_dehazing:
        start_time = time.time()
        def dark_channel_prior(img, patch_size=15):
            dark_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
            return cv2.erode(dark_channel, kernel)

        dark_channel = dark_channel_prior(img)

        def atmospheric_light(img, dark_channel):
            flat_img = img.reshape(-1, 3)
            flat_dark = dark_channel.ravel()
            num_pixels = int(0.001 * len(flat_dark))
            indices = np.argsort(flat_dark)[-num_pixels:]
            A = np.mean(flat_img[indices], axis=0)
            return A

        A = atmospheric_light(img, dark_channel)

        def estimate_transmission(img, A, omega=0.95):
            norm_img = img / A
            dark_channel = dark_channel_prior(norm_img)
            transmission = 1 - omega * dark_channel
            return transmission

        transmission = estimate_transmission(img, A)

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

        def recover_scene(img, A, t, t0=0.1):
            t = np.maximum(t, t0)
            J = (img - A) / t[:, :, None] + A
            return np.clip(J, 0, 255).astype(np.uint8)

        img = recover_scene(img, A, transmission_refined)
        timings['dehazing'] = round((time.time() - start_time) * 1000, 2)

    if apply_clahe:
        start_time = time.time()
        img = apply_CLAHE(img)
        timings['clahe'] = round((time.time() - start_time) * 1000, 2)

    if lut_path:
        start_time = time.time()
        img = apply_LUT(img, lut_path)
        timings['lut'] = round((time.time() - start_time) * 1000, 2)

    if apply_fast_filters_flag:
        start_time = time.time()
        img = apply_fast_filters(img)
        timings['fast_filters'] = round((time.time() - start_time) * 1000, 2)

    return img, timings
