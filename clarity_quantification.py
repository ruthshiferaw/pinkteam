#for Ruth only:
import sys
sys.path.append(r"c:\users\ruth\appdata\local\packages\pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0\localcache\local-packages\python312\site-packages")  # Replace with your actual path

import cv2
import numpy as np
from skimage import exposure, util
from enhancement_helpers import dehaze_image

def calculate_noise_level(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Measure noise using the variance of the Laplacian
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise_level

def calculate_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size
    return edge_density

def calculate_cnr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    mean_intensity = np.mean(gray)
    noise_level = np.std(gray)
    cnr = mean_intensity / (noise_level + 1e-5)  # Avoid division by zero
    return cnr

def custom_clarity_metric(img):
    uiqm_score = calculate_uiqm(img)  # Your existing UIQM calculation
    noise_level = calculate_noise_level(img)
    edge_density = calculate_edge_density(img)
    cnr = calculate_cnr(img)

    # Combine with weights that you can adjust based on performance
    clarity_score = (0.5 * uiqm_score) + (0.3 * edge_density) + (0.2 * cnr) - (0.2 * noise_level)
    return clarity_score

#UIQM Calculation (Colorfulness, Sharpness, Contrast)
def calculate_uiqm(img):
    # Convert to grayscale for sharpness and contrast measures only if it's an RGB image
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_r, mean_g, mean_b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    else:
        gray = img  # Image is already grayscale
        mean_r = mean_g = mean_b = np.mean(img)
    # Continue with the rest of the function using mean_r, mean_g, and mean_b

    colorfulness = np.sqrt((mean_r - mean_g) ** 2 + (mean_g - mean_b) ** 2)
    # Contrast calculation (standard deviation of grayscale image)
    contrast = gray.std()
    # Sharpness calculation (variance of Laplacian)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return colorfulness + contrast + sharpness

image = cv2.imread("Sample Images/1.png")
enhanced_image = cv2.imread("Enhanced Videos/dehazed_img4.png")
# enhanced_image, n = dehaze_image(image, 15, 15, 120, 1e-4, 0.1)
haze_free_image = cv2.imread("Sample Images/3.jpg")

print(f"UIQM Score of original image: {custom_clarity_metric(image)/0.8}%")
print(f"UIQM Score of enhanced image: {custom_clarity_metric(enhanced_image)/0.8}%")
# print(f"UIQM Score of haze-free image: {custom_clarity_metric(haze_free_image)}")

# Display images for comparison
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
#cv2.imshow('Haze-free Image', haze_free_image)
cv2.waitKey(0)
cv2.destroyAllWindows()