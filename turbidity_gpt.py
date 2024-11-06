import cv2
import numpy as np

def calculate_uiqm(image):
    # Convert image to RGB if it is in another color format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Colorfulness calculation
    r, g, b = cv2.split(image_rgb)
    rg = np.absolute(r - g)
    yb = np.absolute(0.5 * (r + g) - b)
    colorfulness = np.mean(rg) + np.mean(yb)

    # 2. Contrast calculation
    contrast = image.std()

    # 3. Sharpness calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Combine into UIQM score
    uiqm = 0.4 * colorfulness + 0.3 * contrast + 0.3 * sharpness
    return uiqm

# Load an underwater image
image_path = 'path/to/your/underwater_image.jpg'
image = cv2.imread(image_path)

# Calculate UIQM score
uiqm_score = calculate_uiqm(image)
print(f'UIQM Score: {uiqm_score}')