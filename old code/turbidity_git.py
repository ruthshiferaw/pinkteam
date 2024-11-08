# # Ruth, uncomment these next two lines:
# import sys
# sys.path.append(r"c:\users\ruth\appdata\local\packages\pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0\localcache\local-packages\python312\site-packages")  # Replace with your actual path

import cv2
import numpy as np
from tkinter import filedialog, Tk, messagebox

# Hide the main Tkinter window
root = Tk()
root.withdraw()

# Select an image file
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.bmp *.png *.tif *.jpeg")])
if not file_path:
    print("No file selected.")
    exit()

# Load the selected image
image = cv2.imread(file_path)
if image is None:
    print("Failed to load image.")
    exit()

# Get the dimensions of the image
height, width, channels = image.shape

# Display the image
cv2.imshow("Original Image", image)
cv2.waitKey(1)  # Display the window briefly

# Crop a central 100x100 region
crop_x1, crop_x2 = width // 2 - 50, width // 2 + 50
crop_y1, crop_y2 = height // 2 - 50, height // 2 + 50
cropped_im = image[crop_y1:crop_y2, crop_x1:crop_x2]

# Calculate mean of the red channel in the cropped region
m_red = np.mean(cropped_im[:, :, 2])  # Red channel is the third channel in OpenCV (BGR format)

# Calculate turbidity using the formula provided
turb = -123.03 * np.exp(-m_red / 202.008) - 184.47115 * np.exp(-m_red / 1157.359) + 313.5892
turbidity_out = round(-10.03 * turb + 1274.35)

# Display turbidity in a message box
message = f"Turbidity (in NTU): {turbidity_out} NTU"
messagebox.showinfo("Turbidity Measurement", message)

# Close the image display window
cv2.destroyAllWindows()