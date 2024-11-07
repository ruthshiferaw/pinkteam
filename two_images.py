import cv2
import numpy as np
import faster_image as proc

def create_side_by_side(img1, img2):
    # Resize images to be the same height (if needed)
    if img1.shape[0] != img2.shape[0]:
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

    # Concatenate images horizontally
    side_by_side = np.hstack((img1, img2))
    return side_by_side

if __name__ == "__main__":
    # Replace these paths with your actual image paths or use images already loaded in memory
    original_image = cv2.imread(r"C:\Users\tgfox\OneDrive\Documents\GitHub\pinkteam\Sample Images\Turbid2.png")  # Replace with your original image path
    enhanced_image = proc.enhance_image(original_image)  # Assuming enhance_image is defined as in your code

    # Create side-by-side comparison image
    comparison_image = create_side_by_side(original_image, enhanced_image)

    # Display the result
    cv2.imshow("Original vs Enhanced", comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
