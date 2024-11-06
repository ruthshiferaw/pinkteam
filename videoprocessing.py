#import sys
#sys.path.append(r"C:\Users\corri\Videos\Discovery")  # Replace with your actual path

import cv2
import numpy as np
import faster_image as proc

def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
    
    if not out.isOpened():
        print("Error: Could not open output video for writing.")
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply the custom function to the frame
        modified_frame = proc.enhance_image(frame)
        
        # Write the modified frame to the output video
        out.write(modified_frame)
    
    # Release the video objects
    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
input_video_path = r"C:\Users\corri\Videos\Discovery\Vision_Test1.mp4"   # Provide the path to your input video file
output_video_path = r"C:\Users\corri\Videos\Discovery\Result1.mp4" # Output video path

process_video(input_video_path, output_video_path)
