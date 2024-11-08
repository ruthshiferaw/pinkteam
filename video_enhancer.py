#for Ruth only:
import sys
sys.path.append(r"c:\users\ruth\appdata\local\packages\pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0\localcache\local-packages\python312\site-packages")  # Replace with your actual path

import os
import cv2
import time
from enhancement_helpers import enhance_image

def main(video_path, lut_path=None, output_path="enhanced_output.mp4", note="", white_balance=True, apply_dehazing=True, apply_clahe=False, apply_fast_filters_flag=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    

    # Extract the original video name and define the enhanced video path
    original_name = os.path.splitext(os.path.basename(video_path))[0]
    # Build the enhanced video file name with optional note
    enhanced_name = f"Enhanced_{original_name}"
    if note:  # Append the note if it's provided
        enhanced_name += f"_{note}"
    enhanced_video_path = os.path.join("Enhanced Videos", f"{enhanced_name}.mp4")
    
    # Set output path to enhanced video path if not specified
    if output_path is None:
        output_path = enhanced_video_path

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / 3, (frame_width, frame_height))

    frame_count = 0
    total_processing_time = 0  # Initialize total processing time

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Frame skipping logic: process every 3rd frame
        if frame_count % 3 != 0:
            continue

        frame_start_time = time.time()

        current_frame, timings = enhance_image(current_frame, white_balance, apply_dehazing, apply_clahe, apply_fast_filters_flag, lut_path)

        frame_end_time = time.time()
        frame_processing_time = round((frame_end_time - frame_start_time) * 1000, 2)
        total_processing_time += frame_processing_time  # Add to total time
        print(f"Frame {frame_count}:")
        for func_name, func_time in timings.items():
            print(f"{func_name}: {func_time} ms")
        print(f"Total Processing Time for Frame {frame_count}: {frame_processing_time} ms\n")

        writer.write(current_frame)

    cap.release()
    writer.release()

    # Calculate and print the average processing time
    num_processed_frames = frame_count // 3  # Since we're only processing every 3rd frame
    if num_processed_frames > 0:
        average_processing_time = total_processing_time / num_processed_frames
        print(f"Average Processing Time per Frame: {round(average_processing_time, 2)} ms")
    else:
        print("No frames processed.")

    print("Video processing complete. Enhanced video saved at:", output_path)

if __name__ == "__main__":
    video_path = "Sample Videos/Vision_Test.mp4"  # Replace with your video file path
    lut_path = "LUTs/Underwater v1_1.GX014035.cube"  # Replace with your LUT file path
    # DONT INCLUDE lut_path IN THE HEADER IF YOU DON'T WANT TO USE LUT FILTER
    # VERY SLOW
    main(video_path, note="no_white_balance") #note indicates modifications, appended to end of video name, which is saved in "Enhanced Videos"

    #goal = 50-20ms on personal laptop, 100ms on pi
    #30 second video = ~1008 frames total
