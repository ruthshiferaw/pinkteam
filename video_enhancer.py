import cv2
import time
from enhancement_helpers import enhance_image

def main(video_path, lut_path=None, output_path="enhanced_output.mp4", white_balance=True, apply_dehazing=True, apply_clahe=True, apply_fast_filters_flag=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / 3, (frame_width, frame_height))

    frame_count = 0

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Frame skipping logic: process every 3rd frame
        if frame_count % 3 != 0:
            continue

        frame_start_time = time.time()

        current_frame = enhance_image(current_frame, white_balance, apply_dehazing, apply_clahe, apply_fast_filters_flag, lut_path)

        frame_end_time = time.time()
        total_frame_time = round((frame_end_time - frame_start_time) * 1000, 2)
        print(f"Frame {frame_count}: Total Processing Time: {total_frame_time} ms")

        writer.write(current_frame)

    cap.release()
    writer.release()
    print("Video processing complete.")

if __name__ == "__main__":
    video_path = "Sample Videos/Vision_Test.mp4"  # Replace with your video file path
    lut_path = "LUTs/Underwater v1_1.GX014035.cube"  # Replace with your LUT file path
    # DONT INCLUDE lut_path IN THE HEADER IF YOU DON'T WANT TO USE LUT FILTER
    # VERY SLOW
    main(video_path)
