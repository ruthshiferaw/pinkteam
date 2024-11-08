import cv2
import time
from enhancement_helpers import apply_LUT, apply_CLAHE, apply_fast_filters

def main(video_path, lut_path=None, output_path="enhanced_output.mp4", use_LUT=True, use_CLAHE=True, use_temporal_smoothing=True, use_fast_filters=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / 3, (frame_width, frame_height))

    prev_frame = None
    frame_count = 0

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Track the total processing time per frame
        frame_start_time = time.time()

        # Frame skipping logic: process every 3rd frame
        if frame_count % 3 != 0:
            continue

        if use_LUT and lut_path:
            start = time.time()
            current_frame = apply_LUT(current_frame, lut_path)
            end = time.time()
            print(f"Frame {frame_count}: LUT Time: {round((end - start) * 1000, 2)} ms")

        if use_CLAHE:
            start = time.time()
            current_frame = apply_CLAHE(current_frame)
            end = time.time()
            print(f"Frame {frame_count}: CLAHE Time: {round((end - start) * 1000, 2)} ms")

        if use_fast_filters:
            start = time.time()
            current_frame = apply_fast_filters(current_frame)
            end = time.time()
            print(f"Frame {frame_count}: Fast Filters Time: {round((end - start) * 1000, 2)} ms")

        if use_temporal_smoothing and prev_frame is not None:
            smoothed_frame = cv2.addWeighted(prev_frame, 0.5, current_frame, 0.5, 0)
        else:
            smoothed_frame = current_frame

        # Write the processed frame to the output
        writer.write(smoothed_frame)
        prev_frame = current_frame.copy()

        # Calculate and print total time for the current frame
        frame_end_time = time.time()
        total_frame_time = round((frame_end_time - frame_start_time) * 1000, 2)
        print(f"Frame {frame_count}: Total Processing Time: {total_frame_time} ms")

    cap.release()
    writer.release()
    print("Video processing complete.")

if __name__ == "__main__":
    video_path = "Sample Videos/Vision_Test.mp4"  # Replace with your video file path
    lut_path = "LUTs/Underwater v1_1.GX014035.cube"  # Replace with your LUT file path
    main(video_path, lut_path, use_LUT=False)
