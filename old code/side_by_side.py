import cv2
import os

def side_by_side_video(video_path1, video_path2, output_dir):
    # Capture videos
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Get properties of the first video
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Output video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, 'side_by_side_output.mp4')
    output = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    frame_count = 0
    ret2, frame2 = cap2.read()  # Read the first frame of the second video

    while cap1.isOpened() and ret2:
        # Read frames from the first video
        ret1, frame1 = cap1.read()

        if not ret1:
            break

        # Update frame2 from the second video every 3 frames
        if frame_count % 3 == 0:
            ret2, frame2 = cap2.read()  # Read the next frame of the second video
            if not ret2:  # Break if the second video ends
                break

        # Concatenate frames side-by-side
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the combined frame to the output video
        output.write(combined_frame)

        frame_count += 1

    # Release resources
    cap1.release()
    cap2.release()
    output.release()
    print("Output video saved at:", output_path)

# File paths
video_path1 = r"C:\Users\tgfox\OneDrive\Documents\GitHub\pinkteam\Sample Videos\Vision_Test_33s.mp4"
video_path2 = r"C:\Users\tgfox\OneDrive\Documents\GitHub\pinkteam\Enhanced Videos\Enhanced_Vision_Test_33s_single_downscaled+no_fast_filter+no_clahe+no_white_balance.mp4"
output_dir = r"C:\Users\tgfox\Downloads\images_009" # replace with your own path

# Call function
side_by_side_video(video_path1, video_path2, output_dir)
