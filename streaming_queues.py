import cv2
import numpy as np
import threading
import queue
import time
import enhancement_helpers as enhance


# Capture and put frames in queue
def capture_camera(camera_index, frame_queue):
    print(f"Camera {camera_index} started", time.time())
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame if queue is full
            frame_queue.put(frame)  # Insert new frame
            time.sleep(1.0/30)
        else:
            print(f"Error: Failed to capture frame from camera {camera_index}.")
            break

    cap.release()

# Process frames from both queues
def process_frames(frame_queue1, frame_queue2, processed_queue):
    while True:
        if not frame_queue1.empty() and not frame_queue2.empty():
            # Get frames from both cameras
            f1 = frame_queue1.get()
            f2 = frame_queue2.get()

            # Process frames
            sq1 = f1[0:1080, 420:1500]
            sq1 = cv2.resize(sq1, (1216, 1216))
            sq2 = f2[0:1080, 420:1500]
            sq2 = cv2.resize(sq2, (1216, 1216))

            left_side = cv2.copyMakeBorder(sq1, 112, 112, 0, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            right_side = cv2.copyMakeBorder(sq2, 112, 112, 64, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            concatenated = np.hstack((left_side, right_side))

            # # Optional: run a delay instead of running the enhancement
            # time.sleep(0.02)
            # # Optional: Perform enhancement if needed
            # concatenated, timing = enhance.enhance_image(concatenated)

            # Place the processed frame in the processed queue
            if processed_queue.full():
                processed_queue.get()  # Remove oldest processed frame if queue is full
            processed_queue.put(concatenated)

display_width=1280
display_height=720
new_size = None

# Display frames from the processed queue
def display_frames(processed_queue):
    while True:
        if not processed_queue.empty():
            # Get the processed frame
            concatenated = processed_queue.get()
            
            # Resize the frame to fit the specified dimensions
            frame_height, frame_width = concatenated.shape[:2]
            scaling_factor = min(display_width / frame_width, display_height / frame_height)
            new_size = (int(frame_width * scaling_factor), int(frame_height * scaling_factor))
            resized_frame = cv2.resize(concatenated, new_size)

            # Display the resized concatenated image
            cv2.imshow('Two Cameras Side by Side', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Create and start the camera threads
frame_queue1 = queue.Queue(maxsize=20)
frame_queue2 = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=10)

# Threads for capturing, processing, and displaying
capture_thread1 = threading.Thread(target=capture_camera, args=(1, frame_queue1))
capture_thread2 = threading.Thread(target=capture_camera, args=(2, frame_queue2))
processing_thread = threading.Thread(target=process_frames, args=(frame_queue1, frame_queue2, processed_queue))
display_thread = threading.Thread(target=display_frames, args=(processed_queue,))

thread1 = threading.Thread(target=capture_camera, args=(1, frame_queue1))
thread2 = threading.Thread(target=capture_camera, args=(2, frame_queue2))

# Start threads
capture_thread1.start()
capture_thread2.start()
processing_thread.start()
display_thread.start()

# Ensure the program ends correctly by waiting for threads
capture_thread1.join()
capture_thread2.join()
processing_thread.join()
display_thread.join()