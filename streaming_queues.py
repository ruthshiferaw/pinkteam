import cv2
import numpy as np
import threading
import queue
import time

# Capture and put frames in queue
def capture_camera(camera_index, frame_queue, stop_event):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            timestamp = time.time()
            frame_queue.put([frame, timestamp])
            time.sleep(1.0 / 30)
        else:
            print(f"Error: Failed to capture frame from camera {camera_index}.")
            break

    cap.release()

# Process frames from both queues and display them
def processing_and_display(frame_queue1, frame_queue2, processed_queue, stop_event):
    sliding_val = 420
    start1, end1 = 1920 - sliding_val - 1080, 1920 - sliding_val
    start2, end2 = sliding_val, 1080 + sliding_val

    while not stop_event.is_set():
        if not frame_queue1.empty() and not frame_queue2.empty():
            frame1, timestamp1 = frame_queue1.get()
            frame2, timestamp2 = frame_queue2.get()

            if abs(timestamp1 - timestamp2) <= 0.03:
                sq1 = frame1[0:1080, start1:end1]
                sq1 = cv2.resize(sq1, (1216, 1216))
                sq2 = frame2[0:1080, start2:end2]
                sq2 = cv2.resize(sq2, (1216, 1216))

                left_side = cv2.copyMakeBorder(sq1, 112, 112, 0, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                right_side = cv2.copyMakeBorder(sq2, 112, 112, 64, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                concatenated = np.hstack((left_side, right_side))

                if processed_queue.full():
                    processed_queue.get()
                processed_queue.put(concatenated)
            else:
                if timestamp1 < timestamp2:
                    frame_queue1.get()
                else:
                    frame_queue2.get()

# Display frames from the processed queue
def display_frames(processed_queue, stop_event):
    prev_width, prev_height = 1280, 720
    frame_count = 0
    cv2.namedWindow('Two Cameras Side by Side', cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        if not processed_queue.empty():
            concatenated = processed_queue.get()
            
            if frame_count % 10 == 0:
                try:
                    window_width = int(cv2.getWindowProperty('Two Cameras Side by Side', cv2.WND_PROP_AUTOSIZE))
                    window_height = int(cv2.getWindowProperty('Two Cameras Side by Side', cv2.WND_PROP_AUTOSIZE))
                    if (window_width != prev_width or window_height != prev_height) and window_width > 0 and window_height > 0:
                        prev_width, prev_height = window_width, window_height
                        frame_height, frame_width = concatenated.shape[:2]
                        scaling_factor = min(window_width / frame_width, window_height / frame_height)
                        new_size = (int(frame_width * scaling_factor), int(frame_height * scaling_factor))
                        resized_frame = cv2.resize(concatenated, new_size)
                    else:
                        resized_frame = cv2.resize(concatenated, (prev_width, prev_height))
                except cv2.error:
                    resized_frame = cv2.resize(concatenated, (prev_width, prev_height))
            else:
                resized_frame = cv2.resize(concatenated, (prev_width, prev_height))

            cv2.imshow('Two Cameras Side by Side', resized_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()  # Signal all threads to stop
                break

    cv2.destroyAllWindows()

# Main execution
frame_queue1 = queue.Queue(maxsize=20)
frame_queue2 = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

# Threads
capture_thread1 = threading.Thread(target=capture_camera, args=(1, frame_queue1, stop_event))
capture_thread2 = threading.Thread(target=capture_camera, args=(2, frame_queue2, stop_event))
processing_thread = threading.Thread(target=processing_and_display, args=(frame_queue1, frame_queue2, processed_queue, stop_event))
display_thread = threading.Thread(target=display_frames, args=(processed_queue, stop_event))

# Start threads
capture_thread1.start()
capture_thread2.start()
processing_thread.start()
display_thread.start()

# Wait for threads to finish
capture_thread1.join()
capture_thread2.join()
processing_thread.join()
display_thread.join()