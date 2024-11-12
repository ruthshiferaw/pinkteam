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

# Process frames from both queues and display them
def processing_and_display(frame_queue1, frame_queue2):
    #affects cropping location
    sliding_val = 420  #420 is exact center, 0 keeps inner edge
    start1, end1 = 1920-sliding_val-1080, 1920-sliding_val
    start2, end2 = sliding_val, 1080+sliding_val

    while True:
        if not frame_queue1.empty() and not frame_queue2.empty():
            # Get frames from both cameras
            f1 = frame_queue1.get()
            f2 = frame_queue2.get()

            # Process frames
            sq1 = f1[0:1080, start1:end1]
            sq1 = cv2.resize(sq1, (1216, 1216))
            sq2 = f2[0:1080, start2:end2]
            sq2 = cv2.resize(sq2, (1216, 1216))

            left_side = cv2.copyMakeBorder(sq1, 112, 112, 0, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            right_side = cv2.copyMakeBorder(sq2, 112, 112, 64, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            concatenated = np.hstack((left_side, right_side))

            time.sleep(0.1)
            # # Perform enhancement if needed
            # concatenated, timing = enhance.enhance_image(concatenated)

            # Display the concatenated image
            cv2.imshow('Two Cameras Side by Side', concatenated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Create and start the camera threads
frame_queue1 = queue.Queue(maxsize=20)
frame_queue2 = queue.Queue(maxsize=20)
thread1 = threading.Thread(target=capture_camera, args=(1, frame_queue1))
thread2 = threading.Thread(target=capture_camera, args=(2, frame_queue2))

# Start threads
thread1.start()
thread2.start()

# Start processing and displaying frames
processing_and_display(frame_queue1, frame_queue2)

# Ensure the program ends correctly by waiting for threads
thread1.join()
thread2.join()