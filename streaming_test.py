import cv2
import numpy as np
import threading
import time

# Global variables to store frames from both cameras
frame1 = None
frame2 = None

#cam1 = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
#cam2 = "v4l2src device=/dev/video4 ! videoconvert ! appsink"

# #try lower quality
#cam1 = "v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"
#cam2 = "v4l2src device=/dev/video4 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"

#Lock objects to safely update frames from different threads
frame1_lock = threading.Lock()
frame2_lock = threading.Lock()

def capture_camera1():
    global frame1
    print("camera 1 started")
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Camera 1 (usually /dev/video0)
    if not cap1.isOpened():
        print("Error: Could not open camera 1.")
        return
    
    while True:
        ret1, f1 = cap1.read()
        if ret1:
            #with frame1_lock:
                frame1 = f1
                time.sleep(1.0/30)
        else:
            print("Error: Failed to capture frame from camera 1.")
            break

    cap1.release()

def capture_camera2():
    global frame2
    print("camera 2 started")
    cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Camera 2 (usually /dev/video1)
    if not cap2.isOpened():
        print("Error: Could not open camera 2.")
        return
    
    while True:
        ret2, f2 = cap2.read()
        if ret2:
            #with frame2_lock:
                frame2 = f2
                time.sleep(1.0/30)
        else:
            print("Error: Failed to capture frame from camera 2.")
            break

    cap2.release()

def display():
    global frame1, frame2

    while True:
        # Wait until both frames are available
        #with frame1_lock, frame2_lock:
        if frame1 is not None and frame2 is not None:
            # Resize frames to match in size (optional but recommended for a better side-by-side display)
            height1, width1 = frame1.shape[:2]
            height2, width2 = frame2.shape[:2]

            # Resize second frame to match the first frame's size (if needed)
            if height1 != height2 or width1 != width2:
                frame2 = cv2.resize(frame2, (width1, height1))

            # Concatenate the frames side by side (horizontally)
            concatenated = np.hstack((frame1, frame2))

            # Display the concatenated image
            cv2.imshow('Two Cameras Side by Side', concatenated)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Create and start the camera threads
thread1 = threading.Thread(target=capture_camera1)
thread2 = threading.Thread(target=capture_camera2)

# Start threads
thread2.start()
thread1.start()

# Start the display function in the main thread
display()

thread1.join() #idk if this should be before or after display
thread2.join()

cv2.destroyAllWindows()