import cv2
import numpy as np
import threading
import enhancement_helpers as enhance
import queue
import time

# Global variables to store frames from both cameras
frame1 = None
frame2 = None

frame_q = queue.Queue(maxsize = 10)

goal_dims = (1280, 1440)
square = (1200)
#cam1 = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
#cam2 = "v4l2src device=/dev/video4 ! videoconvert ! appsink"

# #try lower quality
#cam1 = "v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"
#cam2 = "v4l2src device=/dev/video4 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"

#Lock objects to safely update frames from different threads
#frame1_lock = threading.Lock()
#frame2_lock = threading.Lock()

def capture_camera1():
    global frame1
    print("camera 1 started")
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap1.isOpened():
        print("Error: Could not open camera 1.")
        return
    
    while True:
        ret1, f1 = cap1.read()
        if ret1:
            frame1 = f1
        else:
            print("Error: Failed to capture frame from camera 1.")
            break

    cap1.release()

def capture_camera2():
    global frame2
    print("camera 2 started")
    cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW) 
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap2.isOpened():
        print("Error: Could not open camera 2.")
        return
    
    while True:
        ret2, f2 = cap2.read()
        if ret2:
            frame2 = f2
        else:
            print("Error: Failed to capture frame from camera 2.")
            break

    cap2.release()

sliding_val = 500  #420 is exact center, 0 keeps inner edge, 480 max
start1, end1 = 1920-sliding_val-1080, 1920-sliding_val
start2, end2 = sliding_val, 1080+sliding_val

def processing(f1, f2):
    # crop the images to be square
    sq1 = f1[0:1080, start1:end1]
    sq1 = cv2.resize(sq1, (1216,1216))
    sq2 = f2[0:1080, start2:end2]
    sq2 = cv2.resize(sq2, (1216,1216))


    # add padding to get correct spacing
    left_side = cv2.copyMakeBorder(sq1, 112, 112, 0, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    right_side = cv2.copyMakeBorder(sq2, 112, 112, 64, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Concatenate the frames side by side (horizontally)
    concatenated = np.hstack((left_side, right_side))

    # do frame precessing here !!!
    #concatenated, timing = enhance.enhance_image(concatenated)

    # Display the concatenated image
    cv2.imshow('Two Cameras Side by Side', concatenated)


def display():
    global frame1, frame2
    square = 1080

    while True:
        if frame1 is not None and frame2 is not None:
            #queue = threading.Thread(target=processing, args=(frame1, frame2))
            #queue.start()
            processing(frame1,frame2)
            #time.sleep(1.0/30)
            
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return

# Create and start the camera threads
thread1 = threading.Thread(target=capture_camera1)
thread2 = threading.Thread(target=capture_camera2)


# Start threads
thread2.start()
thread1.start()

# Start the display function in the main thread
display()

#the program doesn't end right and idk how to fix that
#not a huge priority
quit()

# thread1.join()
# thread2.join()
