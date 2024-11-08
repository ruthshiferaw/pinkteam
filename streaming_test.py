import cv2
import numpy as np
import threading
import time
import imutils
#from imutils.video import VideoStream
from vidgear.gears import VideoGear

options = {'THREADED_QUEUE_MODE': True}

# Global variables to store frames from both cameras
frame1 = None
frame2 = None

cam1 = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
cam2 = "v4l2src device=/dev/video4 ! videoconvert ! appsink"

stream1 = VideoGear(source=0, logging=True, **options).start()
stream2 = VideoGear(source=4, logging=True, **options).start()

while True:
     frameA = stream1.read()
     frameB = stream2.read()

     if frameA is None or frameB is None:
          print("frame problem")
          break
     concatenated = np.hstack((frameA, frameB))

    # Display the concatenated image
     cv2.imshow('Two Cameras Side by Side', concatenated)
    #cv2.imshow("1", frameA)
    #cv2.imshow("2", frameB)

cv2.destroyAllWindows()
# #try lower quality
#cam1 = "v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"
#cam2 = "v4l2src device=/dev/video4 ! video/x-raw, framerate=30/1, width=800, height=450 ! videoconvert ! appsink"

# Lock objects to safely update frames from different threads
# frame1_lock = threading.Lock()
# frame2_lock = threading.Lock()

# def capture_camera1():
#     global frame1
#     print("camera 1 started")
#     cap1 = cv2.VideoCapture(cam1)  # Camera 1 (usually /dev/video0)
#     if not cap1.isOpened():
#         print("Error: Could not open camera 1.")
#         return
    
#     while True:
#         ret1, f1 = cap1.read()
#         if ret1:
#             #with frame1_lock:
#                 frame1 = f1
#                 time.sleep(1.0/30)
#         else:
#             print("Error: Failed to capture frame from camera 1.")
#             break

#     cap1.release()

# def capture_camera2():
#     global frame2
#     print("camera 2 started")
#     cap2 = cv2.VideoCapture(cam2)  # Camera 2 (usually /dev/video1)
#     if not cap2.isOpened():
#         print("Error: Could not open camera 2.")
#         return
    
#     while True:
#         ret2, f2 = cap2.read()
#         if ret2:
#             #with frame2_lock:
#                 frame2 = f2
#                 time.sleep(1.0/30)
#         else:
#             print("Error: Failed to capture frame from camera 2.")
#             break

#     cap2.release()

# def display():
#     global frame1, frame2

#     while True:
#         print("frame check", frame1 is not None, frame2 is not None)
#         # Wait until both frames are available
#         #with frame1_lock, frame2_lock:
#         if frame1 is not None and frame2 is not None:
#             # Resize frames to match in size (optional but recommended for a better side-by-side display)
#             height1, width1 = frame1.shape[:2]
#             height2, width2 = frame2.shape[:2]

#             # Resize second frame to match the first frame's size (if needed)
#             if height1 != height2 or width1 != width2:
#                 frame2 = cv2.resize(frame2, (width1, height1))

#             # Concatenate the frames side by side (horizontally)
#             concatenated = np.hstack((frame1, frame2))

#             # Display the concatenated image
#             cv2.imshow('Two Cameras Side by Side', concatenated)

#         # Exit the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()

# # Create and start the camera threads
# thread1 = threading.Thread(target=capture_camera1)
# thread2 = threading.Thread(target=capture_camera2)

# # Start threads
# thread2.start()
# thread1.start()
# print('hello world')

# # Start the display function in the main thread
# display()

# thread1.join() #idk if this should be before or after display
# thread2.join()

# print("shouldn't reach here")
# # Wait for both threads to finish


# #OLD CODE

# # cap1 = cv2.VideoCapture(cam1, cv2.CAP_GSTREAMER)
# # cap2 = cv2.VideoCapture(cam2, cv2.CAP_GSTREAMER)

# # if not cap1.isOpened():
# #     print('Error could not open webcam (cam1)')
# #     exit()

# # if not cap2.isOpened():
# #     print('Error could not open webcam (cam2)')
# #     exit()

# # while True:
# #     #ret1,frame1 = cap1.read()
# #     ret2,frame2 = cap2.read()

# #     if  not ret2:
# #         print("Error: Failed to capture frame1")
# #         break

# #     #frames = np.hstack((frame1,frame2))

# #     cv2.imshow('camera feed', frame2)

# #     #press q to quit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # #cap1.release()
# # cap2.release()
# # cv2.destroyAllWindows()