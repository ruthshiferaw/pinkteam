import cv2
import numpy as np

cam1 = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
cam2 = "v4l2src device=/dev/video4 ! videoconvert ! appsink"

cap1 = cv2.VideoCapture(cam1, cv2.CAP_GSTREAMER)
cap2 = cv2.VideoCapture(cam2, cv2.CAP_GSTREAMER)

if not cap1.isOpened():
    print('Error could not open webcam (cam1)')
    exit()

if not cap2.isOpened():
    print('Error could not open webcam (cam2)')
    exit()

while True:
    #ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()

    if  not ret2:
        print("Error: Failed to capture frame1")
        break

    #frames = np.hstack((frame1,frame2))

    cv2.imshow('camera feed', frame2)

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap1.release()
cap2.release()
cv2.destroyAllWindows()