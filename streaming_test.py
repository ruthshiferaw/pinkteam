import cv2
import numpy as np
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        #self.frame = None  #still figuring this out
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID, cv2.CAP_GSTREAMER)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

cam1 = "v4l2src device=/dev/video0 ! videoconvert ! appsink"
cam2 = "v4l2src device=/dev/video4 ! videoconvert ! appsink"

# Create two threads as follows
thread1 = camThread("Camera 1", cam1)
thread2 = camThread("Camera 2", cam2)
thread1.start()
thread2.start()

#find a way to horizontally concatenate frames

# cap1 = cv2.VideoCapture(cam1, cv2.CAP_GSTREAMER)
# cap2 = cv2.VideoCapture(cam2, cv2.CAP_GSTREAMER)

# if not cap1.isOpened():
#     print('Error could not open webcam (cam1)')
#     exit()

# if not cap2.isOpened():
#     print('Error could not open webcam (cam2)')
#     exit()

# while True:
#     #ret1,frame1 = cap1.read()
#     ret2,frame2 = cap2.read()

#     if  not ret2:
#         print("Error: Failed to capture frame1")
#         break

#     #frames = np.hstack((frame1,frame2))

#     cv2.imshow('camera feed', frame2)

#     #press q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# #cap1.release()
# cap2.release()
# cv2.destroyAllWindows()