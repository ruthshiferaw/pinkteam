import cv2
import numpy as np

#camera tester code
# all_camera_idx_available = []

# for camera_idx in range(1,10):
#     cap = cv2.VideoCapture(camera_idx)
#     if cap.isOpened():
#         print(f'Camera index available: {camera_idx}')
#         all_camera_idx_available.append(camera_idx)
#         cap.release()


cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print('Error could not open webcam 1')
    exit()

while True:
    ret,frame = cap.read()

    if frame is not None:
        cv2.imshow('camera feed', frame)
        

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap1.release()
cap.release()
cv2.destroyAllWindows()