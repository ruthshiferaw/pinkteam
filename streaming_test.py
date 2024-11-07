import cv2
cap = cv2.VideoCapture('/dev/video0') #i think cv2 can't access webcam this way
#i was trying to use gstreamer to access the camera but haven't figured it out yet

if not cap.isOpened():
    print('Error could not open webcam')
    exit()

while True:
    ret,frame = cap.read()

    cv2.imshow('camera feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindoews()