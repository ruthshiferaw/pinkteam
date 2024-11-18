import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera.")
    
#want to measure frame dims, actual fps

framecount = 0
time0 = 0

while framecount <= 45:
    ret, frame = cap.read()
    if ret:
        framecount += 1
        if framecount == 15:
            print("height, width: ", frame.shape[:2])
            time0 = time.time()
    else:
        print(f"Error: Failed to capture frame from camera.")
        break

#print(f"Counted {framecount} frames in 1 second.")
print(f"30 frames in {time.time()-time0} seconds")

print("Starting stream")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera feed', frame)
    else:
        print(f"Error: Failed to capture frame from camera.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
                #stop_event.set()  # Signal all threads to stop
        break
    
cv2.destroyAllWindows()
cap.release()
    
