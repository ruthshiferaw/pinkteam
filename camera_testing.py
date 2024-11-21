import cv2
import numpy as np
import time
#from arducam_mipicamera import ArducamCamera, arducam_init_camera, arducam_dispose_camera

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera.")
    
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"Operating at {cap.get(cv2.CAP_PROP_FPS)} fps")

width = 3840/2  # Desired width
height = 2160/2  # Desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if the resolution was set successfully
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {actual_width}x{actual_height}")

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 100) 
print("focus: ", cv2.CAP_PROP_FOCUS)

            
#want to measure frame dims, actual fps

# framecount = 0
# time0 = 0

# while framecount <= 45:
#     ret, frame = cap.read()
#     if ret:
#         framecount += 1
#         if framecount == 15:
#             print("height, width: ", frame.shape[:2])
#             time0 = time.time()
#     else:
#         print(f"Error: Failed to capture frame from camera.")
#         break

# #print(f"Counted {framecount} frames in 1 second.")
# print(f"30 frames in {time.time()-time0} seconds")

print("Starting stream")
fcount = 0
while True:
    ret, frame = cap.read()
    fcount+=1
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
    
