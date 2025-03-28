import cv2
import sys

s = 0
print(sys.argv)
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) !=27: # Destroy window on ESC key(27 is an Ascii code for ESC)
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release() # Release the camera image from this code
cv2.destroyWindow(win_name)