import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# x=np.arange(6)
# y = x.reshape(2,3)
# print(y)
# z = x.reshape(-1,1)
# print(z)
italy_street=cv2.imread("review_opencv/images/5.jpeg", 1)
italy_street_gray = cv2.cvtColor(italy_street, cv2.COLOR_BGR2GRAY)
ret, italy_street_global_threshold = cv2.threshold(italy_street_gray, 127, 255, cv2.THRESH_BINARY)
italy_street_adaptive_threshold = cv2.adaptiveThreshold(italy_street_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Original Image", italy_street)
cv2.imshow("Grayscale Image", italy_street_gray)
cv2.imshow("Global Threshold Image", italy_street_global_threshold)
cv2.imshow("Adaptive Threshold Image", italy_street_adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()