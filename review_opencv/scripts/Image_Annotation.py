import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 10)
image = cv2.imread("review_opencv/images/4.jpeg", cv2.IMREAD_COLOR)
plt.imshow(image[:, :, ::-1])
plt.show()
# print(image.shape)
# better_image = image[:, 450:2048]
# plt.imshow(better_image)
# plt.show()
# cv2.imwrite("review_opencv/images/4_better.jpeg", better_image)
imageLine = image.copy()
cv2.line(imageLine, (0, 0), (2047, 1535), (0, 255, 255), thickness=10, lineType=cv2.LINE_8) # x,y sequence
plt.imshow(imageLine[:, :, ::-1])
plt.show()
image_rectangle = image.copy()
cv2.rectangle(image_rectangle, (500, 1000), (1500, 500), (0, 255, 0), thickness=5, lineType=cv2.LINE_4)
plt.imshow(image_rectangle[:, :, ::-1])
plt.show()
image_circle = image.copy()
cv2.circle(image_circle, (1024, 768), 500, (255, 0, 0), thickness=10, lineType=cv2.LINE_AA)
plt.imshow(image_circle[:, :, ::-1])
plt.show()
image_text = image.copy()
text = "The Beauty of Paris above the Eiffel Tower"
fontScale = 2.0
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (255, 0, 0)
fontThickness = 5

cv2.putText(image_text, text, (100, 100), fontFace, fontScale, fontColor, fontThickness, lineType=cv2.LINE_AA)

plt.imshow(image_text[:, :, ::-1])
plt.show()