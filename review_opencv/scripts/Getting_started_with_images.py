import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

# from IPython.display import Image

# Image(filename="review_opencv/images/1.png")
# carla_img = cv2.imread("review_opencv/images/1.png", 0)
# carla_img_small = cv2.resize(carla_img, (256, 160))
# print(carla_img_small)
# print("Image size (H, W) is", carla_img_small.shape)
# print("Data type of image is:", carla_img_small.dtype)
# plt.imshow(carla_img)
# plt.show()
# plt.imshow(carla_img_small, cmap="gray")
# plt.show()
# Image(filename="review_opencv/images/2.jpeg")
taiwan_img = cv2.imread("review_opencv/images/2.jpeg", 1)
# taiwan_img_small = cv2.resize(taiwan_img, (256, 160))
# print("Image size (H, W, C) is", taiwan_img.shape) # img.shape is (H, W, C)
# print("Data type of image is:", taiwan_img_small.dtype)
# plt.imshow(taiwan_img)
# plt.show()
# taiwan_img_channels_reversed = taiwan_img[:, :, ::-1]
# plt.imshow(taiwan_img_channels_reversed)
# plt.show()
paris_img_bgr = cv2.imread("review_opencv/images/3.jpeg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(paris_img_bgr)
plt.figure(figsize=(20, 5)) # Image size is width: 20in x height: 5in

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel") # 1st position in row1, colomun 4
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel") # 2nd position in row1, colomun 4
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel") # 3rd position in row1, colomun 4

imgMerged = cv2.merge((b, g, r))
plt.subplot(144) # 4th position in row1, colomun 4
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")
plt.show()
paris_img_rgb = cv2.cvtColor(paris_img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(paris_img_rgb)
plt.show()
paris_img_hsv = cv2.cvtColor(paris_img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(paris_img_hsv)
plt.figure(figsize=(20, 5)) # Image size is width: 20in x height: 5in

plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("Hue Channel") # 1st position in row1, colomun 4
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("Saturation Channel") # 2nd position in row1, colomun 4
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("Value Channel") # 3rd position in row1, colomun 4
plt.subplot(144);plt.imshow(paris_img_rgb);plt.title("Original Image") # 4th position in row1, colomun 4
plt.show()

cv2.imwrite("review_opencv/images/3.jpeg", paris_img_bgr)
# Image(filename="review_opencv/images/3.jpeg")
paris_img_bgr = cv2.imread("review_opencv/images/3.jpeg", 1)
print("Img_bgr size (H, W, C) is", paris_img_bgr.shape) # img.shape is (H, W, C)
paris_img_gry = cv2.imread("review_opencv/images/3.jpeg", 0)
print("Img_gry size (H, W) is", paris_img_gry.shape)
eiffel_img = cv2.imread("review_opencv/images/eiffel.png", 1)
print(eiffel_img.shape)
eiffel_img_resized = cv2.resize(eiffel_img, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_AREA)
cv2.imwrite("review_opencv/images/eiffel_resized.png", eiffel_img_resized)
print(eiffel_img_resized.shape)