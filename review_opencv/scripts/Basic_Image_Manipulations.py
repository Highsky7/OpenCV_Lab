import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from IPython.display import Image

# taiwan_img = cv2.imread("review_opencv/images/2.jpeg", 0)
# taiwan_img_small = cv2.resize(taiwan_img, (18, 18)) # resize 시 (W, H)순서
# plt.imshow(taiwan_img_small, cmap="gray")
# print(taiwan_img)
# print(taiwan_img.shape)
# plt.show()
# Accessing Individual Pixels
# print(taiwan_img_small[0, 0])
# print(taiwan_img_small[0, 255])
# Modifying Image Pixels
# taiwan_img_small_copy = taiwan_img_small.copy()
# taiwan_img_small_copy[0, 0] = 0
# taiwan_img_small_copy[0, 1] = 0
# taiwan_img_small_copy[1, 0] = 0
# taiwan_img_small_copy[1, 1] = 0
# plt.imshow(taiwan_img_small_copy, cmap="gray")
# plt.show()
# print(taiwan_img_small_copy)
paris_img = cv2.imread("review_opencv/images/3.jpeg", cv2.IMREAD_COLOR)
paris_img_rgb = cv2.cvtColor(paris_img, cv2.COLOR_BGR2RGB)
# plt.imshow(paris_img_rgb)
# plt.show()
# cropped_paris_img = paris_img_rgb[750:1250, 500:1000] # H,W sequence(row, colomun)
# plt.imshow(cropped_paris_img)
# plt.show()
# half_resized_cropped_paris_img = cv2.resize(cropped_paris_img, None, fx=0.5, fy=0.5)
# plt.imshow(half_resized_cropped_paris_img)
# plt.show()
# desired_width = 100
# desired_height = 100
# dim = (desired_width, desired_height)
# desired_resized_cropped_paris_img = cv2.resize(cropped_paris_img, dim, interpolation=cv2.INTER_AREA)
# plt.imshow(desired_resized_cropped_paris_img)
# plt.show()
# desired_width = 100
# aspect_ratio = desired_width / cropped_paris_img.shape[1]
# desired_height = int(cropped_paris_img.shape[0] * aspect_ratio)
# dim = (desired_width, desired_height)
# ratio_cropped_paris_img = cv2.resize(cropped_paris_img, dsize=dim, interpolation=cv2.INTER_AREA)
# plt.imshow(ratio_cropped_paris_img)
# plt.show()
# half_resized_cropped_paris_img = half_resized_cropped_paris_img[:, :, ::-1]
# cv2.imwrite("review_opencv/images/3_half.jpeg", half_resized_cropped_paris_img)
# Image(filename="review_opencv/images/3_half.jpeg")
# cropped_paris_img = cropped_paris_img[:, :, ::-1]
# cv2.imwrite("review_opencv/images/3_cropped.jpeg", cropped_paris_img)
# Image(filename="review_opencv/images/3_cropped.jpeg")
paris_img_rgb_horz = cv2.flip(paris_img_rgb, 1)
paris_img_rgb_vert = cv2.flip(paris_img_rgb, 0)
paris_img_rgb_both = cv2.flip(paris_img_rgb, -1)
plt.figure(figsize=(20, 10)) # Image size is width: 20in x height: 10in
plt.subplot(1, 4, 1), plt.imshow(paris_img_rgb_horz), plt.title("Horizontally Flipped")
plt.subplot(1, 4, 2), plt.imshow(paris_img_rgb_vert), plt.title("Vertically Flipped")
plt.subplot(1, 4, 3), plt.imshow(paris_img_rgb_both), plt.title("Both Flipped")
plt.subplot(1, 4, 4), plt.imshow(paris_img_rgb), plt.title("Original Image")
plt.show()