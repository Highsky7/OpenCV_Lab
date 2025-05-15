import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
# Step1: Load Images and show them in one imgae using matplotlib
im1 = cv2.imread("review_opencv/images/form.jpg", 1)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.imread("review_opencv/images/scanned-form.jpg", 1)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.axis('off'), plt.imshow(im1), plt.title('Input Image')
plt.subplot(122), plt.axis('off'), plt.imshow(im2), plt.title('Scanned Image')
plt.show()
# Step2: Detect and draw keypoints(features) in both images using ORB feature detection method
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
im1_features = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im2_features = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.axis('off'), plt.imshow(im1_features), plt.title('Input Image Features')
plt.subplot(122), plt.axis('off'), plt.imshow(im2_features), plt.title('Scanned Image Features')
plt.show()
# Step3: Match features using Hamming distance and draw matches
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = list(matcher.match(descriptors1, descriptors2, None))
matches.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]
im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
plt.figure(figsize=(40, 10))
plt.imshow(im_matches), plt.axis('off'), plt.title('Matches')
plt.show()
# Step 4: Find homography and warp the image
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
height, width, channels = im1.shape
im2_reg = cv2.warpPerspective(im2, H, (width, height))
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.axis('off'), plt.imshow(im1), plt.title('Scanned Image')
plt.subplot(122), plt.axis('off'), plt.imshow(im2_reg), plt.title('Warped Image')
plt.show()