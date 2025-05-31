import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def readImagesAndtimes():
    filenames = ["review_opencv/images/img_0.033.jpg", "review_opencv/images/img_0.25.jpg", "review_opencv/images/img_2.5.jpg", "review_opencv/images/img_15.jpg"]
    times = np.array([1/30.0, 0.25, 2.5, 15.0], dtype = np.float32)
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)
    return images, times
images, times = readImagesAndtimes()
allignMTB = cv2.createAlignMTB()
allignMTB.process(images, images)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)
x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

ax = plt.figure(figsize = (30,10))
plt.title("Debevec Inverse Camera Response Function", fontsize = 24)
plt.xlabel("Measured Pixel Value", fontsize =22)
plt.ylabel("Calibrated Intensity", fontsize = 22)
plt.xlim([0, 260])
plt.grid()
plt.plot(x, y[:, 0], "b", x,y[:, 1], "g", x, y[:, 2], "r")
plt.show()# 25.05.31