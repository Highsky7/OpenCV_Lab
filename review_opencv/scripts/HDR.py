import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# Step1: Capture Multiple Exposures
# Note: This step is usually done with a camera, but here we will read images from files.
# Ensure the images are in the correct directory
def readImagesAndtimes():
    filenames = ["review_opencv/images/img_0.033.jpg", "review_opencv/images/img_0.25.jpg", "review_opencv/images/img_2.5.jpg", "review_opencv/images/img_15.jpg"]
    times = np.array([1/30.0, 0.25, 2.5, 15.0], dtype = np.float32)
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)
    return images, times
# Step2: Allign Images
# Note: This step is usually done with a tripod, but here we will use OpenCV's alignment function.
images, times = readImagesAndtimes()
allignMTB = cv2.createAlignMTB()
allignMTB.process(images, images)
# Step3: Estimate Camera Response Function
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)
x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

ax = plt.figure(figsize = (20,10))
plt.title("Debevec Inverse Camera Response Function", fontsize = 24)
plt.xlabel("Measured Pixel Value", fontsize =22)
plt.ylabel("Calibrated Intensity", fontsize = 22)
plt.xlim([0, 260])
plt.grid()
plt.plot(x, y[:, 0], "b", x,y[:, 1], "g", x, y[:, 2], "r")
plt.show()# 25.05.31
# Step4: Merge Exposure into an HDR Image
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# Step5: Tonemap the HDR Image
# Tonemaping with Drago's method
print("Tonemaping using Drago's method ... ")
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago # Scale the image to [0, 1]
cv2.imwrite("review_opencv/images/ldr-Drago.jpg", 255*ldrDrago)
plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrDrago, 0, 1)[:,:,::-1]);plt.axis("off")
plt.show()
# Tonemaping with Reinhard's method
print("Tonemaping using Reinhard's method ... ")
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0.0, 0.0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
cv2.imwrite("review_opencv/images/ldr-Reinhard.jpg", 255*ldrReinhard)
plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrReinhard, 0, 1)[:,:,::-1]);plt.axis("off")
plt.show()
# Tonemaping with Mantiuk's method
print("Tonemaping using Mantiuk's method ... ")
tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk # Scale the image to [0, 1]
cv2.imwrite("review_opencv/images/ldr-Mantiuk.jpg", 255*ldrMantiuk)
plt.figure(figsize=(20, 10));plt.imshow(np.clip(ldrMantiuk, 0, 1)[:,:,::-1]);plt.axis("off")
plt.show()