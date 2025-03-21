import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image

taiwan_img = cv2.imread("review_opencv/images/2.jpeg", 0)
plt.imshow(taiwan_img, cmap="gray")
print(taiwan_img)
plt.show()