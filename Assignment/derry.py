import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
path = "./image.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Create zeros like M-D array to store the Stretched image
MIN_MAX = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

# run a loop and apply the MIN-MAX formulae
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        MIN_MAX[i, j] = (255 * (img[i, j] - np.min(img)) / (np.max(img) - np.min(img)))

# Display the stretched image

cv2.imshow("Contrast Stretched Image", MIN_MAX)
cv2.waitKey(6000)
cv2.destroyWindow("Constrast Stretching using OpenCV")
