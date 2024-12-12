import cv2
import numpy as np

# Load image
path="./image.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (500,500))

# Apply low pass filter with a kernel size of 5x5
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Display the input and filtered images side by side
combined = np.hstack((img, blur))
cv2.imshow('Original vs Filtered', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()