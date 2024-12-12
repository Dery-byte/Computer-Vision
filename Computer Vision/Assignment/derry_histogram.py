import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
path = "./image.jpg"
image = cv2.imread(path, 0)

# Calculate the histogram
histogram, bins = np.histogram(image, bins=256, range=(0, 256))

# Plotting the histogram
cdf = histogram.cumsum()
normalized = cdf * histogram.max() / cdf.max()

plt.plot(normalized, color="purple")
plt.hist(image.flatten(), bins=256, range=(0, 256), color="g")
plt.xlim([0, 256])
plt.legend(("cdf", "Histogram"), loc="upper right")
plt.show()
