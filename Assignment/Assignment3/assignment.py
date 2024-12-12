import cv2 as cv
import numpy as np
from skimage import data
import matplotlib.pyplot as plt


def sp_noise(image, prob):
    """
    Add salt and pepper noise to image
    Noise_prob: Probability of the noise
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype="uint8")
            white = np.array([255, 255, 255], dtype="uint8")
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype="uint8")
            white = np.array([255, 255, 255, 255], dtype="uint8")

    Noise_prob = np.random.random(output.shape[:2])
    output[Noise_prob < (prob / 2)] = black
    output[Noise_prob > 1 - (prob / 2)] = white

    return output


def divergence(V):
    # Find the gradients of each component of V
    V_grad = np.gradient(V)

    # Calculate the divergence
    div = np.sum(V_grad, axis=0)

    return div

def normalizedImage(img):
    min = np.min(img)
    max = np.max(img)
    return (img - np.min(min)) / (np.max(max) - np.min(min))

path ="image2.jpg"
image = cv.imread(path)
noisy_img = sp_noise(image, 0.3) # adding noise
# Normalize the noisy image to [0, 1]
noisy_img = normalizedImage(noisy_img)
plt.imshow(noisy_img, cmap="gray")
plt.imshow(image, cmap="gray")

plt.show()

# Apply gradient decent denoising
tau = 0.05
lbmda = 3
denoised_img = noisy_img.copy()

for _ in range(20):
    ux, uy = np.gradient(denoised_img)
    denoised_img = denoised_img - tau * (
        denoised_img - noisy_img - lbmda * divergence(ux + uy)
    )
    denoised_img = normalizedImage(denoised_img)

plt.imshow(denoised_img, cmap="gray")
plt.show()
# With different regularization parameter
lbmda = 3
tau = 0.05
epsilon = 0.01
denoised_img2 = noisy_img.copy()

for _ in range(20):
    ux, uy = np.gradient(denoised_img2)
    regularizer = divergence(ux + uy) / np.sqrt(epsilon**2 + divergence(ux + uy) ** 2)
    denoised_img2 = denoised_img2 - tau * (denoised_img2 - noisy_img - regularizer)
    denoised_img2 = normalizedImage(denoised_img2)

plt.imshow(denoised_img2, cmap="gray")
plt.imshow(image, cmap="gray")
plt.show()

