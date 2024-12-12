import cv2 as cv
import numpy as np

"""
DANIEL ASARE KYEI
PS/MCS/22/0009
"""

#Creating a noise function to add noise to our image
def add_noise(image, noise_type='gaussian', noise_level=0.1):
    """
    Add noise of the given type and level to the input image
    """
    # for gausian noise
    if noise_type == 'gs':
        h, w, c = image.shape
        noise = np.random.normal(0, noise_level * 255, (h, w, c))
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
   
    #for salt and pepper noise
    elif noise_type == 'snp':
        h, w, c = image.shape
        noise = np.zeros((h, w, c), dtype=np.uint8)
        cv.randu(noise, 0, 255)
        noisy_image = image.copy()
        noisy_image[noise < 255 * noise_level / 2] = 0
        noisy_image[noise > 255 * (1 - noise_level / 2)] = 255
    
    #if input fails
    else:
        raise ValueError('Unknown noise type')
    
    # CONVERT MY 3 CHANNEL IMAGE TO 2 CHANNEL
    noisy_image= cv.cvtColor(noisy_image, cv.COLOR_BGR2GRAY)
    output = noisy_image
    
    return output


def divergence(V):
    # Calculate the gradients of each component of V
    V_grad = np.gradient(V)

    # Compute the divergence
    div = np.sum(V_grad, axis=0)

    return div

def normalize_image(img):
    min = np.min(img)
    max = np.max(img)
    return (img - np.min(min)) / (np.max(max) - np.min(min))

# reading image
image = cv.imread("image2.jpg")

# calling our noise function #for salt and pepper use 'snp'
"""
***** Very important Note*****

1. For salt and Pepper Use 'snp'
2. For gausian noise use 'gs'

*******************************
"""

noisy_img = add_noise(image, noise_type="gs", noise_level=0.03) 

# Normalize the noisy image to [0, 1]
noisy_img = normalize_image(noisy_img)

# Apply gradient decent denoising
tau = 0.1
lbmda = 2
denoised_img = noisy_img.copy()
rangenumber = 15

for _ in range(rangenumber):
    ux, uy = np.gradient(denoised_img)
    denoised_img = denoised_img - tau * (
        denoised_img - noisy_img - lbmda * divergence(ux + uy)
    )
    denoised_img = normalize_image(denoised_img)

# With different regularization parameter

tau = 0.08
lbmda = 1.8
epsilon = 0.1
denoised_img2 = noisy_img.copy()

for _ in range(20):
    ux, uy = np.gradient(denoised_img2)
    regularizer = divergence(ux + uy) / np.sqrt(epsilon**2 + divergence(ux + uy) ** 2)
    denoised_img2 = denoised_img2 - tau * (denoised_img2 - noisy_img - regularizer)
    denoised_img2 = normalize_image(denoised_img2)

#creating a side by side comparison output with numpy
concatenated_image = np.hstack((noisy_img, denoised_img))
concatenated_image2 = np.hstack((noisy_img, denoised_img2))

#Showing Orignal Image
cv.imshow("Original Image", image)
cv.waitKey(0)
cv.destroyAllWindows()

# printing the noisy image
cv.imshow("noisy_image", noisy_img)
cv.waitKey(0)
cv.destroyAllWindows()

#for original parameters from Madam
cv.imshow("Denoised_with_orignal_values", concatenated_image)
cv.waitKey(0)
cv.destroyAllWindows()

# My generated parameters
cv.imshow("Denoised_with_different_values", concatenated_image2)
cv.waitKey(0)
cv.destroyAllWindows()

