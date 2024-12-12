import cv2
import numpy as np
from matplotlib import pyplot as plt
def grayimage(images):
    # Load the input image
    image = cv2.imread(images)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    # Use the cvtColor() function to grayscale the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Gray-scaled Image', gray_image)
    cv2.imwrite("grayscaledImage.bmp", gray_image)

    cv2.waitKey(0)
    # Window shown waits for any key pressing event
    cv2.destroyAllWindows()
grayimage('./lena.bmp')


def changeIntensity():
    pass
    changeIntensity()

def rReduction(image):
    # Load the original image
    image = cv2.imread("./peppers.png")
    # Resolution reduction
    width = int(image.shape[1] * 0.5)
    height = int(image.shape[0] * 0.5)
    dim = (width, height)
    # Perform spatial resolution using cv2.INTER_AREA interpolation method
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # Save the resized image
    cv2.imwrite("reduced.jpg", resized)
rReduction(image='./lena.bmp')

def rIncrement(image):
    # Load the original image
    image = cv2.imread("./peppers.png")
    # Resolution increment
    width = int(image.shape[1] * 3)
    height = int(image.shape[0] * 2)
    dim = (width, height)
    # Perform spatial resolution increment using cv2.INTER_AREA interpolation method
    increased_size = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # Save the resized image
    cv2.imwrite("increased.jpg", increased_size)

resized =rIncrement('./lena.bmp')


# bat.jpg is the batman image.
img = cv2.imread('./lena.bmp')

# make sure that you have saved it in the same folder
# You can change the kernel size as you want
blurImg = cv2.blur(img,(10,10))
cv2.imshow('blurred image',blurImg)
cv2.waitKey(0)
cv2.destroyAllWindows()





