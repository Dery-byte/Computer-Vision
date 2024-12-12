import cv2
import numpy as np
def LPF(pathlow):
    # Load the image
    # pathlow = "./image.jpg"
    imgLow = cv2.imread(pathlow, 0)

    imgLow = cv2.resize(imgLow, (450, 250), 1)
    # Define the kernel sizes
    kernel_size_lp = (7, 7)  # Low pass filter kernel size
    # Create the low pass filter kernel
    kernel_lp = np.ones(kernel_size_lp, np.float32) / (kernel_size_lp[0] * kernel_size_lp[1])
    # Apply the filters to the image
    img_lp = cv2.filter2D(imgLow, -1, kernel_lp)
    cv2.imshow("Low Pass Filter", img_lp)

    return img_lp

def HPF(path2):
    # Load the image
    # path2 = "./image2.jpg"
    patimg= cv2.imread(path2, 0)
    patimg = cv2.resize(patimg, (450, 250), 1)
    # Define the kernel sizes
    kernel_size_hp = (3, 3)  # High pass filter kernel size
    # Create the high pass filter kernel
    blur =cv2.GaussianBlur(patimg,(5 ,5 ) ,0)
    kernel_hp = np.zeros(kernel_size_hp, np.float32)
    kernel_hp[1, 1] = -4
    kernel_hp += 1
    # Apply the filters to the image
    img_hp = cv2.filter2D(blur, -1, kernel_hp)

    cv2.imshow("High Pass Filter", img_hp)
    return img_hp


def averageHPF_LPF(x, y):
    average1 = (LPF(y) + HPF(x))//2
    cv2.imshow('Average of the filter', average1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
averageHPF_LPF("./image.jpg", "./image3.jpg")



# OBSERVATION
# after performing the average on the results it is observed that the high pass filter
# was sharper and its pixels intensity was maintained, However the lower pass filter averaged out it rapid intensity an a smoothed image pixels were
# shown in the averaged image. In conclusion both filter preserved their properties.
