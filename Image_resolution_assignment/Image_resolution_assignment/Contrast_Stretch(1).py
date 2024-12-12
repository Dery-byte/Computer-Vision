import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretch(path):
    path = './lena.bmp'
    img = cv2.imread(path)

    # half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    # The values of xp and fp can be varied to create custom tables as required
    # and it will stretch the contrast even if min and max pixels are 0 and 255
    xp = [0, 70, 90, 180, 200]
    fp = [0, 30, 40, 120, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    # cv2.LUT will replace the values of the original image with the values in the
    # table. For example, all the pixels having values 1 will be replaced by 0 and
    # all pixels having values 4 will be replaced by 1.
    img = cv2.LUT(img, table)

    cv2.imshow("Contrast Stretched Image", img)
    cv2.waitKey(0)


contrast_stretch('./lena.bmp')
