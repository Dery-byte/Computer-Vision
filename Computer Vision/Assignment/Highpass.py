import cv2
path="./barbara.png"
path2 = "./image.jpg"
img = cv2.imread(path2)
img = cv2.resize(img, (680, 520), interpolation=cv2.INTER_CUBIC)

# subtract the original image with the blurred image
# after subtracting add 127 to the total result
hpf =  cv2.GaussianBlur(img, (25, 25), 3) + 127

# display both original image and filtered image
cv2.imshow("Original", img)
cv2.imshow("High Passed Filter", hpf)

# cv2.waitkey is used to display
# the image continuously
# if you provide 1000 instead of 0 then
# image will close in 1sec
# you pass in milli second
cv2.waitKey(0)
cv2.destroyAllWindows()
