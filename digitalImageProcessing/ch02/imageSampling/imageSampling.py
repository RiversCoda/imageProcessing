import cv2
import numpy as np

# import picture
img = cv2.imread('digitalImageProcessing\ch02\imageSampling\input.png')

# get the height and width of the picture
height, width = img.shape[:2]

# sampling rate
x = 10

# create a new picture
new_img = np.zeros((height, width, 3), np.uint8)
new_img2 = np.zeros((height, width, 3), np.uint8)

# average sampling
for i in range(0, height, x):
    for j in range(0, width, x):
        new_img[i:i+x,j:j+x] = np.mean(np.mean(img[i:i+x, j:j+x], axis = 0), axis = 0)

# jump sampling
for i in range(0, height, x):
    for j in range(0, width, x):
        new_img2[i:i+x, j:j+x] = img[i, j]

# save the pictures
cv2.imwrite('digitalImageProcessing\ch02\imageSampling\output.png', new_img)
cv2.imwrite('digitalImageProcessing\ch02\imageSampling\output2.png', new_img2)

# show the picture and sampled pictures
cv2.imshow('image', img)
cv2.imshow('new_image', new_img)
cv2.imshow('new_image2', new_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
