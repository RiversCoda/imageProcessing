# test grayscale bit in range 2^1 to 2^8
import cv2
import numpy as np

# import picture
img = cv2.imread('digitalImageProcessing\ch02\grayscaleBit\input.png', cv2.IMREAD_GRAYSCALE)

# get the height and width of the picture
height, width = img.shape[:2]

# create 8 new picture
img1 = np.zeros((height, width, 3), np.uint8)
img2 = np.zeros((height, width, 3), np.uint8)
img4 = np.zeros((height, width, 3), np.uint8)
img8 = np.zeros((height, width, 3), np.uint8)

# in range 1, 2, 4 ,8
imgs = [img1, img2, img4, img8]

bits = 2
for i in imgs:
    for j in range(height):
        for k in range(width):
            parts = bits - 1
            length = 256 / parts
            halfLength = length / 2
            level = img[j, k] // length
            if img[j, k] > level * length + halfLength:
                i[j, k] = (level + 1) * length - 1
            else:
                i[j, k] = level * length
    bits = bits * 2

# save the picture
cv2.imwrite('digitalImageProcessing\ch02\grayscaleBit\img1.png', img1)
cv2.imwrite('digitalImageProcessing\ch02\grayscaleBit\img2.png', img2)
cv2.imwrite('digitalImageProcessing\ch02\grayscaleBit\img4.png', img4)
cv2.imwrite('digitalImageProcessing\ch02\grayscaleBit\img8.png', img8)

# show the picture
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img4', img4)
cv2.imshow('img8', img8)

cv2.waitKey(0)
cv2.destroyAllWindows()
