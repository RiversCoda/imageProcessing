import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('digitalImageProcessing\ch02\grayScaleInvert\input.png')

# 将图像灰度取反
img = 255 - img

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('output')

plt.show()

# 保存图像
cv2.imwrite('digitalImageProcessing\ch02\grayScaleInvert\output.png', img)
