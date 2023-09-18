import numpy as np
import matplotlib.pyplot as plt
import cv2

def four_pixel_mean_blur(image):
    output = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            count = 1
            s = float(image[i, j])
            if i > 0:
                s += image[i-1, j]
                count += 1
            if i < h-1:
                s += image[i+1, j]
                count += 1
            if j > 0:
                s += image[i, j-1]
                count += 1
            if j < w-1:
                s += image[i, j+1]
                count += 1
            output[i, j] = s/count
    return np.uint8(output)

def eight_pixel_mean_blur(image):
    output = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            count = 1
            s = float(image[i, j])
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in directions:
                if 0 <= i + dx < h and 0 <= j + dy < w:
                    s += image[i + dx, j + dy]
                    count += 1
            output[i, j] = s/count
    return np.uint8(output)

def custom_gaussian_blur(image, sigma=1):
    # 创建一个3x3的高斯核
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    kernel = kernel / 16.0  # 标准化以使其和为1

    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            local_values = image[i-1:i+2, j-1:j+2]
            output[i, j] = np.sum(local_values * kernel)
            
    return np.uint8(output)

image_path = 'digitalImageProcessing/ch02/stackNoiseReduction/input.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

four_pixel_blurred = four_pixel_mean_blur(image)
eight_pixel_blurred = eight_pixel_mean_blur(image)
eight_pixel_gaussian = custom_gaussian_blur(image)

# 显示图片
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(four_pixel_blurred, cmap='gray')
plt.title('4-Pixel Mean Blur')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(eight_pixel_blurred, cmap='gray')
plt.title('8-Pixel Mean Blur')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(eight_pixel_gaussian, cmap='gray')
plt.title('8-Pixel Gaussian Blur')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存图片
cv2.imwrite('digitalImageProcessing\ch02\\blur\\four_pixel_blurred.png', four_pixel_blurred)
cv2.imwrite('digitalImageProcessing\ch02\\blur\\eight_pixel_blurred.png', eight_pixel_blurred)
cv2.imwrite('digitalImageProcessing\ch02\\blur\\gaussian_blurred.png', eight_pixel_gaussian)
