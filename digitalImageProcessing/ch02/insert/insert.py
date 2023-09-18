import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_neighbor_resize(image, fx, fy):
    height, width, channels = image.shape
    new_height, new_width = int(height * fy), int(width * fx)

    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            iy = int(y / fy)
            ix = int(x / fx)
            resized[y, x] = image[iy, ix]

    return resized

def bilinear_resize(image, fx, fy):
    height, width, channels = image.shape
    new_height, new_width = int(height * fy), int(width * fx)

    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            i = y / fy
            j = x / fx

            i0 = int(i)
            j0 = int(j)

            i1 = min(i0 + 1, height - 1)
            j1 = min(j0 + 1, width - 1)

            a = i - i0
            b = j - j0
            # 按距离加权求值
            resized[y, x] = (1 - a) * (1 - b) * image[i0, j0] + a * (1 - b) * image[i1, j0] + (1 - a) * b * image[i0, j1] + a * b * image[i1, j1]

    return resized

def cubic_interpolation(p0, p1, p2, p3, t):
    p0, p1, p2, p3 = float(p0), float(p1), float(p2), float(p3)

    a0 = p3 - p2 - p0 + p1
    a1 = p0 - p1 - a0
    a2 = p2 - p0
    a3 = p1
    return a0*t**3 + a1*t**2 + a2*t + a3

def bicubic_resize(image, fx, fy):

    height, width, channels = image.shape
    new_height, new_width = int(height * fy), int(width * fx)

    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            i = y / fy
            j = x / fx

            i0 = max(int(i) - 1, 0)
            i1 = int(i)
            i2 = min(i1 + 1, height - 1)
            i3 = min(i1 + 2, height - 1)

            j0 = max(int(j) - 1, 0)
            j1 = int(j)
            j2 = min(j1 + 1, width - 1)
            j3 = min(j1 + 2, width - 1)

            a = i - i1
            b = j - j1

            for channel in range(channels):
                val = cubic_interpolation(
                    cubic_interpolation(image[i0, j0, channel], image[i1, j0, channel], image[i2, j0, channel], image[i3, j0, channel], a),
                    cubic_interpolation(image[i0, j1, channel], image[i1, j1, channel], image[i2, j1, channel], image[i3, j1, channel], a),
                    cubic_interpolation(image[i0, j2, channel], image[i1, j2, channel], image[i2, j2, channel], image[i3, j2, channel], a),
                    cubic_interpolation(image[i0, j3, channel], image[i1, j3, channel], image[i2, j3, channel], image[i3, j3, channel], a),
                    b
                )
                # Clip value to range [0, 255] and convert back to np.uint8
                resized[y, x, channel] = np.clip(int(val), 0, 255)

    return resized

image = cv2.imread('digitalImageProcessing\ch02\insert\input.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for visualization with plt

resized_nn = nearest_neighbor_resize(image, 2, 2)
resized_bilinear = bilinear_resize(image, 2, 2)
resized_bicubic = bicubic_resize(image, 2, 2)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(resized_nn)
plt.title('Nearest Neighbor')

plt.subplot(1,3,2)
plt.imshow(resized_bilinear)
plt.title('Bilinear')

plt.subplot(1,3,3)
plt.imshow(resized_bicubic)
plt.title('Bicubic')

plt.show()
